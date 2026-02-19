"""
Views for Speech-to-Speech Translation
"""

import os
import logging
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from videos.models import Video, ProcessingLog
from .serializers import TranslationRequestSerializer, TranslationStatusSerializer
from .services import SeamlessTranslationEngine

logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_s2s_translation(request, video_id):
    """
    Start speech-to-speech translation for a video.
    
    Args:
        video_id: UUID of the video to translate
        
    Request Body:
        target_language (optional): Target language code ('urd' or 'eng')
        
    Returns:
        200: Translation started successfully
        400: Invalid request or video not ready
        404: Video not found
        500: Translation failed
    """
    logger.info(f"Translation request received for video: {video_id}")
    
    # Validate request data
    serializer = TranslationRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {"error": "Invalid request data", "details": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Fetch video
        video = Video.objects.get(id=video_id, user=request.user)
        logger.info(f"Found video: {video.title}")
        
    except Video.DoesNotExist:
        logger.error(f"Video not found: {video_id}")
        return Response(
            {"error": "Video not found or you don't have permission to access it"},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Check video status
    if video.status in ['processing', 'completed']:
        return Response(
            {"error": f"Video is already in '{video.status}' state"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Check if original video exists
    if not video.original_video or not os.path.exists(video.original_video.path):
        logger.error("Original video file not found")
        return Response(
            {"error": "Original video file not found"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Determine target language
    target_lang = serializer.validated_data.get('target_language')
    if not target_lang:
        target_lang = video.target_language if video.target_language else 'urd'
    
    # Update video target language if needed
    if video.target_language != target_lang:
        video.target_language = target_lang
        video.save()
    
    logger.info(f"Target language: {target_lang}")
    
    # Update video status
    video.status = 'processing'
    video.progress = 10
    video.error_message = None
    video.save()
    
    # Log processing start
    ProcessingLog.objects.create(
        video=video,
        step='translation_start',
        level='info',
        message=f'Starting S2S translation to {target_lang}'
    )
    
    try:
        # Step 1: Check for extracted audio
        # Assuming audio extraction is done in a previous step
        # Audio should be in media/processing/{video_id}/extracted_audio.wav
        processing_dir = Path(settings.MEDIA_ROOT) / "processing" / str(video.id)
        extracted_audio_path = processing_dir / "extracted_audio.wav"
        
        if not extracted_audio_path.exists():
            # Try alternative path or extract now
            logger.warning(f"Extracted audio not found at {extracted_audio_path}")
            # For now, we'll use a placeholder - in production, trigger extraction
            extracted_audio_path = _extract_audio_from_video(video.original_video.path, processing_dir)
        
        logger.info(f"Source audio: {extracted_audio_path}")
        
        # Update progress
        video.progress = 20
        video.save()
        
        # Step 2: Initialize translation engine
        model_path = os.path.join(settings.BASE_DIR.parent, "models", "seamless-m4t-v2-large")
        
        ProcessingLog.objects.create(
            video=video,
            step='translation_init',
            level='info',
            message='Initializing Seamless M4T engine'
        )
        
        engine = SeamlessTranslationEngine(
            model_path=model_path,
            target_lang=target_lang
        )
        
        video.progress = 30
        video.save()
        
        # Step 3: Perform translation
        ProcessingLog.objects.create(
            video=video,
            step='translation_process',
            level='info',
            message='Translating audio...'
        )
        
        result = engine.translate_audio(
            source_audio_path=str(extracted_audio_path),
            original_video_path=video.original_video.path
        )
        
        # Step 4: Handle result
        if result['success']:
            video.progress = 70
            video.save()
            
            ProcessingLog.objects.create(
                video=video,
                step='translation_complete',
                level='info',
                message=f'Translation completed. Duration: {result["duration"]:.2f}s'
            )
            
            logger.info(f"✓ Translation successful")
            
            # Step 5: Voice Cloning - Clone original speaker onto translated audio
            try:
                from s2s_translator.voice_cloner import apply_voice_cloning
                
                ProcessingLog.objects.create(
                    video=video,
                    step='voice_cloning_start',
                    level='info',
                    message='Cloning original speaker voice...'
                )
                
                logger.info("Starting voice cloning with OpenVoice v2...")
                
                # Detect speech segments for cloning
                speech_segments = engine._detect_speech_segments(str(extracted_audio_path))
                
                video.progress = 75
                video.save()
                
                # Apply voice cloning
                cloned_audio_path = Path(result['translated_audio_path']).parent / "cloned_audio.wav"
                
                cloned_audio = apply_voice_cloning(
                    original_audio_path=str(extracted_audio_path),
                    translated_audio_path=result['translated_audio_path'],
                    speech_segments=speech_segments,
                    output_path=str(cloned_audio_path)
                )
                
                logger.info(f"✓ Voice cloning completed: {cloned_audio_path}")
                
                ProcessingLog.objects.create(
                    video=video,
                    step='voice_cloning_complete',
                    level='info',
                    message='Voice cloning completed successfully'
                )
                
                # Use cloned audio as final output
                final_audio_path = str(cloned_audio_path)
                final_message = "Translation and voice cloning completed successfully"
                
            except Exception as clone_error:
                logger.error(f"Voice cloning failed: {str(clone_error)}")
                logger.warning("Using translated audio without cloning")
                
                ProcessingLog.objects.create(
                    video=video,
                    step='voice_cloning_failed',
                    level='warning',
                    message=f'Voice cloning failed: {str(clone_error)}. Using raw translation.'
                )
                
                # Fallback to translated audio without cloning
                final_audio_path = result['translated_audio_path']
                final_message = "Translation completed (voice cloning failed, using raw translation)"
            
            # Update video with final audio path
            video.progress = 80
            video.status = 'translated'
            video.save()
            
            translated_audio_rel_path = os.path.relpath(
                final_audio_path,
                settings.MEDIA_ROOT
            )
            
            # Cleanup engine
            engine.cleanup()
            
            return Response({
                "success": True,
                "message": final_message,
                "video": TranslationStatusSerializer(video).data,
                "translated_audio_path": translated_audio_rel_path,
                "duration": result['duration'],
                "process_id": result['process_id']
            }, status=status.HTTP_200_OK)
        
        else:
            # Translation failed
            error_type = result.get('error_type', 'UNKNOWN')
            error_msg = result.get('error', 'Translation failed')
            
            # Check if it's an OOM error for fallback handling
            if error_type == 'OOM':
                video.status = 'translation_failed_oom'
                video.error_message = f"GPU Out of Memory: {error_msg}"
                
                ProcessingLog.objects.create(
                    video=video,
                    step='translation_failed',
                    level='error',
                    message=f'OOM Error: {error_msg}. Fallback required.'
                )
                
                logger.error(f"OOM during translation: {error_msg}")
                
                # Trigger fallback if configured
                fallback_result = _attempt_fallback_translation(video, extracted_audio_path)
                
                if fallback_result['success']:
                    video.status = 'translated'
                    video.progress = 80
                    video.error_message = "Completed using fallback method (lower quality)"
                    video.save()
                    
                    return Response({
                        "success": True,
                        "message": "Translation completed using fallback method",
                        "warning": "Used lower quality translation due to memory constraints",
                        "video": TranslationStatusSerializer(video).data,
                        "translated_audio_path": fallback_result['audio_path']
                    }, status=status.HTTP_200_OK)
            
            else:
                video.status = 'failed'
                video.error_message = error_msg
            
            video.save()
            
            ProcessingLog.objects.create(
                video=video,
                step='translation_failed',
                level='error',
                message=error_msg
            )
            
            # Cleanup engine
            engine.cleanup()
            
            logger.error(f"Translation failed: {error_msg}")
            
            return Response({
                "success": False,
                "error": error_msg,
                "error_type": error_type,
                "video": TranslationStatusSerializer(video).data
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}", exc_info=True)
        
        video.status = 'failed'
        video.error_message = f"Unexpected error: {str(e)}"
        video.save()
        
        ProcessingLog.objects.create(
            video=video,
            step='translation_error',
            level='error',
            message=f'Unexpected error: {str(e)}'
        )
        
        return Response({
            "success": False,
            "error": str(e),
            "video": TranslationStatusSerializer(video).data
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_translation_status(request, video_id):
    """
    Get translation status for a video.
    
    Args:
        video_id: UUID of the video
        
    Returns:
        200: Status retrieved successfully
        404: Video not found
    """
    try:
        video = Video.objects.get(id=video_id, user=request.user)
        
        # Get recent logs
        recent_logs = video.logs.all()[:10]
        logs_data = [
            {
                "step": log.step,
                "level": log.level,
                "message": log.message,
                "timestamp": log.timestamp
            }
            for log in recent_logs
        ]
        
        return Response({
            "video": TranslationStatusSerializer(video).data,
            "logs": logs_data
        }, status=status.HTTP_200_OK)
        
    except Video.DoesNotExist:
        return Response(
            {"error": "Video not found"},
            status=status.HTTP_404_NOT_FOUND
        )


def _extract_audio_from_video(video_path: str, output_dir: Path) -> Path:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted audio
        
    Returns:
        Path to extracted audio file
    """
    from moviepy.editor import VideoFileClip
    
    logger.info(f"Extracting audio from video: {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "extracted_audio.wav"
    
    with VideoFileClip(video_path) as video:
        video.audio.write_audiofile(
            str(audio_path),
            codec='pcm_s16le',
            fps=16000,
            logger=None  # Suppress moviepy logs
        )
    
    logger.info(f"✓ Audio extracted: {audio_path}")
    return audio_path


def _attempt_fallback_translation(video: Video, source_audio_path: Path) -> dict:
    """
    Attempt fallback translation using a lighter method.
    This is triggered when the main translation fails due to OOM.
    
    Args:
        video: Video instance
        source_audio_path: Path to source audio
        
    Returns:
        Dictionary with success status and audio path
    """
    logger.warning("Attempting fallback translation method...")
    
    ProcessingLog.objects.create(
        video=video,
        step='fallback_translation',
        level='warning',
        message='Attempting fallback translation with reduced quality'
    )
    
    try:
        # Option 1: Use CPU instead of GPU
        # Option 2: Use a smaller/faster model
        # Option 3: Use external API service
        
        # For now, we'll use CPU with smaller chunks
        from .services import SeamlessTranslationEngine
        import os
        from django.conf import settings
        
        model_path = os.path.join(settings.BASE_DIR.parent, "models", "seamless-m4t-v2-large")
        
        # Force CPU usage
        import torch
        original_device = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        
        try:
            engine = SeamlessTranslationEngine(
                model_path=model_path,
                target_lang=video.target_language or 'urd'
            )
            
            # Reduce chunk size for CPU processing
            engine.CHUNK_DURATION_MS = 10000  # 10 seconds
            
            result = engine.translate_audio(
                source_audio_path=str(source_audio_path),
                original_video_path=video.original_video.path
            )
            
            engine.cleanup()
            
            if result['success']:
                logger.info("✓ Fallback translation succeeded")
                
                translated_audio_rel_path = os.path.relpath(
                    result['translated_audio_path'],
                    settings.MEDIA_ROOT
                )
                
                return {
                    "success": True,
                    "audio_path": translated_audio_rel_path,
                    "method": "CPU_fallback"
                }
            else:
                logger.error("Fallback translation also failed")
                return {"success": False}
                
        finally:
            # Restore original CUDA availability
            torch.cuda.is_available = original_device
        
    except Exception as e:
        logger.error(f"Fallback translation failed: {str(e)}")
        
        ProcessingLog.objects.create(
            video=video,
            step='fallback_failed',
            level='error',
            message=f'Fallback translation failed: {str(e)}'
        )
        
        return {"success": False}
