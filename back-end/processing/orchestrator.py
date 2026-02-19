"""
AI Processing Orchestrator
Coordinates all AI modules (audio extraction, transcription, translation)
"""
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)


class AIProcessingOrchestrator:
    """
    Orchestrates the complete AI video processing pipeline.
    Coordinates: Audio Extraction → Speech-to-Speech Translation → Voice Cloning
    """
    
    def __init__(self, video_instance):
        """
        Initialize orchestrator with a Video instance.
        
        Args:
            video_instance: videos.models.Video instance
        """
        self.video = video_instance
        self.processing_dir = Path(settings.MEDIA_ROOT) / 'processing' / str(self.video.id)
        self.results = {
            'audio_extraction': None,
            's2s_translation': None,
            'voice_cloning': None
        }
    
    def process_video(self):
        """
        Process video through complete AI pipeline.
        
        Returns:
            dict: Final results from all modules
        """
        try:
            logger.info(f"Starting AI processing for video {self.video.id}")
            
            # Update video status
            self.video.status = 'processing'
            self.video.progress = 0
            self.video.save()
            
            # Module 1: Audio Extraction
            audio_result = self._extract_audio()
            if not audio_result['success']:
                raise Exception(f"Audio extraction failed: {audio_result.get('error')}")
            
            self.results['audio_extraction'] = audio_result
            logger.info(f"Audio extraction completed: {audio_result['audio_path']}")
            
            # NEW: Module 2: Speech-to-Speech Translation (replaces transcription + translation + voice cloning)
            s2s_result = self._translate_audio_s2s(
                audio_result['audio_path'],
                self.video.original_video.path
            )
            if not s2s_result['success']:
                # Attempt fallback to old pipeline if S2S fails
                logger.warning("S2S translation failed, attempting fallback to old pipeline...")
                s2s_result = self._fallback_to_old_pipeline(audio_result['audio_path'])
                
                if not s2s_result['success']:
                    raise Exception(f"Translation failed: {s2s_result.get('error')}")
            
            self.results['s2s_translation'] = s2s_result
            logger.info(f"S2S Translation completed: {s2s_result.get('duration', 0):.2f}s audio generated")
            logger.info(f"Final cloned audio path: {s2s_result['translated_audio_path']}")
            
            # Mark as completed
            self.video.status = 'completed'
            self.video.progress = 100
            self.video.save()
            
            logger.info(f"AI processing completed successfully for video {self.video.id}")
            
            return {
                'success': True,
                'video_id': str(self.video.id),
                'results': self.results,
                'message': 'All AI modules completed successfully'
            }
            
        except Exception as e:
            error_msg = f"AI processing failed: {str(e)}"
            logger.error(error_msg)
            
            self.video.status = 'failed'
            self.video.error_message = error_msg
            self.video.save()
            
            return {
                'success': False,
                'error': error_msg,
                'results': self.results
            }
    
    def _extract_audio(self):
        """
        Module 1: Extract audio from video using processing service.
        """
        try:
            from processing.services import VideoProcessingService
            
            logger.info("Module 1: Starting audio extraction...")
            service = VideoProcessingService(self.video)
            result = service.extract_audio_from_video()
            
            if result['success']:
                logger.info(f"Module 1 completed: Audio extracted to {result['audio_path']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Module 1 failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _translate_audio_s2s(self, audio_path, video_path):
        """
        NEW Module 2: Speech-to-Speech Translation using Seamless M4T.
        Replaces the old pipeline: transcription → translation → voice cloning.
        
        Args:
            audio_path: Path to extracted audio
            video_path: Path to original video (for duration matching)
        
        Returns:
            dict: Translation result with translated audio path
        """
        try:
            from s2s_translator.services import SeamlessTranslationEngine
            import os
            from django.conf import settings
            
            logger.info("NEW Module 2: Starting Speech-to-Speech translation with Seamless M4T...")
            
            # Update progress
            self.video.progress = 20
            self.video.save()
            
            # Determine target language
            target_lang = self.video.target_language if self.video.target_language else 'urd'
            
            # Map common language codes to Seamless M4T codes
            lang_mapping = {
                'en': 'eng',
                'ur': 'urd',
                'english': 'eng',
                'urdu': 'urd'
            }
            target_lang = lang_mapping.get(target_lang.lower(), target_lang)
            
            logger.info(f"Target language: {target_lang}")
            
            # Initialize S2S translation engine
            model_path = os.path.join(settings.BASE_DIR.parent, "models", "seamless-m4t-v2-large")
            
            from videos.models import ProcessingLog
            ProcessingLog.objects.create(
                video=self.video,
                step='s2s_translation_init',
                level='info',
                message=f'Initializing Seamless M4T for S2S translation to {target_lang}'
            )
            
            engine = SeamlessTranslationEngine(
                model_path=model_path,
                target_lang=target_lang
            )
            
            self.video.progress = 30
            self.video.save()
            
            # Perform S2S translation
            ProcessingLog.objects.create(
                video=self.video,
                step='s2s_translation_process',
                level='info',
                message='Translating audio with speech-to-speech model...'
            )
            
            result = engine.translate_audio(
                source_audio_path=audio_path,
                original_video_path=video_path
            )
            
            # Cleanup engine resources
            engine.cleanup()
            
            if result['success']:
                self.video.progress = 70
                self.video.save()
                
                logger.info(f"S2S translation successful: {result['translated_audio_path']}")
                logger.info(f"Duration: {result['duration']:.2f}s")
                
                ProcessingLog.objects.create(
                    video=self.video,
                    step='s2s_translation_complete',
                    level='info',
                    message=f'S2S translation completed. Duration: {result["duration"]:.2f}s'
                )
                
                # ================================================================
                # VOICE CLONING: Clone original speaker's voice onto translated audio
                # ================================================================
                logger.info("Starting voice cloning with OpenVoice v2...")
                
                ProcessingLog.objects.create(
                    video=self.video,
                    step='voice_cloning_start',
                    level='info',
                    message='Cloning original speaker voice onto translated audio'
                )
                
                try:
                    from s2s_translator.voice_cloner import apply_voice_cloning
                    
                    self.video.progress = 75
                    self.video.save()
                    
                    # Apply voice cloning with DYNAMIC SEGMENTATION
                    # No longer passing speech_segments - VAD runs on translated audio
                    cloned_audio_path = str(Path(result['translated_audio_path']).parent / "cloned_audio.wav")
                    
                    logger.info(f"🎙️ APPLYING VOICE CLONING (Dynamic Segmentation):")
                    logger.info(f"  Original audio: {audio_path}")
                    logger.info(f"  Translated audio: {result['translated_audio_path']}")
                    logger.info(f"  Output (cloned): {cloned_audio_path}")
                    logger.info(f"  Mode: VAD on TRANSLATED audio (decoupled from original)")
                    logger.info(f"  Hardware: M2 Neural Engine (MPS + float16)")
                    logger.info(f"  Quality: 30ms crossfade + soxr_hq resampling")
                    
                    cloned_audio = apply_voice_cloning(
                        original_audio_path=audio_path,
                        translated_audio_path=result['translated_audio_path'],
                        output_path=cloned_audio_path
                    )
                    
                    logger.info(f"✓ Voice cloning completed: {cloned_audio_path}")
                    logger.info(f"✓ Cloned audio file exists: {Path(cloned_audio_path).exists()}")
                    
                    ProcessingLog.objects.create(
                        video=self.video,
                        step='voice_cloning_complete',
                        level='info',
                        message='Voice cloning completed successfully'
                    )
                    
                    # Store both paths - cloned is the final output
                    self.video.translated_audio_path = cloned_audio_path  # Use cloned audio
                    self.video.progress = 80
                    self.video.save()
                    
                    logger.info(f"💾 SAVED TO DATABASE:")
                    logger.info(f"  video.translated_audio_path = {self.video.translated_audio_path}")
                    logger.info(f"  This will be served to download button!")
                    logger.info(f"Module 2 completed: Translation + Voice Cloning successful")
                    
                    # Module 3: Lip Synchronization (80-100%)
                    lipsync_result = self._apply_lip_sync(
                        source_video=self.video.original_video.path,
                        translated_audio=cloned_audio_path
                    )
                    
                    if lipsync_result['success']:
                        logger.info(f"✓ Module 3 (Lip Sync) completed: {lipsync_result['output_path']}")
                        
                        return {
                            'success': True,
                            'translated_audio_path': cloned_audio_path,
                            'processed_video_path': lipsync_result['output_path'],
                            'raw_translated_path': result['translated_audio_path'],
                            'duration': result['duration'],
                            'process_id': result.get('process_id'),
                            'method': 's2s_seamless_with_cloning_and_lipsync'
                        }
                    else:
                        logger.warning(f"Lip sync failed: {lipsync_result.get('error')}")
                        logger.warning("Continuing without lip sync")
                        
                        return {
                            'success': True,
                            'translated_audio_path': cloned_audio_path,
                            'raw_translated_path': result['translated_audio_path'],
                            'duration': result['duration'],
                            'process_id': result.get('process_id'),
                            'method': 's2s_seamless_with_cloning',
                            'warning': 'Lip sync failed, video not generated'
                        }
                    
                    
                except Exception as clone_error:
                    logger.error(f"Voice cloning failed: {str(clone_error)}")
                    logger.warning("Falling back to translated audio without cloning")
                    
                    ProcessingLog.objects.create(
                        video=self.video,
                        step='voice_cloning_failed',
                        level='warning',
                        message=f'Voice cloning failed: {str(clone_error)}. Using translated audio.'
                    )
                    
                    # Fallback: use translated audio without cloning
                    self.video.translated_audio_path = result['translated_audio_path']
                    self.video.progress = 80
                    self.video.save()
                    
                    return {
                        'success': True,
                        'translated_audio_path': result['translated_audio_path'],
                        'duration': result['duration'],
                        'process_id': result.get('process_id'),
                        'method': 's2s_seamless',
                        'warning': 'Voice cloning failed, using raw translation'
                    }
            else:
                error_type = result.get('error_type', 'UNKNOWN')
                error_msg = result.get('error', 'S2S translation failed')
                
                logger.error(f"Module 2 failed: {error_msg} (Type: {error_type})")
                
                ProcessingLog.objects.create(
                    video=self.video,
                    step='s2s_translation_failed',
                    level='error',
                    message=f'S2S translation failed: {error_msg}'
                )
                
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': error_type
                }
            
        except Exception as e:
            logger.error(f"Module 2 failed: {str(e)}", exc_info=True)
            
            from videos.models import ProcessingLog
            ProcessingLog.objects.create(
                video=self.video,
                step='s2s_translation_error',
                level='error',
                message=f'S2S translation error: {str(e)}'
            )
            
            return {
                'success': False,
                'error': str(e),
                'error_type': 'EXCEPTION'
            }
    
    def _fallback_to_old_pipeline(self, audio_path):
        """
        Fallback to the old transcription → translation → voice cloning pipeline.
        Used when S2S translation fails (e.g., due to OOM errors).
        
        Args:
            audio_path: Path to extracted audio
        
        Returns:
            dict: Result from old pipeline
        """
        try:
            logger.warning("=" * 80)
            logger.warning("FALLBACK: Using old transcription + translation + voice cloning pipeline")
            logger.warning("=" * 80)
            
            from videos.models import ProcessingLog
            ProcessingLog.objects.create(
                video=self.video,
                step='fallback_pipeline',
                level='warning',
                message='S2S translation failed, falling back to old pipeline'
            )
            
            # Module 2 (Old): Transcription
            transcription_result = self._transcribe_audio(audio_path)
            if not transcription_result['success']:
                raise Exception(f"Transcription failed: {transcription_result.get('error')}")
            
            self.results['transcription'] = transcription_result
            logger.info(f"Transcription completed: {transcription_result['language']}")
            
            # Module 3 (Old): Translation
            translation_result = self._translate_text(
                transcription_result['text'],
                transcription_result['language'],
                self.video.target_language,
                transcription_result.get('segments')
            )
            if not translation_result['success']:
                raise Exception(f"Translation failed: {translation_result.get('error')}")
            
            self.results['translation'] = translation_result
            logger.info(f"Translation completed: {translation_result['source_language']}→{translation_result['target_language']}")
            
            # Module 4 (Old): Voice Cloning
            voice_cloning_result = self._clone_voice(
                translation_result['translated_text'],
                audio_path,
                translation_result['target_language'],
                transcription_result,
                translation_result
            )
            if not voice_cloning_result['success']:
                raise Exception(f"Voice cloning failed: {voice_cloning_result.get('error')}")
            
            self.results['voice_cloning'] = voice_cloning_result
            logger.info(f"Voice cloning completed: {voice_cloning_result['cloned_audio_path']}")
            
            ProcessingLog.objects.create(
                video=self.video,
                step='fallback_pipeline',
                level='info',
                message='Fallback pipeline completed successfully'
            )
            
            # Return in S2S-compatible format
            return {
                'success': True,
                'translated_audio_path': voice_cloning_result['cloned_audio_path'],
                'duration': voice_cloning_result.get('total_duration', 0),
                'method': 'fallback_old_pipeline',
                'fallback': True
            }
            
        except Exception as e:
            logger.error(f"Fallback pipeline also failed: {str(e)}", exc_info=True)
            
            from videos.models import ProcessingLog
            ProcessingLog.objects.create(
                video=self.video,
                step='fallback_pipeline_failed',
                level='error',
                message=f'Fallback pipeline failed: {str(e)}'
            )
            
            return {
                'success': False,
                'error': f"Both S2S and fallback pipelines failed: {str(e)}",
                'error_type': 'FALLBACK_FAILED'
            }
    
    def _transcribe_audio(self, audio_path):
        """
        Module 2 (DEPRECATED - Old pipeline): Transcribe audio to text using AssemblyAI service.
        Now only used as fallback when S2S translation fails.
        """
        try:
            from transcription.services_assemblyai import AssemblyAITranscriptionService
            
            logger.info("Module 2: Starting transcription with AssemblyAI...")
            service = AssemblyAITranscriptionService(self.video)
            result = service.transcribe_video_audio(audio_path)
            
            if result['success']:
                logger.info(f"Module 2 completed: Transcribed {len(result['text'])} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Module 2 failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _translate_text(self, text, source_lang, target_lang, segments=None):
        """
        Module 3 (DEPRECATED - Old pipeline): Translate text using Google Gemini API service.
        Now only used as fallback when S2S translation fails.
        """
        try:
            from translation.services_translate import DeepLTranslationService
            
            logger.info(f"Module 3: Starting Gemini translation {source_lang}→{target_lang}...")
            service = DeepLTranslationService(self.video)
            result = service.translate_transcription(
                text,
                source_lang,
                target_lang,
                segments=segments
            )
            
            if result['success']:
                logger.info(f"Module 3 completed: Translated {len(result['translated_text'])} characters with Gemini")
                
                # Log usage info if available
                if 'usage_info' in result and result['usage_info']:
                    usage = result['usage_info']
                    logger.info(
                        f"Gemini translation: {usage['characters_used']:,} characters "
                        f"({usage.get('percentage_used', 0):.1f}% of quota)"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Module 3 failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _clone_voice(self, translated_text, original_audio_path, target_language, transcription_result=None, translation_result=None):
        """
        Module 4 (DEPRECATED - Old pipeline): Clone voice using OpenVoice service with timing preservation.
        Now only used as fallback when S2S translation fails.
        
        Extracts voice embedding, synthesizes translation with speaker's voice.
        
        Args:
            translated_text: Translated text to synthesize
            original_audio_path: Path to original extracted audio
            target_language: Target language code
            transcription_result: Original transcription result with segments
            translation_result: Translation result with segment-level translations
        
        Returns:
            dict: Voice cloning result
        """
        try:
            from voice_cloning.services_openvoice import OpenVoiceService
            
            logger.info(f"Module 4: Starting OpenVoice voice cloning for {target_language}...")
            
            # Update video status
            self.video.progress = 75
            self.video.save()
            
            # Initialize OpenVoice service
            service = OpenVoiceService(self.video)
            
            # Update progress
            self.video.progress = 76
            self.video.save()
            
            # Step 1: Extract voice embedding from original audio
            logger.info("Extracting speaker voice embedding...")
            embedding_result = service.extract_voice_embedding(original_audio_path)
            
            if not embedding_result['success']:
                raise Exception(f"Voice embedding extraction failed: {embedding_result.get('error')}")
            
            speaker_embedding = embedding_result['embedding']
            logger.info("Voice embedding extracted successfully")
            
            # Update progress
            self.video.progress = 78
            self.video.save()
            
            # Step 2: Validate segment data for timing preservation
            if not transcription_result or 'segments' not in transcription_result:
                raise Exception("Transcription segments are required for timing preservation")
            
            if not translation_result or 'translated_segments' not in translation_result:
                raise Exception("Translation segments are required for timing preservation")
            
            logger.info(f"Using segment-by-segment voice synthesis with timing preservation")
            
            from videos.models import ProcessingLog
            ProcessingLog.objects.create(
                video=self.video,
                step='voice_cloning',
                level='info',
                message=f'Starting segment synthesis: {len(translation_result["translated_segments"])} segments'
            )
            
            # Step 3: Clone voice with timing preservation
            result = service.clone_voice_with_timing(
                transcription_segments=transcription_result['segments'],
                translated_segments=translation_result['translated_segments'],
                original_audio_path=original_audio_path,
                target_language=target_language,
                speaker_embedding=speaker_embedding
            )
            
            if not result['success']:
                raise Exception(f"Voice cloning failed: {result.get('error')}")
            
            # Update progress to 85%
            self.video.progress = 85
            self.video.save()
            
            logger.info(f"Module 4 completed: {result['segments_processed']} segments processed")
            
            ProcessingLog.objects.create(
                video=self.video,
                step='voice_cloning',
                level='info',
                message=f'Voice cloning completed: {result["segments_processed"]} segments'
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Module 4 failed: {str(e)}", exc_info=True)
            
            from videos.models import ProcessingLog
            ProcessingLog.objects.create(
                video=self.video,
                step='voice_cloning',
                level='error',
                message=f'Voice cloning failed: {str(e)}'
            )
            
            return {
                'success': False,
                'error': str(e)
            }
    
    
    def _apply_lip_sync(self, source_video: str, translated_audio: str) -> dict:
        """
        Module 3: Apply lip synchronization using Wav2Lip
        
        Args:
            source_video: Path to original video
            translated_audio: Path to voice-cloned translated audio
            
        Returns:
            dict: Result with success status and output path
        """
        try:
            from lip_sync.lip_sync_service import LipSyncService
            from videos.models import ProcessingLog
            
            logger.info("\n" + "="*80)
            logger.info("MODULE 3: LIP SYNCHRONIZATION")
            logger.info("="*80)
            
            # Update progress to 80%
            self.video.progress = 80
            self.video.save()
            
            ProcessingLog.objects.create(
                video=self.video,
                step='lip_sync_started',
                level='info',
                message='Starting lip synchronization with Wav2Lip...'
            )
            
            # Initialize Lip Sync Service
            lipsync_service = LipSyncService()
            
            # Define output path for final video
            output_video_path = self.processing_dir / "final_video.mp4"
            
            # Update progress
            self.video.progress = 85
            self.video.save()
            
            # Process: Merge audio + Apply lip sync
            result = lipsync_service.process_video(
                source_video_path=source_video,
                translated_audio_path=translated_audio,
                output_video_path=str(output_video_path),
                temp_dir=str(self.processing_dir / "temp")
            )
            
            if result['success']:
                # Keep the video in processing directory and store the relative path
                # The file is already at: media/processing/{uuid}/final_video.mp4
                relative_path = f"processing/{self.video.id}/final_video.mp4"
                
                # Store the path directly (file already exists at correct location)
                self.video.processed_video.name = relative_path
                self.video.progress = 100
                self.video.status = 'completed'
                # Important: Save to commit the processed_video field to database
                self.video.save(update_fields=['processed_video', 'progress', 'status'])
                
                logger.info(f"✓ Video path saved to database: {relative_path}")
                logger.info(f"✓ Full path: {output_video_path}")
                logger.info(f"✓ processed_video field: {self.video.processed_video}")
                logger.info(f"✓ processed_video.url: {self.video.processed_video.url if self.video.processed_video else 'None'}")
                
                ProcessingLog.objects.create(
                    video=self.video,
                    step='lip_sync_complete',
                    level='info',
                    message=f'Lip synchronization completed. File size: {result.get("file_size_mb", 0):.2f} MB'
                )
                
                logger.info(f"✓ Lip sync completed: {output_video_path}")
                logger.info(f"✓ Video saved to database: {self.video.processed_video}")
                
                return {
                    'success': True,
                    'output_path': str(output_video_path),
                    'file_size_mb': result.get('file_size_mb', 0)
                }
            else:
                logger.error(f"Lip sync failed: {result.get('error')}")
                
                # Mark as completed but without lip sync
                self.video.progress = 80
                self.video.status = 'completed'
                self.video.save()
                
                ProcessingLog.objects.create(
                    video=self.video,
                    step='lip_sync_failed',
                    level='error',
                    message=f'Lip sync failed: {result.get("error")}'
                )
                
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Lip sync module failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Mark as completed without lip sync
            self.video.progress = 80
            self.video.status = 'completed'
            self.video.save()
            
            ProcessingLog.objects.create(
                video=self.video,
                step='lip_sync_error',
                level='error',
                message=f'Lip sync error: {str(e)}'
            )
            
            return {
                'success': False,
                'error': str(e)
            }
    
    # Voice cloning + Lip sync complete the pipeline
    # Lip sync will be implemented later after voice cloning is fully functional
    

def start_ai_processing(video_id):
    """
    Start AI processing for a video (can be called from view or background task).
    
    Args:
        video_id: UUID of the video to process
    
    Returns:
        dict: Processing result
    """
    from videos.models import Video
    
    try:
        video = Video.objects.get(id=video_id)
        orchestrator = AIProcessingOrchestrator(video)
        return orchestrator.process_video()
    except Video.DoesNotExist:
        return {
            'success': False,
            'error': f'Video not found: {video_id}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
