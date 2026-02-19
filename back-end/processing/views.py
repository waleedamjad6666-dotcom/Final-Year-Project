"""
Views for video processing module.
Module 1: Audio extraction from video.
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from videos.models import Video
from .services import VideoProcessingService
from .serializers import VideoProcessingSerializer
import logging

logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def extract_audio(request):
    """
    Extract audio from uploaded video.
    This is Module 1 of the AI processing pipeline.
    
    Request body:
        {
            "video_id": "uuid-of-video"
        }
    
    Response:
        {
            "success": true,
            "audio_path": "/path/to/extracted/audio.wav",
            "audio_duration": 120.5,
            "message": "Audio extracted successfully"
        }
    """
    serializer = VideoProcessingSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    video_id = serializer.validated_data['video_id']
    
    try:
        # Get the video instance
        video = Video.objects.get(id=video_id, user=request.user)
        
        # Initialize processing service
        processing_service = VideoProcessingService(video)
        
        # Extract audio from video
        result = processing_service.extract_audio_from_video()
        
        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Video.DoesNotExist:
        return Response(
            {'error': 'Video not found or you do not have permission to access it'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Unexpected error in extract_audio: {str(e)}")
        return Response(
            {'error': f'Unexpected error: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_video_metadata(request):
    """
    Get metadata from uploaded video (duration, resolution, etc.).
    
    Request body:
        {
            "video_id": "uuid-of-video"
        }
    
    Response:
        {
            "success": true,
            "duration": 120.5,
            "resolution": "1920x1080",
            "video_info": {...}
        }
    """
    serializer = VideoProcessingSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    video_id = serializer.validated_data['video_id']
    
    try:
        # Get the video instance
        video = Video.objects.get(id=video_id, user=request.user)
        
        # Initialize processing service
        processing_service = VideoProcessingService(video)
        
        # Get video metadata
        result = processing_service.get_video_metadata()
        
        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Video.DoesNotExist:
        return Response(
            {'error': 'Video not found or you do not have permission to access it'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_video_metadata: {str(e)}")
        return Response(
            {'error': f'Unexpected error: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def extract_thumbnail(request):
    """
    Extract thumbnail image from video.
    
    Request body:
        {
            "video_id": "uuid-of-video"
        }
    
    Response:
        {
            "success": true,
            "thumbnail_path": "/path/to/thumbnail.jpg",
            "message": "Thumbnail extracted successfully"
        }
    """
    serializer = VideoProcessingSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    video_id = serializer.validated_data['video_id']
    
    try:
        # Get the video instance
        video = Video.objects.get(id=video_id, user=request.user)
        
        # Initialize processing service
        processing_service = VideoProcessingService(video)
        
        # Extract thumbnail
        result = processing_service.extract_thumbnail()
        
        if result['success']:
            return Response(result, status=status.HTTP_200_OK)
        else:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Video.DoesNotExist:
        return Response(
            {'error': 'Video not found or you do not have permission to access it'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Unexpected error in extract_thumbnail: {str(e)}")
        return Response(
            {'error': f'Unexpected error: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
