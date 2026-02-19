from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from .models import Video, ProcessingLog
from .serializers import (
    VideoSerializer, VideoCreateSerializer,
    VideoUpdateSerializer, VideoListSerializer,
    ProcessingLogSerializer
)
import threading
import logging

logger = logging.getLogger(__name__)


class VideoListCreateView(generics.ListCreateAPIView):
    """View for listing and creating videos"""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return VideoCreateSerializer
        return VideoListSerializer
    
    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)
    
    def create(self, request, *args, **kwargs):
        """Override create to add better error logging"""
        serializer = self.get_serializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            logger.error(f"Validation errors: {serializer.errors if hasattr(serializer, 'errors') else 'No serializer errors'}")
            logger.error(f"Request data keys: {request.data.keys()}")
            return Response(
                {
                    "error": str(e),
                    "validation_errors": serializer.errors if hasattr(serializer, 'errors') else None
                },
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def perform_create(self, serializer):
        video = serializer.save(user=self.request.user)
        
        # Update file size
        if video.original_video:
            video.file_size = video.original_video.size
            video.save()
        
        # Start AI processing in background thread
        logger.info(f"Starting AI processing for video {video.id}")
        thread = threading.Thread(target=self._start_ai_processing, args=(video.id,))
        thread.daemon = True
        thread.start()
        
        return video
    
    def _start_ai_processing(self, video_id):
        """Start AI processing in background"""
        try:
            from processing.orchestrator import start_ai_processing
            logger.info(f"AI processing thread started for video {video_id}")
            result = start_ai_processing(video_id)
            if result['success']:
                logger.info(f"AI processing completed for video {video_id}")
            else:
                logger.error(f"AI processing failed for video {video_id}: {result['error']}")
        except Exception as e:
            logger.error(f"Error in AI processing thread: {str(e)}")


class VideoDetailView(generics.RetrieveUpdateDestroyAPIView):
    """View for retrieving, updating, and deleting a video"""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def get_serializer_class(self):
        if self.request.method in ['PUT', 'PATCH']:
            return VideoUpdateSerializer
        return VideoSerializer
    
    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)


class VideoDownloadView(APIView):
    """View for downloading processed video"""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, pk):
        video = get_object_or_404(Video, pk=pk, user=request.user)
        
        if not video.processed_video:
            return Response(
                {"error": "Processed video not available"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response({
            "url": video.processed_video.url,
            "title": video.title,
            "file_size": video.file_size
        })


class VideoStatusView(APIView):
    """View for checking video processing status"""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, pk):
        video = get_object_or_404(Video, pk=pk, user=request.user)
        
        return Response({
            "id": video.id,
            "status": video.status,
            "progress": video.progress,
            "error_message": video.error_message,
            "created_at": video.created_at,
            "updated_at": video.updated_at,
            "processed_at": video.processed_at
        })


class VideoLogsView(generics.ListAPIView):
    """View for retrieving video processing logs"""
    
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ProcessingLogSerializer
    
    def get_queryset(self):
        video_id = self.kwargs.get('pk')
        video = get_object_or_404(Video, pk=video_id, user=self.request.user)
        return ProcessingLog.objects.filter(video=video)


class UserVideosStatsView(APIView):
    """View for getting user's video statistics"""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        user = request.user
        videos = Video.objects.filter(user=user)
        
        total_videos = videos.count()
        completed_videos = videos.filter(status='completed').count()
        processing_videos = videos.filter(status='processing').count()
        failed_videos = videos.filter(status='failed').count()
        total_storage = sum(video.file_size for video in videos)
        
        return Response({
            "total_videos": total_videos,
            "completed_videos": completed_videos,
            "processing_videos": processing_videos,
            "failed_videos": failed_videos,
            "total_storage_bytes": total_storage,
            "total_storage_mb": round(total_storage / (1024 * 1024), 2)
        })
