"""
Lip Sync Views
API endpoints for lip synchronization operations.
"""
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404

from videos.models import Video
from .models import LipSyncJob
from .serializers import (
    LipSyncJobSerializer,
    LipSyncStartRequestSerializer,
    LipSyncStatusSerializer
)

logger = logging.getLogger(__name__)


class LipSyncJobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for LipSyncJob.
    Provides list, retrieve operations and custom actions.
    """
    
    queryset = LipSyncJob.objects.all()
    serializer_class = LipSyncJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter lip sync jobs by user."""
        return LipSyncJob.objects.filter(
            video__user=self.request.user
        ).select_related('video', 'voice_cloning_task')
    
    @action(detail=False, methods=['post'], url_path='start')
    def start(self, request):
        """
        Start lip sync processing for a video.
        POST /api/lip-sync/start/
        
        Request body:
            {
                "video_id": "uuid-of-video",
                "force_restart": false
            }
        
        Response:
            {
                "success": true,
                "job_id": "uuid-of-job",
                "message": "Lip sync job started"
            }
        """
        serializer = LipSyncStartRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        video_id = serializer.validated_data['video_id']
        force_restart = serializer.validated_data.get('force_restart', False)
        
        try:
            # Get video
            video = get_object_or_404(Video, id=video_id, user=request.user)
            
            # Check if video has completed voice cloning
            latest_task = VoiceCloningTask.objects.filter(
                video=video,
                status='completed'
            ).order_by('-completed_at').first()
            
            if not latest_task:
                return Response(
                    {'error': 'No completed voice cloning task found for this video'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check for existing job
            existing_job = LipSyncJob.objects.filter(
                video=video,
                status__in=['pending', 'processing']
            ).first()
            
            if existing_job and not force_restart:
                return Response(
                    {
                        'success': False,
                        'message': 'Lip sync job already in progress',
                        'job_id': str(existing_job.id)
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get input paths
            input_video_path = video.original_video.path
            input_audio_path = latest_task.cloned_audio.path
            
            # Create new job
            job = LipSyncJob.objects.create(
                video=video,
                voice_cloning_task=latest_task,
                input_video_path=input_video_path,
                input_audio_path=input_audio_path,
                status='pending'
            )
            
            logger.info(f"Lip sync job created: {job.id} for video {video.id}")
            
            # Note: Actual processing will be triggered by orchestrator
            # This endpoint just creates the job record
            
            return Response({
                'success': True,
                'job_id': str(job.id),
                'message': 'Lip sync job created',
                'job': LipSyncJobSerializer(job).data
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error starting lip sync: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'], url_path='by-video/(?P<video_id>[^/.]+)')
    def by_video(self, request, video_id=None):
        """
        Get all lip sync jobs for a specific video.
        GET /api/lip-sync/by-video/{video_id}/
        """
        try:
            video = get_object_or_404(Video, id=video_id, user=request.user)
            jobs = LipSyncJob.objects.filter(video=video).order_by('-created_at')
            
            serializer = self.get_serializer(jobs, many=True)
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error retrieving lip sync jobs: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='status')
    def status_check(self, request, pk=None):
        """
        Get current status of a lip sync job.
        GET /api/lip-sync/{job_id}/status/
        """
        try:
            job = self.get_object()
            serializer = LipSyncStatusSerializer(job)
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error retrieving job status: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
