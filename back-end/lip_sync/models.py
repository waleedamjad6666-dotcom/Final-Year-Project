"""
Lip Sync Models
Stores lip synchronization job information.
"""
from django.db import models
import uuid


class LipSyncJob(models.Model):
    """
    Model for tracking lip synchronization operations.
    Links video, cloned audio, and final output.
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.ForeignKey(
        'videos.Video',
        on_delete=models.CASCADE,
        related_name='lip_sync_jobs'
    )
    # voice_cloning_task = models.ForeignKey(
    #     'voice_cloning.VoiceCloningTask',
    #     on_delete=models.SET_NULL,
    #     null=True,
    #     blank=True,
    #     related_name='lip_sync_jobs',
    #     help_text="The voice cloning task that produced the audio"
    # )
    
    # Input files
    input_video_path = models.CharField(
        max_length=500,
        help_text="Path to original video file"
    )
    input_audio_path = models.CharField(
        max_length=500,
        help_text="Path to cloned audio file"
    )
    
    # Output files
    lip_synced_video_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to lip-synced video (before composition)"
    )
    final_video_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to final composed video"
    )
    
    # Processing status
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    progress = models.IntegerField(
        default=85,
        help_text="Processing progress percentage (85-100)"
    )
    error_message = models.TextField(blank=True, null=True)
    
    # Processing metadata
    processing_time = models.FloatField(
        default=0.0,
        help_text="Total processing time in seconds"
    )
    video_duration = models.FloatField(
        default=0.0,
        help_text="Duration of output video in seconds"
    )
    use_gpu = models.BooleanField(
        default=False,
        help_text="Whether GPU was used for processing"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Lip Sync Job'
        verbose_name_plural = 'Lip Sync Jobs'
        indexes = [
            models.Index(fields=['video', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"LipSyncJob for Video {self.video.id} - {self.status}"
    
    def update_status(self, status, progress=None, error=None):
        """Helper method to update job status."""
        self.status = status
        if progress is not None:
            self.progress = progress
        if error:
            self.error_message = error
        self.save()
