from django.db import models
from django.conf import settings
import uuid


class Video(models.Model):
    """Model for storing video information"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='videos')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Original video
    original_video = models.FileField(upload_to='videos/original/%Y/%m/%d/', max_length=500)
    thumbnail = models.ImageField(upload_to='videos/thumbnails/%Y/%m/%d/', blank=True, null=True)
    
    # Processed video
    processed_video = models.FileField(upload_to='videos/processed/%Y/%m/%d/', blank=True, null=True, max_length=500)
    
    # Translated audio (NEW - for S2S translation)
    translated_audio_path = models.CharField(max_length=500, blank=True, null=True, help_text="Path to translated audio file")
    
    # Video metadata
    duration = models.FloatField(default=0.0, help_text="Duration in seconds")
    file_size = models.BigIntegerField(default=0, help_text="File size in bytes")
    resolution = models.CharField(max_length=20, blank=True)
    
    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0, help_text="Processing progress percentage")
    error_message = models.TextField(blank=True, null=True)
    
    # Translation settings
    source_language = models.CharField(max_length=10, blank=True, default='en')
    target_language = models.CharField(max_length=10, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.title} - {self.user.email}"
    
    @property
    def file_size_mb(self):
        return round(self.file_size / (1024 * 1024), 2)
    
    @property
    def is_processing(self):
        return self.status == 'processing'
    
    @property
    def is_completed(self):
        return self.status == 'completed'


class ProcessingLog(models.Model):
    """Model for storing processing logs"""
    
    LOG_LEVEL_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
    ]
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='logs')
    step = models.CharField(max_length=100)
    level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES, default='info')
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.video.title} - {self.step} - {self.level}"
