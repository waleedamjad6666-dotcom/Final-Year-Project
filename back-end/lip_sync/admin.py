"""
Admin configuration for Lip Sync app
"""
from django.contrib import admin
from .models import LipSyncJob


@admin.register(LipSyncJob)
class LipSyncJobAdmin(admin.ModelAdmin):
    """Admin configuration for LipSyncJob model"""
    
    list_display = [
        'id', 'video', 'status', 'progress', 
        'use_gpu', 'processing_time', 'created_at'
    ]
    list_filter = ['status', 'use_gpu', 'created_at']
    search_fields = ['id', 'video__title', 'video__id']
    readonly_fields = [
        'id', 'created_at', 'updated_at', 
        'started_at', 'completed_at'
    ]
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('id', 'video', 'voice_cloning_task')
        }),
        ('Input Files', {
            'fields': ('input_video_path', 'input_audio_path')
        }),
        ('Output Files', {
            'fields': ('lip_synced_video_path', 'final_video_path')
        }),
        ('Status', {
            'fields': ('status', 'progress', 'error_message')
        }),
        ('Metadata', {
            'fields': ('processing_time', 'video_duration', 'use_gpu')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'started_at', 'completed_at')
        }),
    )
