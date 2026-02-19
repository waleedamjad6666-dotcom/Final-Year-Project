"""
Lip Sync Serializers
"""
from rest_framework import serializers
from .models import LipSyncJob


class LipSyncJobSerializer(serializers.ModelSerializer):
    """Serializer for LipSyncJob model"""
    
    video_title = serializers.CharField(source='video.title', read_only=True)
    video_id = serializers.UUIDField(source='video.id', read_only=True)
    
    class Meta:
        model = LipSyncJob
        fields = [
            'id', 'video', 'video_id', 'video_title',
            'voice_cloning_task', 'input_video_path', 'input_audio_path',
            'lip_synced_video_path', 'final_video_path',
            'status', 'progress', 'error_message',
            'processing_time', 'video_duration', 'use_gpu',
            'created_at', 'updated_at', 'started_at', 'completed_at'
        ]
        read_only_fields = [
            'id', 'status', 'progress', 'error_message',
            'lip_synced_video_path', 'final_video_path',
            'processing_time', 'video_duration', 'use_gpu',
            'created_at', 'updated_at', 'started_at', 'completed_at'
        ]


class LipSyncStartRequestSerializer(serializers.Serializer):
    """Serializer for starting lip sync processing"""
    
    video_id = serializers.UUIDField(required=True)
    force_restart = serializers.BooleanField(default=False, required=False)


class LipSyncStatusSerializer(serializers.ModelSerializer):
    """Lightweight serializer for status checking"""
    
    class Meta:
        model = LipSyncJob
        fields = ['id', 'status', 'progress', 'error_message', 'updated_at']
