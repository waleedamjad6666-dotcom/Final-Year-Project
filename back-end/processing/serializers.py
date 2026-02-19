"""
Serializers for video processing module.
"""
from rest_framework import serializers
from videos.models import Video


class VideoProcessingSerializer(serializers.Serializer):
    """
    Serializer for video processing requests.
    """
    video_id = serializers.UUIDField(help_text="UUID of the video to process")


class AudioExtractionResponseSerializer(serializers.Serializer):
    """
    Serializer for audio extraction response.
    """
    success = serializers.BooleanField()
    audio_path = serializers.CharField(required=False)
    audio_duration = serializers.FloatField(required=False)
    message = serializers.CharField(required=False)
    error = serializers.CharField(required=False)


class VideoMetadataResponseSerializer(serializers.Serializer):
    """
    Serializer for video metadata response.
    """
    success = serializers.BooleanField()
    duration = serializers.FloatField(required=False)
    resolution = serializers.CharField(required=False)
    video_info = serializers.DictField(required=False)
    error = serializers.CharField(required=False)
