from rest_framework import serializers
from .models import Video, ProcessingLog


class ProcessingLogSerializer(serializers.ModelSerializer):
    """Serializer for processing logs"""
    
    class Meta:
        model = ProcessingLog
        fields = ['id', 'step', 'level', 'message', 'timestamp']
        read_only_fields = ['id', 'timestamp']


class VideoSerializer(serializers.ModelSerializer):
    """Serializer for video model"""
    
    file_size_mb = serializers.ReadOnlyField()
    is_processing = serializers.ReadOnlyField()
    is_completed = serializers.ReadOnlyField()
    logs = ProcessingLogSerializer(many=True, read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)
    cloned_audio_url = serializers.SerializerMethodField()
    translated_audio_url = serializers.SerializerMethodField()
    processed_video_url = serializers.SerializerMethodField()
    
    def get_cloned_audio_url(self, obj):
        """Get the cloned audio URL from the latest voice cloning task."""
        try:
            # Get the latest completed voice cloning task for this video
            task = obj.voice_cloning_tasks.filter(status='completed').order_by('-completed_at').first()
            if task and task.cloned_audio:
                # Return the URL to the cloned audio file
                return task.cloned_audio.url if hasattr(task.cloned_audio, 'url') else str(task.cloned_audio)
            return None
        except Exception:
            return None
    
    def get_translated_audio_url(self, obj):
        """Get the translated/cloned audio URL from S2S translation + voice cloning."""
        if obj.translated_audio_path:
            # Extract just the filename from the full path
            # Path format: media/processing/{uuid}/cloned_audio.wav or translated_audio.wav
            from pathlib import Path
            filename = Path(obj.translated_audio_path).name
            return f"/media/processing/{obj.id}/{filename}"
        return None
    
    def get_processed_video_url(self, obj):
        """Get the final processed video URL (lip synced video)."""
        if obj.processed_video and obj.processed_video.name:
            # processed_video is stored as: processing/{uuid}/final_video.mp4
            return f"/media/{obj.processed_video.name}"
        return None
    
    class Meta:
        model = Video
        fields = [
            'id', 'user', 'user_email', 'title', 'description',
            'original_video', 'thumbnail', 'processed_video',
            'duration', 'file_size', 'file_size_mb', 'resolution',
            'status', 'progress', 'error_message',
            'source_language', 'target_language',
            'created_at', 'updated_at', 'processed_at',
            'is_processing', 'is_completed', 'logs', 'cloned_audio_url',
            'translated_audio_url', 'processed_video_url', 'translated_audio_path'
        ]
        read_only_fields = [
            'id', 'user', 'status', 'progress', 'error_message',
            'duration', 'file_size', 'resolution', 'processed_video',
            'thumbnail', 'created_at', 'updated_at', 'processed_at',
            'translated_audio_path'
        ]


class VideoCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating videos"""
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'description', 'original_video',
            'source_language', 'target_language', 'status', 'progress'
        ]
        read_only_fields = ['id', 'status', 'progress']
    
    def validate_original_video(self, value):
        import subprocess
        import tempfile
        import os
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if value.size > max_size:
            raise serializers.ValidationError(f"File size cannot exceed 100MB. Current size: {round(value.size / (1024 * 1024), 2)}MB")
        
        # Check file extension
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_extension = value.name.lower()[value.name.rfind('.'):];
        if file_extension not in allowed_extensions:
            raise serializers.ValidationError(f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}")
        
        # Check video duration (max 1 minute = 60 seconds)
        try:
            # Save uploaded file temporarily to check duration
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                for chunk in value.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            # Reset file pointer after reading
            value.seek(0)
            
            # Use ffprobe to get video duration
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                if duration > 60:
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    raise serializers.ValidationError(
                        f"Video duration must be 1 minute or less. Your video is {minutes}:{seconds:02d} minutes long."
                    )
        except subprocess.TimeoutExpired:
            # If ffprobe times out, log but don't block upload
            pass
        except Exception as e:
            # If duration check fails, log but don't block upload
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not verify video duration: {str(e)}")
        
        return value


class VideoUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating video information"""
    
    class Meta:
        model = Video
        fields = ['title', 'description', 'source_language', 'target_language']


class VideoListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for video list"""
    
    file_size_mb = serializers.ReadOnlyField()
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'thumbnail', 'status', 'progress',
            'duration', 'file_size_mb', 'created_at', 'updated_at'
        ]
