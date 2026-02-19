"""
Video processing service for audio extraction.
This is the first module - separating audio from video.
Based on Murtaza's processing pipeline strategy.
"""
import logging
from pathlib import Path
from django.conf import settings
from .utils.audio_utils import extract_audio, get_duration
from .utils.video_utils import extract_still_image, get_video_info

logger = logging.getLogger(__name__)


class VideoProcessingService:
    """
    Service for handling video processing operations.
    Currently implements audio extraction (Module 1).
    """
    
    def __init__(self, video_instance):
        """
        Initialize the processing service with a Video instance.
        
        Args:
            video_instance: videos.models.Video instance
        """
        self.video = video_instance
        self.processing_dir = Path(settings.MEDIA_ROOT) / 'processing' / str(self.video.id)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_audio_from_video(self):
        """
        Extract audio from the uploaded video file.
        This is Module 1 of the AI processing pipeline.
        
        Returns:
            dict: Result containing success status and extracted audio path
        """
        try:
            # Update video status
            self.video.status = 'processing'
            self.video.progress = 10
            self.video.save()
            
            # Log the extraction start
            self._log_processing('audio_extraction', 'info', 'Starting audio extraction from video')
            
            # Get video file path
            video_path = self.video.original_video.path
            
            # Define output audio path
            audio_filename = 'extracted_audio.wav'
            audio_path = self.processing_dir / audio_filename
            
            # Extract audio using ffmpeg
            success = extract_audio(video_path, audio_path)
            
            if not success:
                raise Exception("Failed to extract audio from video")
            
            # Get audio duration
            audio_duration = get_duration(audio_path)
            
            # Update video progress
            self.video.progress = 20
            self.video.save()
            
            self._log_processing('audio_extraction', 'info', 
                               f'Audio extracted successfully. Duration: {audio_duration}s')
            
            return {
                'success': True,
                'audio_path': str(audio_path),
                'audio_duration': audio_duration,
                'message': 'Audio extracted successfully'
            }
            
        except Exception as e:
            error_msg = f"Error in audio extraction: {str(e)}"
            logger.error(error_msg)
            
            self._log_processing('audio_extraction', 'error', error_msg)
            
            # Update video status to failed
            self.video.status = 'failed'
            self.video.error_message = error_msg
            self.video.save()
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_video_metadata(self):
        """
        Extract metadata from the video file.
        
        Returns:
            dict: Video metadata including duration, resolution, etc.
        """
        try:
            video_path = self.video.original_video.path
            
            # Get video duration
            duration = get_duration(video_path)
            
            # Get video info (resolution, fps, codec)
            video_info = get_video_info(video_path)
            
            # Update video model with metadata
            self.video.duration = duration
            if video_info.get('width') and video_info.get('height'):
                self.video.resolution = f"{video_info['width']}x{video_info['height']}"
            self.video.save()
            
            return {
                'success': True,
                'duration': duration,
                'resolution': self.video.resolution,
                'video_info': video_info
            }
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def extract_thumbnail(self):
        """
        Extract a thumbnail image from the video.
        
        Returns:
            dict: Result containing success status and thumbnail path
        """
        try:
            video_path = self.video.original_video.path
            
            # Define thumbnail path
            thumbnail_filename = 'thumbnail.jpg'
            thumbnail_path = self.processing_dir / thumbnail_filename
            
            # Extract still image
            success = extract_still_image(video_path, thumbnail_path)
            
            if not success:
                raise Exception("Failed to extract thumbnail")
            
            # Save thumbnail path to video model
            # We'll store relative path from MEDIA_ROOT
            relative_path = f'processing/{self.video.id}/{thumbnail_filename}'
            self.video.thumbnail = relative_path
            self.video.save()
            
            return {
                'success': True,
                'thumbnail_path': str(thumbnail_path),
                'message': 'Thumbnail extracted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error extracting thumbnail: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _log_processing(self, step, level, message):
        """
        Helper method to log processing steps.
        
        Args:
            step: Processing step name
            level: Log level (info, warning, error)
            message: Log message
        """
        from videos.models import ProcessingLog
        
        ProcessingLog.objects.create(
            video=self.video,
            step=step,
            level=level,
            message=message
        )
    
    def cleanup_processing_files(self):
        """
        Clean up temporary processing files.
        """
        try:
            import shutil
            if self.processing_dir.exists():
                shutil.rmtree(self.processing_dir)
                logger.info(f"Cleaned up processing directory: {self.processing_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up processing files: {str(e)}")
