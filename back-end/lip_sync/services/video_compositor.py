"""
Video Compositor Service
Handles final video composition with audio merging.
"""
import logging
import subprocess
import shutil
from pathlib import Path
from django.conf import settings
from django.core.files import File

logger = logging.getLogger(__name__)


class VideoCompositorService:
    """
    Service for composing final video output.
    Merges lip-synced video with cloned audio using FFmpeg.
    """
    
    def __init__(self, video_instance, lip_sync_job):
        """
        Initialize compositor service.
        
        Args:
            video_instance: videos.models.Video instance
            lip_sync_job: LipSyncJob instance
        """
        self.video = video_instance
        self.job = lip_sync_job
        self.processing_dir = Path(settings.MEDIA_ROOT) / 'processing' / str(self.video.id)
    
    def compose_final_video(self, lip_synced_video_path, audio_path):
        """
        Compose final video by merging lip-synced video with cloned audio.
        
        Args:
            lip_synced_video_path: Path to lip-synced video
            audio_path: Path to cloned audio
        
        Returns:
            dict: Result containing final video path
        """
        try:
            logger.info(f"Starting final video composition for video {self.video.id}")
            
            # Update job status
            self.job.update_status('processing', progress=95)
            self._log_processing('composition', 'info', 'Starting final video composition...')
            
            # Verify input files exist and have correct streams
            lip_synced_path = Path(lip_synced_video_path)
            audio_file_path = Path(audio_path)
            
            if not lip_synced_path.exists():
                raise FileNotFoundError(f"Lip-synced video not found: {lip_synced_video_path}")
            if not audio_file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            logger.info(f"Input video exists: {lip_synced_video_path} ({lip_synced_path.stat().st_size} bytes)")
            logger.info(f"Input audio exists: {audio_path} ({audio_file_path.stat().st_size} bytes)")
            
            # Probe input video to check streams
            probe_cmd = ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(lip_synced_video_path)]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            logger.info(f"Video stream info: {probe_result.stdout[:200]}")
            
            # Create output directory with date structure
            from datetime import datetime
            now = datetime.now()
            output_dir = Path(settings.MEDIA_ROOT) / 'videos' / 'processed' / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output filename
            output_filename = f"{self.video.id}_final.mp4"
            output_path = output_dir / output_filename
            
            # Merge video and audio using FFmpeg
            cmd = [
                'ffmpeg',
                '-i', str(lip_synced_video_path),  # Input video
                '-i', str(audio_path),              # Input audio
                '-c:v', 'copy',                     # Copy video codec (no re-encoding)
                '-c:a', 'aac',                      # Encode audio to AAC
                '-b:a', '192k',                     # Audio bitrate
                '-map', '0:v:0',                    # Map video from first input
                '-map', '1:a:0',                    # Map audio from second input
                '-shortest',                        # Match shortest stream
                '-y',                               # Overwrite output
                str(output_path)
            ]
            
            logger.info(f"Running FFmpeg composition: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Log FFmpeg output for debugging
            if result.stdout:
                logger.info(f"FFmpeg stdout: {result.stdout[:500]}")
            if result.stderr:
                logger.info(f"FFmpeg stderr: {result.stderr[:500]}")
            
            # Check return code
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, 
                    cmd, 
                    output=result.stdout, 
                    stderr=result.stderr
                )
            
            # Verify output file
            if not output_path.exists():
                raise FileNotFoundError(f"Composed video not found at {output_path}")
            
            logger.info(f"Video composition completed: {output_path}")
            
            # Update job with final path
            self.job.final_video_path = str(output_path)
            self.job.update_status('processing', progress=98)
            
            # NOTE: Video model's processed_video field is set in orchestrator
            # after this method returns, so we don't set it here
            
            self._log_processing('composition', 'info',
                               f'Final video composed successfully: {output_filename}')
            
            return {
                'success': True,
                'output_path': str(output_path),
                'message': 'Final video composition completed'
            }
            
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg composition failed: {e.stderr}"
            logger.error(error_msg)
            self._log_processing('composition', 'error', error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Video composition failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_processing('composition', 'error', error_msg)
            return {
                'success': False,
                'error': error_msg
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
