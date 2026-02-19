"""
Wav2Lip Service
Handles lip synchronization using Wav2Lip model.
"""
import os
import sys
import time
import logging
import subprocess
import torch
from pathlib import Path
from django.conf import settings
from django.core.files import File
from django.utils import timezone

logger = logging.getLogger(__name__)


class Wav2LipService:
    """
    Service for lip synchronization using Wav2Lip.
    Manages the external Wav2Lip inference process.
    """
    
    def __init__(self, video_instance, lip_sync_job):
        """
        Initialize Wav2Lip service.
        
        Args:
            video_instance: videos.models.Video instance
            lip_sync_job: LipSyncJob instance
        """
        self.video = video_instance
        self.job = lip_sync_job
        self.processing_dir = Path(settings.MEDIA_ROOT) / 'processing' / str(self.video.id) / 'lip_sync'
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu = torch.cuda.is_available()
        logger.info(f"Wav2LipService initialized with device: {self.device}")
        
        # Wav2Lip paths
        self.wav2lip_repo_path = Path(settings.BASE_DIR).parent / 'extras' / 'Wav2Lip'
        self.checkpoint_path = Path(settings.BASE_DIR).parent / 'models' / 'wav2lip' / 'checkpoints' / 'wav2lip_gan.pth'
        
        # Verify paths
        if not self.wav2lip_repo_path.exists():
            raise FileNotFoundError(f"Wav2Lip repository not found at {self.wav2lip_repo_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Wav2Lip checkpoint not found at {self.checkpoint_path}. "
                f"Please download wav2lip_gan.pth from "
                f"https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
            )
    
    def run_lip_sync(self, video_path, audio_path):
        """
        Run Wav2Lip inference to synchronize lips with audio.
        
        Args:
            video_path: Path to input video file
            audio_path: Path to input audio file (cloned voice)
        
        Returns:
            dict: Result containing output video path and metadata
        """
        try:
            logger.info(f"Starting lip sync for video {self.video.id}")
            start_time = time.time()
            
            # Update job status
            self.job.update_status('processing', progress=85)
            self.job.use_gpu = self.use_gpu
            self.job.started_at = timezone.now()
            self.job.save()
            
            # Define output path
            output_filename = f"lip_synced_{self.video.id}.mp4"
            output_path = self.processing_dir / output_filename
            
            # Ensure output directory and Wav2Lip temp directory exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            wav2lip_temp = self.wav2lip_repo_path / 'temp'
            wav2lip_temp.mkdir(exist_ok=True)
            
            # Prepare Wav2Lip inference command
            inference_script = self.wav2lip_repo_path / 'inference.py'
            
            cmd = [
                sys.executable,
                str(inference_script),
                '--checkpoint_path', str(self.checkpoint_path),
                '--face', str(video_path),
                '--audio', str(audio_path),
                '--outfile', str(output_path),
            ]
            
            # Optional: Add quality parameters
            # Resize factor (1 for original, 2 for half, etc.)
            cmd.extend(['--resize_factor', '1'])
            
            # FPS for output video
            cmd.extend(['--fps', '25'])
            
            # Padding for face detection
            cmd.extend(['--pads', '0', '10', '0', '0'])
            
            logger.info(f"Running Wav2Lip command: {' '.join(cmd)}")
            
            # Calculate timeout based on video/audio duration
            # Wav2Lip processes roughly at 0.5x-1x real-time on CPU, slower on longer videos
            # Estimate: 30 seconds per minute of video, with minimum 15 minutes
            try:
                from processing.utils.video_utils import get_duration
                video_duration = get_duration(str(video_path))
                # Calculate timeout: 30 seconds per minute of video + 5 minute buffer
                calculated_timeout = int(video_duration * 30 + 300)
                timeout = max(900, calculated_timeout)  # Minimum 15 minutes
                logger.info(f"Calculated Wav2Lip timeout: {timeout}s for {video_duration:.1f}s video")
            except Exception as e:
                logger.warning(f"Could not calculate video duration: {e}, using default timeout")
                timeout = 1800  # Default 30 minute timeout
            
            # Update progress
            self.job.update_status('processing', progress=88)
            self._log_processing('lip_sync', 'info', f'Running Wav2Lip inference (may take up to {timeout//60} minutes)...')
            
            # Run Wav2Lip inference
            env = os.environ.copy()
            # Add Wav2Lip repo to Python path
            env['PYTHONPATH'] = str(self.wav2lip_repo_path) + os.pathsep + env.get('PYTHONPATH', '')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(self.wav2lip_repo_path),
                timeout=timeout
            )
            
            # Log stdout and stderr for debugging
            if result.stdout:
                logger.info(f"Wav2Lip stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Wav2Lip stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = f"Wav2Lip inference failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            logger.info(f"Wav2Lip inference completed with return code 0")
            
            # Verify output file exists
            if not output_path.exists():
                error_details = (
                    f"Expected output: {output_path}\n"
                    f"Wav2Lip working dir: {self.wav2lip_repo_path}\n"
                    f"Stdout: {result.stdout[:500] if result.stdout else 'None'}\n"
                    f"Stderr: {result.stderr[:500] if result.stderr else 'None'}"
                )
                logger.error(f"Wav2Lip did not create output file. Details:\n{error_details}")
                raise FileNotFoundError(
                    f"Wav2Lip output not found at {output_path}. "
                    f"Check if FFmpeg is installed and accessible from PATH."
                )
            
            # Get video duration
            from processing.utils.video_utils import get_duration
            video_duration = get_duration(str(output_path))
            
            # Update job
            processing_time = time.time() - start_time
            self.job.lip_synced_video_path = str(output_path)
            self.job.video_duration = video_duration
            self.job.processing_time = processing_time
            self.job.update_status('processing', progress=92)
            
            self._log_processing('lip_sync', 'info',
                               f'Lip sync completed successfully in {processing_time:.2f}s')
            
            return {
                'success': True,
                'output_path': str(output_path),
                'video_duration': video_duration,
                'processing_time': processing_time,
                'use_gpu': self.use_gpu
            }
            
        except subprocess.TimeoutExpired:
            error_msg = f"Wav2Lip inference timed out after {timeout//60} minutes. Consider using a shorter video or GPU acceleration."
            logger.error(error_msg)
            self._log_processing('lip_sync', 'error', error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Lip sync failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_processing('lip_sync', 'error', error_msg)
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
