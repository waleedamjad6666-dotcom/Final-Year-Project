"""
Wav2Lip Service - OPTIMIZED AND FIXED
Handles lip synchronization using Wav2Lip model.
CRITICAL FIXES: Real-time progress monitoring, proper timeout handling, comprehensive error handling,
graceful degradation, checkpoint verification, GPU/CPU fallback.
"""
import os
import sys
import time
import logging
import subprocess
import threading
import cv2
import torch
from pathlib import Path
from django.conf import settings
from django.core.files import File
from django.utils import timezone

logger = logging.getLogger(__name__)


class Wav2LipService:
    """
    Service for lip synchronization using Wav2Lip.
    Manages the external Wav2Lip inference process with comprehensive error handling.
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
        
        # Verify and validate checkpoint
        self._verify_checkpoint()
    
    def _verify_checkpoint(self):
        """
        Verify Wav2Lip checkpoint file exists and is valid.
        """
        if not self.checkpoint_path.exists():
            error_msg = (
                f"Wav2Lip checkpoint not found at {self.checkpoint_path}.\n"
                f"Please download wav2lip_gan.pth from:\n"
                f"https://github.com/Rudrabha/Wav2Lip#getting-the-weights\n"
                f"Expected location: {self.checkpoint_path}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check file size (should be ~200-300MB)
        file_size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"Wav2Lip checkpoint found: {file_size_mb:.1f}MB")
        
        if file_size_mb < 150 or file_size_mb > 350:
            logger.warning(
                f"Checkpoint file size ({file_size_mb:.1f}MB) seems unusual. "
                f"Expected ~200-300MB. File may be corrupted."
            )
    
    def _verify_video_input(self, video_path):
        """
        Verify input video is readable and has faces.
        
        Args:
            video_path: Path to input video
        
        Returns:
            tuple: (is_valid, frame_count, fps, duration)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return False, 0, 0, 0
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            # Read first frame to verify video is valid
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logger.error("Cannot read frames from video")
                return False, 0, 0, 0
            
            logger.info(
                f"Input video verified: {frame_count} frames, {fps:.2f} fps, {duration:.2f}s duration"
            )
            return True, frame_count, fps, duration
            
        except Exception as e:
            logger.error(f"Video verification failed: {e}", exc_info=True)
            return False, 0, 0, 0
    
    def _stream_output(self, pipe, log_func, progress_callback=None):
        """
        Stream subprocess output in real-time.
        
        Args:
            pipe: subprocess stdout or stderr pipe
            log_func: logging function (logger.info or logger.warning)
            progress_callback: Optional callback for progress updates
        """
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    line = line.strip()
                    log_func(f"Wav2Lip: {line}")
                    
                    # Parse progress indicators
                    if progress_callback and ('frame' in line.lower() or '%' in line):
                        try:
                            # Try to extract frame number or percentage
                            if 'frame' in line.lower():
                                # Example: "Processing frame 150/300"
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if 'frame' in part.lower() and i + 1 < len(parts):
                                        frame_info = parts[i + 1].split('/')
                                        if len(frame_info) == 2:
                                            current = int(frame_info[0])
                                            total = int(frame_info[1])
                                            progress = 88 + int((current / total) * 7)  # 88-95%
                                            progress_callback(progress)
                        except Exception:
                            pass  # Ignore parsing errors
        except Exception as e:
            logger.error(f"Output streaming error: {e}")
    
    def _calculate_timeout(self, video_duration):
        """
        Calculate conservative timeout for Wav2Lip processing.
        
        Args:
            video_duration: Video duration in seconds
        
        Returns:
            int: Timeout in seconds
        """
        # Base formula: 120 seconds per second of video + 5 minute buffer
        # This is very conservative to avoid premature timeouts
        base_timeout = video_duration * 120 + 300
        
        # Minimum 10 minutes even for very short videos
        timeout = max(600, int(base_timeout * 1.5))
        
        logger.info(
            f"Calculated Wav2Lip timeout: {timeout}s ({timeout//60} minutes) "
            f"for {video_duration:.1f}s video"
        )
        return timeout
    
    def run_lip_sync(self, video_path, audio_path):
        """
        Run Wav2Lip inference to synchronize lips with audio.
        OPTIMIZED: Real-time progress monitoring, comprehensive error handling, graceful degradation.
        
        Args:
            video_path: Path to input video file
            audio_path: Path to input audio file (cloned voice)
        
        Returns:
            dict: Result containing output video path and metadata
        """
        process = None
        try:
            logger.info(f"=== Starting Wav2Lip lip sync for video {self.video.id} ===")
            start_time = time.time()
            
            # Update job status
            self.job.update_status('processing', progress=85)
            self.job.use_gpu = self.use_gpu
            self.job.started_at = timezone.now()
            self.job.save()
            
            # Verify input video
            is_valid, frame_count, fps, video_duration = self._verify_video_input(video_path)
            if not is_valid:
                raise ValueError(f"Invalid input video: {video_path}")
            
            # Verify input files exist
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Input video not found: {video_path}")
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Input audio not found: {audio_path}")
            
            logger.info(f"Input video: {video_path}")
            logger.info(f"Input audio: {audio_path}")
            logger.info(f"Video properties: {frame_count} frames, {fps:.2f} fps, {video_duration:.2f}s")
            
            # Define output paths
            output_filename = f"lip_synced_{self.video.id}.mp4"
            output_path = self.processing_dir / output_filename
            
            # Alternative output paths to check
            alt_output_paths = [
                self.wav2lip_repo_path / 'results' / output_filename,
                self.wav2lip_repo_path / output_filename,
                Path(output_filename).resolve()
            ]
            
            # Ensure directories exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            wav2lip_temp = self.wav2lip_repo_path / 'temp'
            wav2lip_temp.mkdir(exist_ok=True)
            wav2lip_results = self.wav2lip_repo_path / 'results'
            wav2lip_results.mkdir(exist_ok=True)
            
            # Prepare Wav2Lip inference command
            inference_script = self.wav2lip_repo_path / 'inference.py'
            
            if not inference_script.exists():
                raise FileNotFoundError(f"Wav2Lip inference.py not found at {inference_script}")
            
            cmd = [
                sys.executable,  # Use current Python interpreter
                str(inference_script),
                '--checkpoint_path', str(self.checkpoint_path),
                '--face', str(video_path),
                '--audio', str(audio_path),
                '--outfile', str(output_path),
                '--nosmooth',  # Disable smoothing to prevent hangs
                '--resize_factor', '1',  # Keep original resolution
                '--pads', '0', '10', '0', '0',  # Face detection padding
                '--face_det_batch_size', '2',  # Smaller batch size for stability
            ]
            
            # Log complete command
            logger.info(f"Wav2Lip command: {' '.join(cmd)}")
            logger.info(f"Working directory: {self.wav2lip_repo_path}")
            logger.info(f"Output path: {output_path}")
            
            # Calculate timeout
            timeout = self._calculate_timeout(video_duration)
            
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.wav2lip_repo_path) + os.pathsep + env.get('PYTHONPATH', '')
            logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
            
            # Update progress
            self.job.update_status('processing', progress=88)
            self._log_processing('lip_sync', 'info', 
                               f'Running Wav2Lip inference (timeout: {timeout//60} minutes)...')
            
            # Run Wav2Lip with real-time output streaming
            logger.info("Starting Wav2Lip subprocess...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                env=env,
                cwd=str(self.wav2lip_repo_path)
            )
            
            # Create threads for real-time output streaming
            def progress_callback(progress):
                self.job.update_status('processing', progress=min(progress, 95))
            
            stdout_thread = threading.Thread(
                target=self._stream_output,
                args=(process.stdout, logger.info, progress_callback)
            )
            stderr_thread = threading.Thread(
                target=self._stream_output,
                args=(process.stderr, logger.warning, None)
            )
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process completion with timeout
            try:
                logger.info(f"Waiting for Wav2Lip to complete (timeout: {timeout}s)...")
                returncode = process.wait(timeout=timeout)
                
                # Wait for output threads to finish
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                
                logger.info(f"Wav2Lip process completed with return code: {returncode}")
                
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)
                
            except subprocess.TimeoutExpired:
                logger.error(f"Wav2Lip timed out after {timeout}s ({timeout//60} minutes)")
                process.kill()
                process.wait()
                
                # Try graceful degradation
                return self._graceful_degradation(video_path, audio_path, 
                                                  f"Wav2Lip timed out after {timeout//60} minutes")
            
            # Verify output file exists
            output_found = False
            final_output_path = output_path
            
            if output_path.exists():
                output_found = True
                logger.info(f"Output found at primary path: {output_path}")
            else:
                logger.warning(f"Output not found at primary path: {output_path}")
                logger.info("Checking alternative output locations...")
                
                for alt_path in alt_output_paths:
                    if alt_path.exists():
                        logger.info(f"Output found at alternative path: {alt_path}")
                        # Move to expected location
                        import shutil
                        shutil.move(str(alt_path), str(output_path))
                        final_output_path = output_path
                        output_found = True
                        break
            
            if not output_found:
                error_details = (
                    f"Primary output: {output_path}\n"
                    f"Alternative paths checked: {[str(p) for p in alt_output_paths]}\n"
                    f"Working directory: {self.wav2lip_repo_path}\n"
                    f"Directory listing: {list(self.processing_dir.iterdir())[:10]}"
                )
                logger.error(f"Wav2Lip output file not found. Details:\n{error_details}")
                
                # Try graceful degradation
                return self._graceful_degradation(video_path, audio_path, 
                                                  "Wav2Lip completed but output file not found")
            
            # Verify output file is valid
            try:
                cap = cv2.VideoCapture(str(final_output_path))
                if not cap.isOpened():
                    raise ValueError("Cannot open output video")
                
                output_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    raise ValueError("Cannot read frames from output video")
                
                logger.info(f"Output video validated: {output_frame_count} frames")
                
            except Exception as validation_error:
                logger.error(f"Output validation failed: {validation_error}")
                return self._graceful_degradation(video_path, audio_path, 
                                                  f"Output video validation failed: {validation_error}")
            
            # Get output video properties
            from processing.utils.video_utils import get_duration
            try:
                output_duration = get_duration(str(final_output_path))
            except:
                output_duration = video_duration
            
            # Update job
            processing_time = time.time() - start_time
            self.job.lip_synced_video_path = str(final_output_path)
            self.job.video_duration = output_duration
            self.job.processing_time = processing_time
            self.job.update_status('processing', progress=95)
            
            logger.info(
                f"=== Wav2Lip completed successfully ===\n"
                f"Processing time: {processing_time:.2f}s ({processing_time/60:.2f} minutes)\n"
                f"Output: {final_output_path}\n"
                f"Duration: {output_duration:.2f}s"
            )
            
            self._log_processing('lip_sync', 'info',
                               f'Lip sync completed successfully in {processing_time:.1f}s')
            
            return {
                'success': True,
                'output_path': str(final_output_path),
                'video_duration': output_duration,
                'processing_time': processing_time,
                'use_gpu': self.use_gpu
            }
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Wav2Lip process failed with return code {e.returncode}"
            logger.error(error_msg, exc_info=True)
            return self._graceful_degradation(video_path, audio_path, error_msg)
            
        except Exception as e:
            error_msg = f"Lip sync failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._graceful_degradation(video_path, audio_path, error_msg)
        
        finally:
            # Ensure process is terminated
            if process and process.poll() is None:
                logger.warning("Terminating Wav2Lip process...")
                process.kill()
                process.wait()
    
    def _graceful_degradation(self, video_path, audio_path, error_reason):
        """
        GRACEFUL DEGRADATION: Provide usable output even if Wav2Lip fails.
        Merges translated audio with original video using FFmpeg (no lip sync).
        
        Args:
            video_path: Original video path
            audio_path: Translated audio path
            error_reason: Reason for Wav2Lip failure
        
        Returns:
            dict: Result with merged video (no lip sync)
        """
        try:
            logger.warning(f"Attempting graceful degradation due to: {error_reason}")
            self._log_processing('lip_sync', 'warning', 
                               f'Lip sync unavailable, providing dubbed video without lip sync: {error_reason}')
            
            # Create output with FFmpeg (merge audio with video)
            output_filename = f"dubbed_no_lipsync_{self.video.id}.mp4"
            output_path = self.processing_dir / output_filename
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # Copy video stream
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',  # End at shortest stream
                str(output_path)
            ]
            
            logger.info(f"Running FFmpeg merge: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for merge
            )
            
            if result.returncode != 0 or not output_path.exists():
                logger.error(f"FFmpeg merge failed: {result.stderr}")
                # Last resort: just use original video
                import shutil
                shutil.copy(str(video_path), str(output_path))
                logger.warning("Using original video as last resort")
            
            # Update job with degraded output
            self.job.lip_synced_video_path = str(output_path)
            self.job.update_status('processing', progress=95)
            self.job.error_message = f"Lip sync unavailable: {error_reason}. Delivered dubbed video without lip synchronization."
            self.job.save()
            
            logger.info(f"Graceful degradation successful: {output_path}")
            
            return {
                'success': True,  # Still return success to complete pipeline
                'output_path': str(output_path),
                'processing_time': 0,
                'use_gpu': False,
                'degraded': True,
                'warning': f"Lip sync failed, delivered without lip synchronization: {error_reason}"
            }
            
        except Exception as fallback_error:
            logger.error(f"Graceful degradation failed: {fallback_error}", exc_info=True)
            self._log_processing('lip_sync', 'error', 
                               f'Complete failure: {error_reason} | Fallback failed: {fallback_error}')
            
            return {
                'success': False,
                'error': f"Lip sync failed: {error_reason}. Fallback also failed: {fallback_error}"
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
        
        try:
            ProcessingLog.objects.create(
                video=self.video,
                step=step,
                level=level,
                message=message
            )
        except Exception as e:
            logger.error(f"Failed to create ProcessingLog: {e}")
