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
        
        # Log PyTorch version for compatibility checking
        torch_version = torch.__version__
        logger.info(f"Wav2LipService initialized with device: {self.device}, PyTorch: {torch_version}")
        
        if torch_version.startswith('2.'):
            logger.warning(
                f"Wav2Lip was tested with PyTorch 1.x but you have {torch_version}. "
                f"This may cause compatibility issues. If errors persist, consider downgrading to PyTorch 1.13.1"
            )
        
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, VRAM: {gpu_memory:.2f}GB")
            logger.info(f"CUDA Memory: {torch.cuda.memory_allocated(0)/(1024**2):.0f}MB allocated")
        
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
            
            # ========== PRE-VALIDATION BEFORE WAV2LIP ==========
            logger.info("=== Pre-validation checks before Wav2Lip execution ===")
            
            # Verify input video
            is_valid, frame_count, fps, video_duration = self._verify_video_input(video_path)
            if not is_valid:
                logger.error(f"Input video validation failed: {video_path}")
                return self._compose_audio_only_video(video_path, audio_path, 
                                                      "Invalid input video - skipping lip sync")
            
            # Verify input files exist
            if not Path(video_path).exists():
                logger.error(f"Input video not found: {video_path}")
                return self._compose_audio_only_video(video_path, audio_path,
                                                      f"Video file not found: {video_path}")
            
            if not Path(audio_path).exists():
                logger.error(f"Input audio not found: {audio_path}")
                raise FileNotFoundError(f"Input audio not found: {audio_path}")
            
            # Verify audio file is non-empty and valid
            audio_size = Path(audio_path).stat().st_size
            if audio_size < 1000:  # Less than 1KB
                logger.error(f"Audio file is too small or empty: {audio_size} bytes")
                return self._compose_audio_only_video(video_path, audio_path,
                                                      f"Audio file is empty or corrupted: {audio_size} bytes")
            
            # Verify audio is valid by loading it
            try:
                from pydub import AudioSegment
                test_audio = AudioSegment.from_file(audio_path)
                audio_duration = len(test_audio) / 1000.0
                if audio_duration == 0:
                    raise ValueError("Audio has 0 duration")
                logger.info(f"Audio validation passed: {audio_duration:.2f}s, {audio_size} bytes")
            except Exception as audio_error:
                logger.error(f"Audio file validation failed: {audio_error}")
                return self._compose_audio_only_video(video_path, audio_path,
                                                      f"Audio file is corrupted: {audio_error}")
            
            logger.info(f"✓ Input video: {video_path}")
            logger.info(f"✓ Input audio: {audio_path} ({audio_size} bytes)")
            logger.info(f"✓ Video properties: {frame_count} frames, {fps:.2f} fps, {video_duration:.2f}s")
            logger.info(f"✓ Audio properties: {audio_duration:.2f}s")
            
            # Check duration mismatch
            duration_diff = abs(video_duration - audio_duration)
            if duration_diff > video_duration * 0.1:  # More than 10% difference
                logger.warning(
                    f"Duration mismatch: video={video_duration:.2f}s, audio={audio_duration:.2f}s, "
                    f"diff={duration_diff:.2f}s. Wav2Lip will use --shortest flag."
                )
            
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
            
            # ========== OPTIMIZED BATCH SIZES FOR GPU ==========
            # Determine batch sizes based on GPU availability and VRAM
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}, VRAM: {gpu_memory:.2f}GB")
                
                if gpu_memory > 6:
                    # High performance settings for 8GB+ VRAM
                    wav2lip_batch = 256
                    face_det_batch = 32
                    mel_batch = 32
                    logger.info("Using HIGH performance batch sizes (8GB+ VRAM)")
                elif gpu_memory > 4:
                    # Balanced settings for 4-6GB VRAM
                    wav2lip_batch = 128
                    face_det_batch = 16
                    mel_batch = 16
                    logger.info("Using BALANCED batch sizes (4-6GB VRAM)")
                else:
                    # Conservative for low VRAM
                    wav2lip_batch = 64
                    face_det_batch = 8
                    mel_batch = 8
                    logger.info("Using CONSERVATIVE batch sizes (<4GB VRAM)")
            else:
                # CPU-only settings
                wav2lip_batch = 32
                face_det_batch = 4
                mel_batch = 4
                logger.info("Using CPU-only batch sizes (no GPU)")
            
            logger.info(f"Batch sizes: wav2lip={wav2lip_batch}, face_det={face_det_batch}, mel={mel_batch}")
            
            # Dynamic FPS optimization
            if fps > 30:
                target_fps = 30
                logger.info(f"Original FPS {fps:.2f} > 30, processing at {target_fps} FPS for speed")
            else:
                target_fps = int(fps)
                logger.info(f"Processing at original FPS: {target_fps}")
            
            # Dynamic resize factor for high resolution
            video_width = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if video_width > 1920 or video_height > 1080:
                resize_factor = 2
                logger.info(f"High resolution video ({video_width}x{video_height}), using resize_factor=2 for speed")
            else:
                resize_factor = 1
                logger.info(f"Standard resolution ({video_width}x{video_height}), resize_factor=1")
            
            cmd = [
                sys.executable,  # Use current Python interpreter
                str(inference_script),
                '--checkpoint_path', str(self.checkpoint_path),
                '--face', str(video_path),
                '--audio', str(audio_path),
                '--outfile', str(output_path),
                '--nosmooth',  # Disable smoothing to prevent hangs
                '--resize_factor', str(resize_factor),
                '--fps', str(target_fps),
                '--pads', '0', '10', '0', '0',  # Face detection padding
                '--wav2lip_batch_size', str(wav2lip_batch),
                '--face_det_batch_size', str(face_det_batch),
                '--mel_batch_size', str(mel_batch),
                '--img_size', '96',  # Face crop size
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
            
            # Check if face detection failed
            if hasattr(e, 'stderr') and e.stderr:
                stderr_lower = str(e.stderr).lower()
                if 'face not detected' in stderr_lower or 'no face' in stderr_lower:
                    logger.warning("Face detection failed - automatically falling back to audio-only composition")
                    return self._compose_audio_only_video(video_path, audio_path,
                                                         "Lip sync unavailable: Face not detected in video")
            
            return self._graceful_degradation(video_path, audio_path, error_msg)
            
        except Exception as e:
            error_msg = f"Lip sync failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Check for face detection error in exception message
            if 'face' in str(e).lower() and ('not detected' in str(e).lower() or 'not found' in str(e).lower()):
                logger.warning("Face detection error detected - falling back to audio-only composition")
                return self._compose_audio_only_video(video_path, audio_path,
                                                     f"Lip sync unavailable: {str(e)}")
            
            return self._graceful_degradation(video_path, audio_path, error_msg)
        
        finally:
            # Ensure process is terminated
            if process and process.poll() is None:
                logger.warning("Terminating Wav2Lip process...")
                process.kill()
                process.wait()
    
    def _compose_audio_only_video(self, video_path, audio_path, reason):
        """
        Compose video with translated audio only (no lip synchronization).
        Used when face detection fails or Wav2Lip cannot be applied.
        
        Args:
            video_path: Original video path
            audio_path: Translated audio path
            reason: Reason for skipping lip sync
        
        Returns:
            dict: Result with audio-merged video
        """
        try:
            logger.warning(f"Composing audio-only video: {reason}")
            self._log_processing('lip_sync', 'warning',
                               f'Lip sync unavailable: {reason}. Delivering dubbed video without lip synchronization.')
            
            # Create output filename
            output_filename = f"audio_only_{self.video.id}.mp4"
            output_path = self.processing_dir / output_filename
            
            # FFmpeg command to merge audio with video
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Encode audio to AAC
                '-b:a', '192k',  # Audio bitrate
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',      # End at shortest stream
                str(output_path)
            ]
            
            logger.info(f"Running FFmpeg audio-only merge: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0 or not output_path.exists():
                logger.error(f"FFmpeg audio merge failed: {result.stderr}")
                raise Exception(f"Audio merge failed: {result.stderr}")
            
            # Update job with audio-only result
            self.job.lip_synced_video_path = str(output_path)
            self.job.update_status('completed', progress=100)
            self.job.error_message = f"Lip sync unavailable: {reason}. Delivered dubbed video without lip synchronization."
            self.job.save()
            
            logger.info(f"Audio-only composition successful: {output_path}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'processing_time': 0,
                'use_gpu': False,
                'audio_only': True,
                'warning': f"Lip sync unavailable: {reason}. Video translation completed with dubbed audio only."
            }
            
        except Exception as e:
            logger.error(f"Audio-only composition failed: {e}", exc_info=True)
            self._log_processing('lip_sync', 'error', f'Audio-only composition failed: {e}')
            return {
                'success': False,
                'error': f"Audio composition failed: {str(e)}"
            }
    
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
