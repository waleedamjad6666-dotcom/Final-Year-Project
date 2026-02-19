"""
Lip Sync Service using Wav2Lip
Integrates Wav2Lip for high-quality lip synchronization
"""
import sys
import os
import logging
import subprocess
from pathlib import Path
import torch
import cv2
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Add Wav2Lip to path
# File is at: back-end/lip_sync/lip_sync_service.py
# Go up 3 levels: lip_sync -> back-end -> FINAL YEAR PROJECT
WAV2LIP_PATH = Path(__file__).parent.parent.parent / "models" / "Wav2Lip"
if WAV2LIP_PATH.exists() and str(WAV2LIP_PATH) not in sys.path:
    sys.path.insert(0, str(WAV2LIP_PATH))


class LipSyncService:
    """
    Service for lip synchronization using Wav2Lip
    Combines video with voice-cloned translated audio
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize Lip Sync Service
        
        Args:
            checkpoint_path: Path to Wav2Lip checkpoint
                           If None, uses default wav2lip_gan.pth
        """
        # Prioritize Apple Silicon MPS, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Initializing LipSyncService with device: {self.device}")
        
        # Set checkpoint path
        if checkpoint_path is None:
            # Use wav2lip_gan for better quality
            # Go up 3 levels: lip_sync -> back-end -> FINAL YEAR PROJECT
            checkpoint_path = str(
                Path(__file__).parent.parent.parent / 
                "models" / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth"
            )
        
        self.checkpoint_path = Path(checkpoint_path)
        
        # Verify checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Wav2Lip checkpoint not found at: {self.checkpoint_path}\n"
                f"Please ensure wav2lip_gan.pth exists in models/Wav2Lip/checkpoints/"
            )
        
        logger.info(f"Using Wav2Lip checkpoint: {self.checkpoint_path}")
        
        # Wav2Lip parameters
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        self.resize_factor = 1
        self.pads = [0, 10, 0, 0]  # top, bottom, left, right
        self.nosmooth = False
        self.img_size = 96
        
    
    def merge_audio_video(
        self, 
        video_path: str, 
        audio_path: str, 
        output_path: str
    ) -> bool:
        """
        Merge voice-cloned translated audio with original video using FFmpeg
        
        Args:
            video_path: Path to source video
            audio_path: Path to voice-cloned translated audio
            output_path: Path to save merged video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Merging audio and video:")
            logger.info(f"  Video: {video_path}")
            logger.info(f"  Audio: {audio_path}")
            logger.info(f"  Output: {output_path}")
            
            # FFmpeg command to merge audio with video
            # -i input video, -i input audio
            # -c:v copy - copy video without re-encoding
            # -c:a aac - encode audio as AAC
            # -map 0:v:0 - use video from first input
            # -map 1:a:0 - use audio from second input
            # -shortest - match length of shortest stream
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-i', video_path,  # Input video
                '-i', audio_path,  # Input audio
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',  # AAC audio codec
                '-b:a', '192k',  # Audio bitrate
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',  # Match shortest stream length
                output_path
            ]
            
            # Run FFmpeg
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
            # Verify output exists
            if not Path(output_path).exists():
                logger.error("Merged video file was not created")
                return False
            
            logger.info(f"✓ Audio-video merge successful: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge audio and video: {e}")
            return False
    
    
    def apply_lip_sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> Dict:
        """
        Apply lip synchronization using Wav2Lip
        
        Args:
            video_path: Path to input video (with translated audio merged)
            audio_path: Path to translated audio for reference
            output_path: Path to save lip-synced video
            
        Returns:
            Dictionary with success status and output path
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("APPLYING LIP SYNCHRONIZATION - WAV2LIP")
            logger.info("="*80)
            logger.info(f"Input video: {video_path}")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Checkpoint: {self.checkpoint_path}")
            logger.info(f"Device: {self.device}")
            
            # Prepare output directory
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Also ensure Wav2Lip temp directory exists
            wav2lip_temp = WAV2LIP_PATH / "temp"
            wav2lip_temp.mkdir(parents=True, exist_ok=True)
            
            # Build inference command
            inference_script = WAV2LIP_PATH / "inference.py"
            
            if not inference_script.exists():
                raise FileNotFoundError(f"Wav2Lip inference.py not found at {inference_script}")
            
            # Command to run Wav2Lip inference
            command = [
                sys.executable,
                str(inference_script),
                '--checkpoint_path', str(self.checkpoint_path),
                '--face', str(video_path),
                '--audio', str(audio_path),
                '--outfile', str(output_path),
                '--face_det_batch_size', str(self.face_det_batch_size),
                '--wav2lip_batch_size', str(self.wav2lip_batch_size),
                '--resize_factor', str(self.resize_factor),
                '--pads'] + [str(p) for p in self.pads]
            
            if self.nosmooth:
                command.append('--nosmooth')
            
            logger.info(f"Running Wav2Lip inference...")
            logger.info(f"Command: {' '.join(command)}")
            
            # Run Wav2Lip inference
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(WAV2LIP_PATH)
            )
            
            # Log output
            if result.stdout:
                logger.info(f"Wav2Lip output:\n{result.stdout}")
            
            if result.stderr:
                logger.warning(f"Wav2Lip stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Wav2Lip failed with return code {result.returncode}")
                return {
                    'success': False,
                    'error': f"Wav2Lip inference failed with return code {result.returncode}: {result.stderr}"
                }
            
            # Wait for file to be written (FFmpeg in inference.py needs time)
            import time
            max_wait = 10  # seconds
            wait_interval = 0.5
            elapsed = 0
            
            while elapsed < max_wait:
                if Path(output_path).exists():
                    # File found, wait a bit more to ensure it's fully written
                    time.sleep(1)
                    break
                time.sleep(wait_interval)
                elapsed += wait_interval
            
            # Verify output exists
            if not Path(output_path).exists():
                logger.error(f"Lip-synced video was not created at: {output_path}")
                logger.error("Check if FFmpeg is installed and accessible")
                return {
                    'success': False,
                    'error': "Output video file not created. Ensure FFmpeg is installed."
                }
            
            # Get output file size
            output_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            
            logger.info("="*80)
            logger.info("LIP SYNC COMPLETE")
            logger.info("="*80)
            logger.info(f"✓ Output: {output_path}")
            logger.info(f"✓ Size: {output_size:.2f} MB")
            logger.info("="*80 + "\n")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'file_size_mb': output_size
            }
            
        except Exception as e:
            logger.error(f"Lip sync failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    
    def process_video(
        self,
        source_video_path: str,
        translated_audio_path: str,
        output_video_path: str,
        temp_dir: Optional[str] = None
    ) -> Dict:
        """
        Complete pipeline: Merge audio + Apply lip sync
        
        Args:
            source_video_path: Original video
            translated_audio_path: Voice-cloned translated audio
            output_video_path: Final lip-synced video output
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            Dictionary with success status and paths
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("COMPLETE LIP SYNC PIPELINE")
            logger.info("="*80)
            
            # Setup temp directory
            if temp_dir is None:
                temp_dir = Path(output_video_path).parent / "temp"
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Merge audio with video
            merged_video_path = temp_dir / "merged_video.mp4"
            logger.info("\nSTEP 1: Merging translated audio with video...")
            
            merge_success = self.merge_audio_video(
                video_path=source_video_path,
                audio_path=translated_audio_path,
                output_path=str(merged_video_path)
            )
            
            if not merge_success:
                return {
                    'success': False,
                    'error': 'Failed to merge audio and video'
                }
            
            # Step 2: Apply lip synchronization
            logger.info("\nSTEP 2: Applying lip synchronization...")
            
            lipsync_result = self.apply_lip_sync(
                video_path=str(merged_video_path),
                audio_path=translated_audio_path,
                output_path=output_video_path
            )
            
            if not lipsync_result['success']:
                return lipsync_result
            
            # Cleanup temp files
            try:
                if merged_video_path.exists():
                    merged_video_path.unlink()
                logger.info("✓ Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Could not clean up temp files: {e}")
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETE")
            logger.info("="*80)
            logger.info(f"✓ Final video: {output_video_path}")
            logger.info("="*80 + "\n")
            
            return {
                'success': True,
                'output_path': output_video_path,
                'file_size_mb': lipsync_result.get('file_size_mb', 0)
            }
            
        except Exception as e:
            logger.error(f"Complete pipeline failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
