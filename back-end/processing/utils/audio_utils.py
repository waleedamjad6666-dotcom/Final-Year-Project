"""
Audio processing utilities for video dubbing pipeline.
Based on Murtaza's audio_utils strategy using ffmpeg.
"""
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path, output_audio_path):
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_audio_path: Path to output audio file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert paths to strings
        video_path = str(video_path)
        output_audio_path = str(output_audio_path)
        
        # Ensure output directory exists
        Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract audio with high quality
        cmd = [
            "ffmpeg", "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            output_audio_path,
            "-y"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Audio extracted successfully: {output_audio_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting audio: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False


def convert_mp3_to_wav(input_mp3, output_wav):
    """
    Convert MP3 to WAV format with specific audio settings.
    
    Args:
        input_mp3: Path to input MP3 file
        output_wav: Path to output WAV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        input_mp3 = str(input_mp3)
        output_wav = str(output_wav)
        
        # Ensure output directory exists
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", input_mp3,
            "-ar", "44100",  # Sample rate
            "-ac", "2",       # Stereo
            "-c:a", "pcm_s16le",  # Codec
            output_wav,
            "-y"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"MP3 converted to WAV: {output_wav}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error converting MP3 to WAV: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error converting MP3 to WAV: {str(e)}")
        return False


def get_duration(file_path):
    """
    Get duration of audio/video file using ffprobe.
    
    Args:
        file_path: Path to audio or video file
    
    Returns:
        float: Duration in seconds, or 0.0 if error
    """
    try:
        file_path = str(file_path)
        
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = float(result.stdout.strip())
        logger.info(f"Duration of {file_path}: {duration}s")
        return duration
        
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting duration: {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error getting duration: {str(e)}")
        return 0.0


def adjust_speed(input_file, output_file, speed_factor, is_video=True):
    """
    Adjust playback speed of audio or video file.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        speed_factor: Speed multiplication factor (e.g., 2.0 = twice as fast)
        is_video: True for video files, False for audio files
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        input_file = str(input_file)
        output_file = str(output_file)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        if is_video:
            # Adjust video speed
            cmd = [
                "ffmpeg", "-i", input_file,
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                "-an",  # No audio
                output_file,
                "-y"
            ]
        else:
            # Adjust audio speed (handle atempo filter limitations)
            filters = []
            temp_factor = speed_factor
            
            # atempo filter only works between 0.5 and 2.0
            while temp_factor < 0.5:
                filters.append("atempo=0.5")
                temp_factor /= 0.5
            
            while temp_factor > 2.0:
                filters.append("atempo=2.0")
                temp_factor /= 2.0
            
            filters.append(f"atempo={round(temp_factor, 3)}")
            
            cmd = [
                "ffmpeg", "-i", input_file,
                "-filter:a", ",".join(filters),
                "-vn",  # No video
                output_file,
                "-y"
            ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Speed adjusted successfully: {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error adjusting speed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error adjusting speed: {str(e)}")
        return False
