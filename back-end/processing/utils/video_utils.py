"""
Video processing utilities for video dubbing pipeline.
Based on Murtaza's video_utils strategy using ffmpeg.
"""
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_still_image(video_path, output_image_path, timestamp="00:00:00.100"):
    """
    Extract a still frame from video at specified timestamp.
    
    Args:
        video_path: Path to input video file
        output_image_path: Path to output image file
        timestamp: Timestamp to extract frame from (default: 0.1 seconds)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        video_path = str(video_path)
        output_image_path = str(output_image_path)
        
        # Ensure output directory exists
        Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-ss", timestamp,
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            output_image_path,
            "-y"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Still image extracted: {output_image_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error extracting still image: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error extracting still image: {str(e)}")
        return False


def get_video_info(video_path):
    """
    Get basic video information (resolution, fps, codec).
    
    Args:
        video_path: Path to video file
    
    Returns:
        dict: Video information or empty dict if error
    """
    try:
        video_path = str(video_path)
        
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-of", "json",
            video_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        import json
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            return {
                'width': stream.get('width'),
                'height': stream.get('height'),
                'fps': stream.get('r_frame_rate'),
                'codec': stream.get('codec_name')
            }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return {}


def get_duration(video_path):
    """
    Get video duration in seconds.
    
    Args:
        video_path: Path to video file
    
    Returns:
        float: Duration in seconds, or 0.0 if error
    """
    try:
        video_path = str(video_path)
        
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = float(result.stdout.strip())
        return duration
        
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        return 0.0
