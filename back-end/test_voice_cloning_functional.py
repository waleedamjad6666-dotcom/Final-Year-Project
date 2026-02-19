"""
Test script to verify voice cloning actually transfers tone from original to translated audio
"""
import sys
import os
import django
import logging

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio(output_path, duration=5.0, frequency=440, sr=16000):
    """Create a simple test audio file with a pure tone"""
    t = np.linspace(0, duration, int(duration * sr))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    sf.write(output_path, audio, sr)
    logger.info(f"Created test audio: {output_path} ({duration}s, {frequency}Hz)")
    return output_path

def test_voice_cloning():
    """Test the complete voice cloning pipeline"""
    logger.info("="*80)
    logger.info("TESTING VOICE CLONING PIPELINE")
    logger.info("="*80)
    
    try:
        # Import voice cloner
        from s2s_translator.voice_cloner import VoiceCloner
        logger.info("✅ VoiceCloner imported")
        
        # Initialize
        cloner = VoiceCloner()
        logger.info(f"✅ VoiceCloner initialized (device: {cloner.device})")
        
        # Create test audio files
        test_dir = Path("test_voice_cloning_data")
        test_dir.mkdir(exist_ok=True)
        
        # Original audio (440Hz tone - will be reference voice)
        original_path = test_dir / "original.wav"
        create_test_audio(original_path, duration=3.0, frequency=440, sr=16000)
        
        # Translated audio (880Hz tone - different voice to be cloned onto)
        translated_path = test_dir / "translated.wav"
        create_test_audio(translated_path, duration=3.0, frequency=880, sr=16000)
        
        # Create fake speech segments (covering most of the audio)
        segments = [
            {'start': 0.5, 'end': 2.5, 'duration': 2.0}
        ]
        
        # Output path
        output_path = test_dir / "cloned_output.wav"
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING VOICE CLONING")
        logger.info("="*80)
        logger.info(f"Original (reference): {original_path}")
        logger.info(f"Translated (to clone): {translated_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Segments: {segments}")
        
        # Run voice cloning
        result = cloner.clone_voice(
            original_audio_path=str(original_path),
            translated_audio_path=str(translated_path),
            segments=segments,
            output_path=str(output_path)
        )
        
        # Verify output
        if output_path.exists():
            cloned_audio, sr = librosa.load(str(output_path), sr=None)
            logger.info(f"\n✅ CLONING SUCCESSFUL!")
            logger.info(f"   Output file: {output_path}")
            logger.info(f"   Duration: {len(cloned_audio)/sr:.2f}s")
            logger.info(f"   Sample rate: {sr}Hz")
            logger.info(f"   Non-zero samples: {np.count_nonzero(cloned_audio)}/{len(cloned_audio)}")
            
            # Analyze spectral content to see if voice changed
            original_audio, _ = librosa.load(str(original_path), sr=sr)
            translated_audio, _ = librosa.load(str(translated_path), sr=sr)
            
            logger.info(f"\n📊 SPECTRAL ANALYSIS:")
            logger.info(f"   Original RMS: {np.sqrt(np.mean(original_audio**2)):.4f}")
            logger.info(f"   Translated RMS: {np.sqrt(np.mean(translated_audio**2)):.4f}")
            logger.info(f"   Cloned RMS: {np.sqrt(np.mean(cloned_audio**2)):.4f}")
            
            return True
        else:
            logger.error("❌ Output file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_cloning()
    if success:
        print("\n" + "="*80)
        print("✅ VOICE CLONING TEST PASSED")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ VOICE CLONING TEST FAILED")
        print("="*80)
        sys.exit(1)
