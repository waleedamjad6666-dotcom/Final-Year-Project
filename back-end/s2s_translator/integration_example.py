"""
Integration Example: SeamlessM4T Translation + OpenVoice v2 Voice Cloning

This script demonstrates the complete pipeline:
1. Speech-to-Speech translation using SeamlessM4T v2
2. Voice cloning using OpenVoice v2 (clone original speaker onto translated speech)
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def translate_and_clone_voice(
    video_path: str,
    source_audio_path: str,
    target_lang: str = "Urdu",
    seamless_model_path: str = None,
    openvoice_model_path: str = None,
    output_dir: str = None
):
    """
    Complete pipeline: Extract audio → Translate → Clone voice
    
    Args:
        video_path: Path to original video file
        source_audio_path: Path to extracted source audio WAV
        target_lang: Target language for translation
        seamless_model_path: Path to SeamlessM4T model
        openvoice_model_path: Path to OpenVoice v2 checkpoints
        output_dir: Directory for outputs
        
    Returns:
        Dictionary with paths to translated and cloned audio
    """
    from s2s_translator.services import SeamlessTranslationEngine
    from s2s_translator.voice_cloner import VoiceCloner
    
    logger.info("="*80)
    logger.info("COMPLETE PIPELINE: TRANSLATION + VOICE CLONING")
    logger.info("="*80)
    
    # Set default paths
    if seamless_model_path is None:
        seamless_model_path = Path(__file__).parent.parent.parent / "models" / "seamless-m4t-v2-large"
    
    if openvoice_model_path is None:
        openvoice_model_path = Path(__file__).parent.parent.parent / "extras" / "OpenVoice" / "checkpoints_v2"
    
    if output_dir is None:
        output_dir = Path(source_audio_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ========================================================================
        # PHASE 1: SPEECH-TO-SPEECH TRANSLATION (SeamlessM4T v2)
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: SPEECH-TO-SPEECH TRANSLATION")
        logger.info("="*80)
        
        # Initialize translation engine
        logger.info(f"Initializing SeamlessM4T engine (target: {target_lang})...")
        engine = SeamlessTranslationEngine(
            model_path=str(seamless_model_path),
            target_lang=target_lang
        )
        
        # Detect speech segments (needed for voice cloning later)
        logger.info("Detecting speech segments with Silero VAD...")
        speech_segments = engine._detect_speech_segments(source_audio_path)
        logger.info(f"✓ Detected {len(speech_segments)} speech segments")
        
        # Perform translation
        logger.info("Translating audio...")
        translation_result = engine.translate_audio(
            source_audio_path=source_audio_path,
            original_video_path=video_path
        )
        
        if not translation_result['success']:
            raise Exception(f"Translation failed: {translation_result.get('error', 'Unknown error')}")
        
        translated_audio_path = translation_result['translated_audio_path']
        logger.info(f"✓ Translation complete: {translated_audio_path}")
        
        # Cleanup engine to free VRAM
        engine.cleanup()
        
        # ========================================================================
        # PHASE 2: VOICE CLONING (OpenVoice v2)
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: VOICE CLONING")
        logger.info("="*80)
        
        # Initialize voice cloner
        logger.info("Initializing OpenVoice v2 cloner...")
        cloner = VoiceCloner(model_path=str(openvoice_model_path))
        
        # Apply voice cloning
        cloned_audio_path = output_dir / "cloned_audio.wav"
        logger.info("Cloning original speaker's voice onto translated speech...")
        
        cloned_audio = cloner.clone_voice(
            original_audio_path=source_audio_path,
            translated_audio_path=translated_audio_path,
            segments=speech_segments,
            output_path=str(cloned_audio_path)
        )
        
        logger.info(f"✓ Voice cloning complete: {cloned_audio_path}")
        
        # Cleanup cloner
        cloner.cleanup()
        
        # ========================================================================
        # FINAL RESULTS
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE - RESULTS")
        logger.info("="*80)
        logger.info(f"1. Original audio: {source_audio_path}")
        logger.info(f"2. Translated audio (SeamlessM4T): {translated_audio_path}")
        logger.info(f"3. Cloned audio (OpenVoice): {cloned_audio_path}")
        logger.info(f"\nFinal output with original speaker's voice: {cloned_audio_path}")
        logger.info("="*80 + "\n")
        
        return {
            "success": True,
            "original_audio": source_audio_path,
            "translated_audio": translated_audio_path,
            "cloned_audio": str(cloned_audio_path),
            "segments_count": len(speech_segments)
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """
    Example usage from command line.
    
    Usage:
        python integration_example.py <video_path> <audio_path> <target_lang>
    
    Example:
        python integration_example.py video.mp4 audio.wav Urdu
    """
    if len(sys.argv) < 4:
        print("Usage: python integration_example.py <video_path> <audio_path> <target_lang>")
        print("\nExample:")
        print("  python integration_example.py video.mp4 source_audio.wav Urdu")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    target_lang = sys.argv[3]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    result = translate_and_clone_voice(
        video_path=video_path,
        source_audio_path=audio_path,
        target_lang=target_lang
    )
    
    if result['success']:
        print(f"\n✓ SUCCESS!")
        print(f"  Final cloned audio: {result['cloned_audio']}")
    else:
        print(f"\n✗ FAILED: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
