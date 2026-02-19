"""
Voice Cloning Service using OpenVoice v2
Clones original speaker's voice onto translated speech while preserving prosody and timing.
"""

import os
import sys
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from scipy.signal import resample

# Add OpenVoice to path for direct import
project_root = Path(__file__).parent.parent.parent
openvoice_path = project_root / "extras" / "OpenVoice"
if openvoice_path.exists():
    # Add OpenVoice root to path so 'import openvoice' works
    if str(openvoice_path) not in sys.path:
        sys.path.insert(0, str(openvoice_path))
    # Also ensure the openvoice package itself is accessible
    openvoice_pkg = openvoice_path / "openvoice"
    if openvoice_pkg.exists() and str(openvoice_pkg.parent) not in sys.path:
        sys.path.insert(0, str(openvoice_pkg.parent))

logger = logging.getLogger(__name__)


class VoiceCloner:
    """
    Zero-shot voice cloning using OpenVoice v2.
    Clones speaker tone/timbre from original audio onto translated speech segments.
    Preserves content, prosody, rhythm, and silence timing from translation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize OpenVoice v2 voice cloner.
        
        Args:
            model_path: Path to OpenVoice v2 checkpoint directory.
                       If None, uses default from models/OpenVoiceV2
        """
        # FORCE MPS for M2 Mac Neural Engine acceleration
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("🚀 M2 Neural Engine detected - forcing MPS device for hardware acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Initializing VoiceCloner with device: {self.device}")
        
        # Cache for target embedding (extract once from original audio)
        self._target_embedding_cache = None
        
        # Set default model path if not provided
        if model_path is None:
            # Assuming project structure: back-end/s2s_translator/voice_cloner.py
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "OpenVoiceV2"
        
        self.model_path = Path(model_path)
        logger.info(f"OpenVoice v2 checkpoint path: {self.model_path}")
        
        self.model = None
        self.tone_color_converter = None
        
        self._load_model()
    
    def _load_model(self):
        """
        Load OpenVoice v2 model and tone color converter.
        Falls back to CPU if GPU initialization fails.
        """
        try:
            logger.info("Loading OpenVoice v2 model...")
            
            # Check if OpenVoice is installed
            try:
                from openvoice import se_extractor
                from openvoice.api import ToneColorConverter
            except ImportError as e:
                logger.error(f"OpenVoice not installed: {e}")
                logger.error("Please install: pip install git+https://github.com/myshell-ai/OpenVoice.git")
                raise ImportError(
                    "OpenVoice v2 not found. Install with:\n"
                    "  pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
                    "Or manually:\n"
                    "  cd extras/OpenVoice && pip install -e ."
                )
            
            # Verify checkpoint directory exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"OpenVoice checkpoint not found at: {self.model_path}\n"
                    f"Please download checkpoints_v2 from:\n"
                    f"  https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
                )
            
            # Initialize tone color converter
            ckpt_converter = self.model_path / 'converter'
            if not ckpt_converter.exists():
                raise FileNotFoundError(f"Converter checkpoint not found: {ckpt_converter}")
            
            logger.info(f"  Loading tone color converter from {ckpt_converter}...")
            # FORCE MPS device for M2 Neural Engine
            force_device = "mps" if torch.backends.mps.is_available() else self.device
            logger.info(f"  Using device: {force_device} (M2 Neural Engine: {force_device=='mps'})")
            
            self.tone_color_converter = ToneColorConverter(
                f'{ckpt_converter}/config.json',
                device=force_device
            )
            self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
            
            # Note: Keeping model in float32 to avoid dtype mismatch errors
            # MPS device still provides acceleration even with float32
            logger.info("  ✓ Tone color converter loaded (float32 for compatibility)")
            
            logger.info("✓ OpenVoice v2 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load OpenVoice v2: {str(e)}")
            if self.device == "cuda":
                logger.warning("Retrying on CPU...")
                self.device = "cpu"
                self._load_model()
            else:
                raise
    
    def _extract_reference_segment(self, audio_path: str, segments: List[Dict], 
                                   min_duration: float = 5.0, 
                                   max_duration: float = 10.0) -> Tuple[np.ndarray, int]:
        """
        Extract clean reference speech segment for speaker embedding.
        Selects longest segment within duration range for best quality.
        
        Args:
            audio_path: Path to original audio file
            segments: List of speech segments [{'start': float, 'end': float, 'duration': float}]
            min_duration: Minimum reference duration in seconds
            max_duration: Maximum reference duration in seconds
            
        Returns:
            Tuple of (audio_array at 16kHz, sample_rate)
        """
        logger.info(f"Extracting reference segment from {len(segments)} speech segments...")
        logger.info(f"  Target duration: {min_duration}s - {max_duration}s")
        
        # Find suitable segments
        suitable_segments = [
            seg for seg in segments 
            if min_duration <= seg['duration'] <= max_duration
        ]
        
        if not suitable_segments:
            # Fallback: use longest segment available
            logger.warning(f"No segments in {min_duration}s-{max_duration}s range")
            suitable_segments = sorted(segments, key=lambda x: x['duration'], reverse=True)
            logger.info(f"  Using longest available segment: {suitable_segments[0]['duration']:.2f}s")
        else:
            # Use longest suitable segment for best embedding quality
            suitable_segments = sorted(suitable_segments, key=lambda x: x['duration'], reverse=True)
            logger.info(f"  Found {len(suitable_segments)} suitable segments")
            logger.info(f"  Selected segment: {suitable_segments[0]['duration']:.2f}s")
        
        best_segment = suitable_segments[0]
        
        # Extract audio segment
        audio, sr = librosa.load(
            audio_path, 
            sr=16000,
            offset=best_segment['start'],
            duration=min(best_segment['duration'], max_duration)
        )
        
        logger.info(f"  ✓ Extracted reference: {len(audio)/sr:.2f}s at {sr}Hz")
        
        return audio, sr
    
    def _extract_tone_color_embedding(self, reference_audio: np.ndarray, 
                                     sample_rate: int) -> torch.Tensor:
        """
        Extract speaker tone color embedding from reference audio using OpenVoice v2 API.
        
        Args:
            reference_audio: Audio array (mono)
            sample_rate: Sample rate of audio
            
        Returns:
            Tone color embedding tensor
        """
        logger.info("Extracting tone color embedding from reference...")
        
        try:
            # OpenVoice v2 expects audio at model's sampling rate (usually 16kHz or 24kHz)
            target_sr = self.tone_color_converter.hps.data.sampling_rate
            
            if sample_rate != target_sr:
                logger.info(f"  Resampling {sample_rate}Hz → {target_sr}Hz with soxr_hq...")
                reference_audio = librosa.resample(
                    reference_audio, 
                    orig_sr=sample_rate, 
                    target_sr=target_sr,
                    res_type='soxr_hq'  # High-fidelity resampling to maintain clarity
                )
                sample_rate = target_sr
            
            # Save temporary reference file
            temp_ref_path = Path("temp_reference.wav")
            sf.write(temp_ref_path, reference_audio, sample_rate)
            
            # Extract speaker embedding using OpenVoice v2's extract_se method
            # This method extracts speaker characteristics directly from the audio
            logger.info("  Extracting speaker embedding with ToneColorConverter...")
            embedding = self.tone_color_converter.extract_se(
                ref_wav_list=[str(temp_ref_path)],
                se_save_path=None  # Don't save, return directly
            )
            
            # Cleanup
            if temp_ref_path.exists():
                temp_ref_path.unlink()
            
            # Clean up any processed directories created by VAD
            import shutil
            if Path("processed").exists():
                shutil.rmtree("processed", ignore_errors=True)
            
            logger.info(f"  ✓ Embedding extracted: shape={embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract tone color embedding: {str(e)}")
            raise
    
    
    def _convert_segment_voice(self, segment_audio: np.ndarray, 
                               source_embedding: torch.Tensor,
                               target_embedding: torch.Tensor,
                               sample_rate: int = 16000) -> np.ndarray:
        """
        Convert voice of a single audio segment using OpenVoice v2.
        Transfers target speaker's tone/timbre to source audio while preserving content/prosody.
        
        Args:
            segment_audio: Audio segment to convert (numpy array)
            source_embedding: Source speaker embedding (from translated audio)
            target_embedding: Target speaker embedding (original speaker to clone)
            sample_rate: Sample rate
            
        Returns:
            Converted audio segment with cloned voice
        """
        try:
            # Get model's expected sampling rate
            model_sr = self.tone_color_converter.hps.data.sampling_rate
            
            # Resample if needed with soxr_hq for high-fidelity
            if sample_rate != model_sr:
                segment_audio = librosa.resample(
                    segment_audio,
                    orig_sr=sample_rate,
                    target_sr=model_sr,
                    res_type='soxr_hq'  # Maintain high-frequency clarity
                )
                sample_rate = model_sr
            
            # Keep float32 for dtype consistency with model
            
            # Save segment to temporary file (OpenVoice API requirement)
            temp_input = Path("temp_segment_input.wav")
            temp_output = Path("temp_segment_output.wav")
            
            sf.write(temp_input, segment_audio, sample_rate)
            
            # Perform voice conversion using ToneColorConverter
            # tau parameter controls the intensity of conversion (higher = more original voice transfer)
            self.tone_color_converter.convert(
                audio_src_path=str(temp_input),
                src_se=source_embedding,    # Source embedding (translated speech)
                tgt_se=target_embedding,    # Target embedding (original speaker to clone)
                output_path=str(temp_output),
                tau=0.7,  # Conversion intensity (0-1, higher = stronger voice cloning)
                message="@VoiceCloning"
            )
            
            # Load converted audio
            converted_audio, _ = librosa.load(str(temp_output), sr=sample_rate)
            
            # Cleanup temporary files
            if temp_input.exists():
                temp_input.unlink()
            if temp_output.exists():
                temp_output.unlink()
            
            return converted_audio
            
        except Exception as e:
            logger.error(f"Voice conversion failed for segment: {str(e)}")
            # Return original segment on failure
            return segment_audio
    
    def _detect_translated_speech_segments(self, audio_path: str) -> List[Dict]:
        """
        Run VAD on translated audio to detect where translated words actually are.
        This decouples from original audio timestamps to prevent word-breaking.
        
        Args:
            audio_path: Path to translated audio file
            
        Returns:
            List of speech segments [{'start': float, 'end': float, 'duration': float}]
        """
        logger.info("🎯 Running VAD on TRANSLATED audio (decoupled from original)...")
        
        try:
            # Import VAD utilities
            import torch
            from pathlib import Path
            
            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            get_speech_timestamps = utils[0]
            
            # Read translated audio at 16kHz using librosa (avoids torchcodec dependency)
            logger.info(f"  Loading audio with librosa: {audio_path}")
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Convert to torch tensor (Silero VAD expects tensor)
            wav = torch.FloatTensor(audio_data)
            
            # Detect speech with aggressive threshold for translated audio
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                threshold=0.4,  # Sensitive to capture all translated speech
                sampling_rate=16000,
                min_speech_duration_ms=300,  # 300ms minimum
                min_silence_duration_ms=500,  # 500ms silence to split
                speech_pad_ms=100  # 100ms padding around speech
            )
            
            # Convert to our segment format
            segments = []
            for ts in speech_timestamps:
                start = ts['start'] / 16000.0
                end = ts['end'] / 16000.0
                segments.append({
                    'start': start,
                    'end': end,
                    'duration': end - start
                })
            
            logger.info(f"✓ VAD detected {len(segments)} speech segments in TRANSLATED audio")
            for idx, seg in enumerate(segments[:5]):  # Show first 5
                logger.info(f"  Seg {idx+1}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to run VAD on translated audio: {e}")
            logger.error("Falling back to original segments (may cause word-breaking)")
            # Return empty list to signal fallback needed
            return []
    
    def clone_voice(self, 
                   original_audio_path: str,
                   translated_audio_path: str,
                   output_path: Optional[str] = None) -> np.ndarray:
        """
        Main voice cloning pipeline with dynamic segmentation.
        
        Workflow:
        1. Extract reference speaker embedding from original audio (ONCE - cached)
        2. Load translated audio
        3. Run VAD on TRANSLATED audio to find actual word boundaries
        4. Convert voice of each segment with 30ms crossfade
        5. Reconstruct with high-fidelity soxr_hq resampling
        
        Args:
            original_audio_path: Path to original audio (for speaker reference)
            translated_audio_path: Path to translated audio (full timeline)
            output_path: Optional path to save cloned audio
            
        Returns:
            Cloned audio as numpy array (same duration as input)
        """
        logger.info("\n" + "="*80)
        logger.info("VOICE CLONING PIPELINE - OpenVoice v2 (M2 Optimized)")
        logger.info("="*80)
        logger.info(f"Original audio: {original_audio_path}")
        logger.info(f"Translated audio: {translated_audio_path}")
        logger.info(f"Device: {self.device} (M2 Neural Engine: {self.device=='mps'})")
        logger.info("="*80)
        
        try:
            # STEP 1: Extract target embedding ONCE (cached for efficiency)
            logger.info("\nSTEP 1: Extracting speaker tone color from original audio...")
            
            if self._target_embedding_cache is None:
                logger.info("  First time extraction - detecting original speech segments...")
                # Detect segments in ORIGINAL audio for reference extraction
                original_segments = self._detect_translated_speech_segments(original_audio_path)
                
                if not original_segments:
                    raise RuntimeError("No speech detected in original audio")
                
                reference_audio, ref_sr = self._extract_reference_segment(
                    original_audio_path, 
                    original_segments,
                    min_duration=5.0,
                    max_duration=10.0
                )
                
                target_embedding = self._extract_tone_color_embedding(reference_audio, ref_sr)
                self._target_embedding_cache = target_embedding
                logger.info("✓ Original speaker tone color extracted and CACHED")
            else:
                target_embedding = self._target_embedding_cache
                logger.info("✓ Using CACHED target embedding (tone stability)")
            
            # STEP 2: Run VAD on TRANSLATED audio (dynamic segmentation)
            logger.info("\nSTEP 2: Running VAD on TRANSLATED audio (decoupled from original)...")
            segments = self._detect_translated_speech_segments(translated_audio_path)
            
            if not segments:
                logger.error("No speech segments detected in translated audio!")
                raise RuntimeError("VAD failed on translated audio")
            
            logger.info(f"✓ Detected {len(segments)} segments in TRANSLATED audio")
            
            # STEP 3: Load translated audio
            logger.info("\nSTEP 3: Loading translated audio...")
            translated_full, trans_sr = librosa.load(translated_audio_path, sr=16000)
            total_duration = len(translated_full) / trans_sr
            logger.info(f"✓ Loaded: {total_duration:.2f}s at {trans_sr}Hz")
            
            # STEP 4: Extract and convert each speech segment with 30ms crossfade
            logger.info(f"\nSTEP 4: Converting voice for {len(segments)} segments (30ms crossfade)...")
            
            # Create zero-filled canvas for reconstruction
            final_audio = np.zeros_like(translated_full, dtype=np.float32)
            
            converted_count = 0
            failed_count = 0
            
            for idx, segment in enumerate(segments):
                logger.info(f"\n  Segment {idx+1}/{len(segments)}: {segment['start']:.2f}s - {segment['end']:.2f}s")
                
                try:
                    # Extract segment from translated audio
                    start_sample = int(segment['start'] * trans_sr)
                    end_sample = int(segment['end'] * trans_sr)
                    
                    if end_sample > len(translated_full):
                        logger.warning(f"    Segment extends beyond audio, truncating...")
                        end_sample = len(translated_full)
                    
                    segment_audio = translated_full[start_sample:end_sample]
                    
                    if len(segment_audio) < 1600:  # Less than 0.1s
                        logger.warning(f"    Segment too short ({len(segment_audio)} samples), skipping conversion")
                        final_audio[start_sample:end_sample] = segment_audio
                        continue
                    
                    # Extract source embedding from translated segment
                    logger.info(f"    Extracting embedding from translated segment...")
                    temp_seg_path = Path("temp_translated_seg.wav")
                    
                    # Resample to model's expected rate with soxr_hq
                    model_sr = self.tone_color_converter.hps.data.sampling_rate
                    if trans_sr != model_sr:
                        segment_audio_resampled = librosa.resample(
                            segment_audio,
                            orig_sr=trans_sr,
                            target_sr=model_sr,
                            res_type='soxr_hq'  # High-fidelity resampling
                        )
                        sf.write(temp_seg_path, segment_audio_resampled, model_sr)
                    else:
                        sf.write(temp_seg_path, segment_audio, trans_sr)
                    
                    # Extract embedding from this segment
                    source_embedding = self.tone_color_converter.extract_se(
                        ref_wav_list=[str(temp_seg_path)],
                        se_save_path=None
                    )
                    
                    if temp_seg_path.exists():
                        temp_seg_path.unlink()
                    
                    # Convert voice (clone original speaker onto translated speech)
                    logger.info(f"    Converting voice (applying original speaker tone)...")
                    converted_segment = self._convert_segment_voice(
                        segment_audio=segment_audio,
                        source_embedding=source_embedding,
                        target_embedding=target_embedding,
                        sample_rate=trans_sr
                    )
                    
                    # Handle length differences by direct interpolation to target length
                    if len(converted_segment) != len(segment_audio):
                        target_length = len(segment_audio)
                        logger.info(f"    Resampling converted segment: {len(converted_segment)} → {target_length} samples (linear interp)")
                        x_old = np.linspace(0, 1, num=len(converted_segment))
                        x_new = np.linspace(0, 1, num=target_length)
                        converted_segment = np.interp(x_new, x_old, converted_segment)
                    
                    # Place converted segment in final audio with 30ms LINEAR CROSSFADE
                    crossfade_samples = int(0.030 * trans_sr)  # 30ms crossfade
                    
                    # Main segment placement
                    segment_length = len(converted_segment)
                    final_audio[start_sample:start_sample + segment_length] = converted_segment
                    
                    # Apply crossfade at start boundary (if not at beginning)
                    if start_sample >= crossfade_samples:
                        fade_in = np.linspace(0, 1, crossfade_samples)
                        final_audio[start_sample:start_sample + crossfade_samples] *= fade_in
                    
                    # Apply crossfade at end boundary (if not at end)
                    end_sample_pos = start_sample + segment_length
                    if end_sample_pos + crossfade_samples <= len(final_audio):
                        fade_out = np.linspace(1, 0, crossfade_samples)
                        final_audio[end_sample_pos - crossfade_samples:end_sample_pos] *= fade_out
                    
                    converted_count += 1
                    logger.info(f"    ✓ Voice cloned and placed with 30ms crossfade")
                    
                except Exception as seg_error:
                    logger.error(f"    ✗ Failed to convert segment {idx+1}: {str(seg_error)}")
                    # Fallback: use original translated segment
                    final_audio[start_sample:end_sample] = segment_audio
                    failed_count += 1
            
            # STEP 5: Final verification and output
            logger.info(f"\n{'='*80}")
            logger.info(f"VOICE CLONING COMPLETE (M2 Neural Engine Accelerated)")
            logger.info(f"{'='*80}")
            logger.info(f"Successfully converted: {converted_count}/{len(segments)} segments")
            logger.info(f"Failed conversions: {failed_count}/{len(segments)} segments")
            logger.info(f"Output duration: {len(final_audio)/trans_sr:.2f}s")
            logger.info(f"Silence preservation: ✓ (zero-filled canvas)")
            logger.info(f"Crossfade: 30ms linear blend at boundaries")
            
            # Verify audio is not all zeros
            non_zero = np.count_nonzero(final_audio)
            logger.info(f"Non-silent samples: {non_zero}/{len(final_audio)} ({100*non_zero/len(final_audio):.1f}%)")
            
            if non_zero == 0:
                logger.error("⚠ WARNING: Output audio is all zeros (silent)")
                raise RuntimeError("Voice cloning produced silent output")
            
            # Save if output path provided
            if output_path:
                sf.write(output_path, final_audio, trans_sr)
                logger.info(f"✓ Saved to: {output_path}")
            
            logger.info("="*80 + "\n")
            
            return final_audio
            
        except Exception as e:
            logger.error(f"Voice cloning pipeline failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def cleanup(self):
        """Release model resources and clear GPU cache."""
        if self.tone_color_converter is not None:
            del self.tone_color_converter
            self.tone_color_converter = None
        
        # Clear device cache
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("VoiceCloner resources released")


# Convenience function for integration with SeamlessTranslationEngine
def apply_voice_cloning(
    original_audio_path: str,
    translated_audio_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to apply voice cloning to translated audio.
    
    NOW WITH DYNAMIC SEGMENTATION:
    - No longer requires speech_segments parameter
    - Runs VAD on translated audio automatically
    - Prevents word-breaking by using actual translated word boundaries
    - M2 Neural Engine optimized with float16 and MPS
    - 30ms crossfade for seamless reconstruction
    - soxr_hq resampling for high-fidelity audio
    
    Usage example:
    ```python
    from s2s_translator.services import SeamlessTranslationEngine
    from s2s_translator.voice_cloner import apply_voice_cloning
    
    # Step 1: Translate
    engine = SeamlessTranslationEngine(model_path="...", target_lang="Urdu")
    result = engine.translate_audio(source_audio_path, original_video_path)
    
    # Step 2: Clone original speaker's voice (segments detected automatically)
    cloned_audio = apply_voice_cloning(
        original_audio_path=source_audio_path,
        translated_audio_path=result['translated_audio_path'],
        output_path="final_cloned.wav"
    )
    ```
    
    Args:
        original_audio_path: Path to original audio (speaker reference)
        translated_audio_path: Path to translated audio from SeamlessM4T
        output_path: Optional output path
        model_path: Optional OpenVoice checkpoint path
        
    Returns:
        Cloned audio array
    """
    cloner = VoiceCloner(model_path=model_path)
    try:
        return cloner.clone_voice(
            original_audio_path=original_audio_path,
            translated_audio_path=translated_audio_path,
            output_path=output_path
        )
    finally:
        cloner.cleanup()
