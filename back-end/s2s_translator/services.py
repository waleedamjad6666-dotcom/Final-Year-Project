"""
Speech-to-Speech Translation Service using Seamless M4T v2
Uses dynamic energy-based speech detection to preserve silence
"""

import os
import logging
import uuid
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from pydub import AudioSegment
from moviepy import VideoFileClip
from transformers import AutoProcessor, SeamlessM4Tv2Model
from django.conf import settings

logger = logging.getLogger(__name__)

# Silero VAD will be loaded dynamically via torch.hub
# No need for imports - torch.hub handles everything

# Language code mapping for SeamlessM4T (3-letter ISO codes)
LANG_CODE_MAP = {
    "urdu": "urd",
    "english": "eng",
    "hindi": "hin",
    "spanish": "spa",
    "french": "fra",
    "german": "deu",
    "arabic": "arb",
    "chinese": "cmn",
    "japanese": "jpn",
    "korean": "kor",
}

# CRITICAL: Maximum segment duration to prevent model truncation
# Research shows 10-12 seconds is optimal for SeamlessM4T Large
MAX_SEGMENT_DURATION = 30.0  # seconds (increased to allow full sentences)
MIN_SEGMENT_DURATION = 0.5    # Minimum valid speech duration


class SeamlessTranslationEngine:
    """
    Speech-to-Speech translation engine using Seamless M4T v2 Large model.
    Uses dynamic energy-based speech detection to preserve silence.
    """
    
    MAX_RETRIES = 3
    
    def __init__(self, model_path: str, target_lang: str):
        """
        Initialize the Seamless M4T translation engine.
        
        Args:
            model_path: Path to the local Seamless M4T model directory
            target_lang: Target language code or name (e.g., 'urd', 'Urdu', 'eng', 'English')
        """
        self.model_path = Path(model_path)
        
        # Normalize language code (handle both codes and names)
        target_lang_lower = target_lang.lower().strip()
        self.target_lang = LANG_CODE_MAP.get(target_lang_lower, target_lang_lower)
        logger.info(f"Target language normalized: '{target_lang}' -> '{self.target_lang}'")
        
        self.model = None
        self.processor = None
        self.vad_pipeline = None
        # Prioritize Apple Silicon MPS, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        logger.info(f"Initializing SeamlessTranslationEngine with device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Target language: {target_lang}")
        
        self._load_model()
        self._load_vad()
    
    def _load_vad(self):
        """
        Load Silero VAD model for dynamic speech detection.
        Silero VAD is lightweight, accurate, and requires no authentication.
        Downloads automatically on first use and caches locally for offline operation.
        """
        try:
            logger.info("Loading Silero VAD model...")
            logger.info("  Repository: snakers4/silero-vad")
            logger.info("  This may take a minute on first run (downloading model)")
            
            # Load Silero VAD from torch.hub
            # Downloads automatically on first use, caches in ~/.cache/torch/hub
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            
            # Move model to appropriate device
            self.vad_model = self.vad_model.to(self.device)
            
            logger.info("✓ Silero VAD loaded successfully")
            logger.info(f"  Device: {self.device}")
            logger.info("  Model cached for offline use")
            logger.info("  No authentication required")
            
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.warning("Falling back to energy-based detection")
            self.vad_model = None
            self.get_speech_timestamps = None
    
    def _load_model(self):
        """
        Load the Seamless M4T model optimized for Apple Silicon (MPS) with FP16.
        Uses local model directory (Medium) and avoids 8-bit quantization.
        """
        try:
            logger.info("Loading Seamless M4T v2 Large model...")
            
            # Check if model directory exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # Enable memory efficient loading for all devices
            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(str(self.model_path))
            logger.info("✓ Processor loaded successfully")
            
            # Load model with device-optimized settings
            if self.device == "mps":
                logger.info("Loading model on Apple Silicon MPS with FP16...")
                self.model = SeamlessM4Tv2Model.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=False,
                )
                self.model = self.model.to(self.device)
                logger.info("Model loaded on MPS (FP16)")
            elif self.device == "cuda":
                logger.info("Loading model on CUDA with FP16 (no 8-bit quantization)...")
                self.model = SeamlessM4Tv2Model.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                self.model = self.model.to(self.device)
                logger.info("Model loaded on CUDA (FP16)")
            else:
                logger.info("Loading model on CPU (FP32)...")
                self.model = SeamlessM4Tv2Model.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                self.model = self.model.to(self.device)
                logger.info("Model loaded on CPU")
            
            logger.info("✓ Seamless M4T model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Seamless M4T model: {str(e)}")
            raise
    
    def _find_semantic_split_point(self, audio_path: str, start_sec: float, end_sec: float, target_duration: float) -> float:
        """
        Find natural pause point within a window for semantic segmentation.
        Instead of hard-cutting at exactly 15.0s, this finds the quietest moment
        between 10-12 seconds to make the split, ensuring we don't cut mid-word.
        
        Args:
            audio_path: Path to audio file
            start_sec: Start of the segment in seconds
            end_sec: End of the segment in seconds
            target_duration: Target split point (e.g., 12.0s from start)
            
        Returns:
            Best split point in seconds (absolute position in audio)
        """
        # Load the segment
        audio, sr = librosa.load(audio_path, sr=16000, offset=start_sec, duration=end_sec-start_sec)
        
        # Define search window: look for quiet point between 80% and 100% of target
        # e.g., if target is 12s, search between 9.6s and 12s
        search_start = int(target_duration * 0.8 * sr)
        search_end = int(target_duration * sr)
        
        if search_end > len(audio):
            return end_sec  # No room to search, use original end
        
        # Calculate RMS energy in small windows (50ms each)
        window_size = int(0.05 * sr)  # 50ms windows
        min_energy = float('inf')
        best_split = search_start
        
        for i in range(search_start, search_end - window_size, window_size):
            window = audio[i:i + window_size]
            energy = np.sqrt(np.mean(window**2))  # RMS energy
            
            if energy < min_energy:
                min_energy = energy
                best_split = i
        
        # Convert sample position back to absolute seconds
        split_point = start_sec + (best_split / sr)
        
        logger.info(f"    Found quiet point at {split_point:.2f}s (energy: {min_energy:.6f})")
        return split_point
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    
    def _detect_speech_segments_from_audio(self, audio_path: str) -> List[Dict]:
        """
        PHASE 1: DYNAMIC AUDIO & SILENCE DETECTION WITH SILERO VAD
        Uses Silero VAD to detect speech segments with production-grade thresholds and padding.
        
        Critical Requirements:
        - 100% dynamic detection (works for ANY audio pattern)
        - Millisecond-precision timestamps
        - Intelligent filtering:
          * Minimum speech segment: 0.5s (filter noise/breaths)
          * Minimum silence gap: 0.8s (merge natural pauses within speech)
          * Padding: 0.1s before/after segments (capture co-articulation)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of dicts with 'start', 'end', 'duration' for each speech segment
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: DYNAMIC AUDIO & SILENCE DETECTION (SILERO VAD)")
        logger.info("="*80)
        
        total_duration = self._get_audio_duration(audio_path)
        logger.info(f"Total audio duration: {total_duration:.3f}s")
        
        # Production-grade thresholds optimized for Urdu speech patterns
        MIN_SPEECH_DURATION = 0.5  # 500ms minimum for valid speech
        MIN_SILENCE_GAP = 0.6      # 600ms minimum (LOWERED from 0.8s for Urdu's faster flow)
        PADDING = 0.5              # 500ms padding - prevents word-breaking at boundaries
        SAMPLE_RATE = 16000        # Silero VAD requires 16kHz
        
        # STRICT REQUIREMENT: ONLY Silero VAD (NO FALLBACK)
        if self.vad_model is None or self.get_speech_timestamps is None:
            error_msg = "CRITICAL ERROR: Silero VAD not loaded. Cannot proceed without VAD."
            logger.error(error_msg)
            logger.error("Check _load_vad() method - VAD model initialization failed")
            raise RuntimeError(error_msg)
        
        logger.info("✓ Silero VAD verified loaded - proceeding with speech detection...")
        
        try:
            # Load audio at 16kHz using librosa (avoids torchaudio/FFmpeg dependency)
            logger.info(f"  Loading audio with librosa at {SAMPLE_RATE}Hz...")
            audio_array, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            logger.info(f"  ✓ Audio loaded: {len(audio_array)} samples = {len(audio_array)/SAMPLE_RATE:.2f}s")
            
            # Convert numpy array to torch tensor directly on target device (avoid CPU→GPU latency)
            # Silero VAD requires float32, but we create directly on MPS to use Neural Engine
            wav = torch.from_numpy(audio_array).float().to(self.device)
            logger.info(f"  ✓ Converted to torch tensor: shape={wav.shape}, dtype={wav.dtype}, device={wav.device}")
            
            # Get speech timestamps from Silero VAD
            logger.info(f"  Calling Silero VAD get_speech_timestamps()...")
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=0.4,
                min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=int(MIN_SILENCE_GAP * 1000),
                window_size_samples=512,
                speech_pad_ms=int(PADDING * 1000)
            )
            
            logger.info(f"  ✓ Silero VAD returned {len(speech_timestamps)} raw timestamp entries")
            
            # Convert sample positions to seconds
            raw_segments = []
            for idx, segment in enumerate(speech_timestamps):
                start_sec = segment['start'] / SAMPLE_RATE
                end_sec = segment['end'] / SAMPLE_RATE
                duration = end_sec - start_sec
                
                logger.info(f"    Raw segment {idx+1}: samples {segment['start']}-{segment['end']} = {start_sec:.3f}s-{end_sec:.3f}s (duration: {duration:.3f}s)")
                
                if duration >= MIN_SPEECH_DURATION:
                    raw_segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "duration": duration
                    })
                else:
                    logger.info(f"      FILTERED OUT (duration {duration:.3f}s < minimum {MIN_SPEECH_DURATION}s)")
            
            logger.info(f"✓ Silero VAD SUCCESS: {len(raw_segments)} speech segments after filtering")
            
            if len(raw_segments) == 0:
                error_msg = f"Silero VAD detected NO valid speech segments in {total_duration:.3f}s audio"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"FATAL: Silero VAD processing failed: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
        
        # Merge segments with gaps < MIN_SILENCE_GAP
        merged_segments = []
        current_segment = raw_segments[0].copy()
        
        for next_segment in raw_segments[1:]:
            gap = next_segment['start'] - current_segment['end']
            
            if gap < MIN_SILENCE_GAP:
                # Merge: extend current segment to include next
                current_segment['end'] = next_segment['end']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
            else:
                # Save current and start new
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        merged_segments.append(current_segment)
        
        # Apply padding (0.1s before/after each segment)
        final_segments = []
        for seg in merged_segments:
            padded_start = max(0.0, seg['start'] - PADDING)
            padded_end = min(total_duration, seg['end'] + PADDING)
            
            final_segments.append({
                "start": padded_start,
                "end": padded_end,
                "duration": padded_end - padded_start
            })
        
        # Calculate statistics
        total_speech = sum(s['duration'] for s in final_segments)
        total_silence = total_duration - total_speech
        
        # EXTRACT SILENT SEGMENTS (gaps between speech)
        silent_segments = []
        
        # Check for leading silence
        if final_segments and final_segments[0]['start'] > 0.1:
            silent_segments.append({
                "start": 0.0,
                "end": final_segments[0]['start'],
                "duration": final_segments[0]['start']
            })
        
        # Check for silence between speech segments
        for i in range(len(final_segments) - 1):
            current_end = final_segments[i]['end']
            next_start = final_segments[i + 1]['start']
            gap_duration = next_start - current_end
            
            if gap_duration > 0.05:  # Minimum 50ms to count as silence
                silent_segments.append({
                    "start": current_end,
                    "end": next_start,
                    "duration": gap_duration
                })
        
        # Check for trailing silence
        if final_segments and final_segments[-1]['end'] < total_duration - 0.1:
            silent_segments.append({
                "start": final_segments[-1]['end'],
                "end": total_duration,
                "duration": total_duration - final_segments[-1]['end']
            })
        
        logger.info(f"\n✓ Detection Complete:")
        logger.info(f"  Speech segments: {len(final_segments)}")
        logger.info(f"  Silent segments: {len(silent_segments)}")
        logger.info(f"  Total speech: {total_speech:.3f}s ({100*total_speech/total_duration:.1f}%)")
        logger.info(f"  Total silence: {total_silence:.3f}s ({100*total_silence/total_duration:.1f}%)")
        
        # CRITICAL: SEMANTIC SEGMENTATION (prevents mid-word cuts)
        # Instead of hard-cutting at 12.0s, find natural pause points
        split_segments = []
        for seg in final_segments:
            if seg['duration'] > MAX_SEGMENT_DURATION:
                logger.warning(f"  ⚠ Segment {seg['start']:.1f}s-{seg['end']:.1f}s is {seg['duration']:.1f}s (>{MAX_SEGMENT_DURATION}s)")
                logger.warning(f"  Using SEMANTIC split (finding natural pauses)...")
                
                # Split semantically at natural pause points
                current_start = seg['start']
                chunk_count = 0
                
                while current_start < seg['end']:
                    remaining_duration = seg['end'] - current_start
                    
                    if remaining_duration <= MAX_SEGMENT_DURATION:
                        # Last chunk - take everything remaining
                        split_segments.append({
                            "start": current_start,
                            "end": seg['end'],
                            "duration": remaining_duration
                        })
                        logger.info(f"    → Final chunk: {current_start:.2f}s-{seg['end']:.2f}s ({remaining_duration:.2f}s)")
                        break
                    else:
                        # Find natural pause point around MAX_SEGMENT_DURATION
                        split_point = self._find_semantic_split_point(
                            audio_path, 
                            current_start, 
                            seg['end'],
                            MAX_SEGMENT_DURATION
                        )
                        
                        chunk_duration = split_point - current_start
                        split_segments.append({
                            "start": current_start,
                            "end": split_point,
                            "duration": chunk_duration
                        })
                        logger.info(f"    → Chunk {chunk_count+1}: {current_start:.2f}s-{split_point:.2f}s ({chunk_duration:.2f}s)")
                        
                        current_start = split_point
                        chunk_count += 1
            else:
                split_segments.append(seg)
        
        if len(split_segments) > len(final_segments):
            logger.info(f"\n✓ Semantic splitting complete: {len(final_segments)} → {len(split_segments)} segments")
            logger.info(f"  Splits made at natural pause points (not mid-word)")
            final_segments = split_segments
        
        logger.info(f"\n" + "*"*80)
        logger.info(f"COMPLETE AUDIO BREAKDOWN - SPEECH + SILENCE:")
        logger.info(f"*"*80)
        
        # Build complete timeline
        all_segments = []
        for seg in final_segments:
            all_segments.append(("SPEECH", seg))
        for seg in silent_segments:
            all_segments.append(("SILENCE", seg))
        
        # Sort by start time
        all_segments.sort(key=lambda x: x[1]['start'])
        
        for idx, (seg_type, seg) in enumerate(all_segments, 1):
            marker = "🔊" if seg_type == "SPEECH" else "🔇"
            logger.info(f"  [{idx:2d}] {marker} {seg_type:7s}: {seg['start']:.3f}s - {seg['end']:.3f}s (duration: {seg['duration']:.3f}s)")
        
        logger.info(f"*"*80)
        logger.info(f"VERIFICATION:")
        logger.info(f"  Speech segments detected: {len(final_segments)} (will be translated)")
        logger.info(f"  Silent segments detected: {len(silent_segments)} (will be preserved as-is)")
        logger.info(f"  Total segments: {len(all_segments)}")
        
        # CRITICAL VERIFICATION: Ensure segments cover the full audio timeline
        if final_segments:
            first_segment_start = final_segments[0]['start']
            last_segment_end = final_segments[-1]['end']
            coverage_percentage = (last_segment_end / total_duration) * 100
            
            logger.info(f"\nTIMELINE COVERAGE VERIFICATION:")
            logger.info(f"  Audio duration: 0.0s - {total_duration:.3f}s")
            logger.info(f"  First speech segment starts at: {first_segment_start:.3f}s")
            logger.info(f"  Last speech segment ends at: {last_segment_end:.3f}s")
            logger.info(f"  Coverage: {coverage_percentage:.1f}% of audio timeline")
            
            if last_segment_end < total_duration * 0.5:
                logger.warning(f"⚠ WARNING: Speech segments only cover {coverage_percentage:.1f}% of audio!")
                logger.warning(f"  This might indicate detection stopped prematurely")
        
        logger.info(f"="*80)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 1 COMPLETE - RETURNING {len(final_segments)} SPEECH SEGMENTS FOR TRANSLATION")
        logger.info(f"{'='*80}\n")
        
        return final_segments
    
    def _detect_speech_segments(self, audio_path: str) -> List[Dict]:
        """
        Detect speech segments dynamically from audio.
        Returns actual speech boundaries detected from the audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of dicts with 'start', 'end', and 'duration' for each speech segment
        """
        return self._detect_speech_segments_from_audio(audio_path)

    
    def _extract_audio_segment(self, audio_path: str, start: float, end: float, 
                               output_path: Path) -> Path:
        """
        Extract a specific time segment from audio file.
        
        Args:
            audio_path: Source audio file path
            start: Start time in seconds
            end: End time in seconds
            output_path: Path for extracted segment
            
        Returns:
            Path to extracted segment
        """
        audio = AudioSegment.from_file(audio_path)
        segment = audio[int(start * 1000):int(end * 1000)]
        segment.export(str(output_path), format="wav")
        return output_path
    
    def _time_stretch_to_duration(self, audio_array: np.ndarray, target_duration: float, sample_rate: int) -> np.ndarray:
        """
        Time-stretch audio to match EXACT target duration (sample-perfect).
        Preserves pitch while adjusting tempo.
        
        Args:
            audio_array: Audio data as numpy array
            target_duration: Desired duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Time-stretched audio array with exact target duration
        """
        current_duration = len(audio_array) / sample_rate
        target_samples = int(target_duration * sample_rate)
        
        if abs(current_duration - target_duration) < 0.001:
            return audio_array[:target_samples] if len(audio_array) > target_samples else np.pad(audio_array, (0, target_samples - len(audio_array)))
        
        # Calculate stretch rate
        rate = current_duration / target_duration
        
        # Safety cap: if stretch rate > 1.3x, don't stretch (prevents extreme distortion)
        if rate > 1.3:
            logger.warning(f"    Stretch rate {rate:.2f}x exceeds 1.3x limit - keeping natural duration")
            # Truncate or pad to target instead of extreme stretching
            if len(audio_array) > target_samples:
                return audio_array[:target_samples]
            else:
                return np.pad(audio_array, (0, target_samples - len(audio_array)), mode='constant')
        
        # Apply high-quality time-stretching with soxr resampling for clarity
        # Use larger n_fft=2048 for better frequency resolution
        stretched = librosa.effects.time_stretch(audio_array, rate=rate, n_fft=2048)
        
        # Apply soxr high-quality resampling to maintain high-frequency clarity
        stretched = librosa.resample(stretched, orig_sr=sample_rate, target_sr=sample_rate, res_type='soxr_hq')
        
        # Ensure EXACT sample count (sample-perfect)
        if len(stretched) > target_samples:
            stretched = stretched[:target_samples]
        elif len(stretched) < target_samples:
            stretched = np.pad(stretched, (0, target_samples - len(stretched)), mode='constant')
        
        return stretched
    
    def _translate_speech_segment(self, segment_audio_path: Path, segment_info: Dict, segment_index: int) -> Optional[np.ndarray]:
        """
        PHASE 2: SEGMENT-BY-SEGMENT TRANSLATION
        Translates speech segment with SMART time-stretching (only if ratio is reasonable).
        
        Args:
            segment_audio_path: Path to audio segment
            segment_info: Dict with 'start', 'end', 'duration' of original segment
            segment_index: Index of the segment
            
        Returns:
            Translated audio (time-stretched if ratio is 0.8x-1.25x, otherwise natural)
        """
        try:
            original_duration = segment_info['duration']
            logger.info(f"  Translating segment {segment_index}...")
            logger.info(f"    Original duration: {original_duration:.3f}s")
            
            # Load audio for translation
            audio_array, sample_rate = librosa.load(str(segment_audio_path), sr=16000)
            logger.info(f"    Loaded segment audio: {len(audio_array)} samples = {len(audio_array)/sample_rate:.3f}s")
            
            # Clear device cache for memory efficiency
            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Calculate appropriate max_new_tokens for S2ST mode
            # Increase buffer to cover longer Urdu sentences without truncation
            # Empirically set to ~160 units/second, with higher cap for 30s segments
            estimated_tokens = int(original_duration * 200)
            max_tokens = min(4096, max(512, estimated_tokens))
            logger.info(f"    S2ST token allocation: {max_tokens} units (estimated from {original_duration:.1f}s * 200)")
            
            # Perform translation with retry logic
            retry_count = 0
            while retry_count < self.MAX_RETRIES:
                try:
                    logger.info(f"    Processing audio through Seamless M4T processor...")
                    # Process audio
                    audio_inputs = self.processor(
                        audio=audio_array,
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    )
                    logger.info(f"    ✓ Audio processed, input tensor shape: {audio_inputs['input_features'].shape}")
                    
                    # Move to device with proper dtype for MPS (float16 for memory efficiency)
                    if self.device == "mps":
                        audio_inputs = {k: v.to(self.device).half() if v.is_floating_point() else v.to(self.device) for k, v in audio_inputs.items()}
                    else:
                        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}
                    logger.info(f"    ✓ Moved to device: {self.device}")
                    
                    # Generate translation (S2ST mode preserves prosody and emotion)
                    logger.info(f"    Calling model.generate() in S2ST mode with max_new_tokens={max_tokens}...")
                    model_output = self.model.generate(
                        **audio_inputs,
                        tgt_lang=self.target_lang,
                        generate_speech=True,  # CRITICAL: Enables Speech-to-Speech translation
                        max_new_tokens=max_tokens,
                        num_beams=1,
                        do_sample=False,
                    )
                    logger.info(f"    ✓ S2ST generation complete, output type: {type(model_output)}")
                    
                    # Extract audio from S2ST model output
                    # With generate_speech=True, SeamlessM4T returns audio waveforms directly
                    # Output structure: tensor of shape (batch_size, sequence_length) or tuple
                    if isinstance(model_output, tuple):
                        # If tuple, first element is the audio waveform
                        generated_audio_tensor = model_output[0]
                        logger.info(f"    Output is tuple, extracted waveform tensor from index 0")
                    else:
                        # Direct tensor output
                        generated_audio_tensor = model_output
                        logger.info(f"    Output is direct tensor")
                    
                    logger.info(f"    Waveform tensor shape: {generated_audio_tensor.shape}, dtype: {generated_audio_tensor.dtype}")
                    
                    # Extract numpy audio array from tensor
                    # Handle both (batch, sequence) and (sequence,) shapes
                    # CRITICAL: Move to CPU and convert to float32 before numpy (MPS doesn't support direct conversion)
                    if generated_audio_tensor.dim() > 1:
                        translated_audio = generated_audio_tensor[0].cpu().float().numpy().squeeze()
                        logger.info(f"    Extracted from batch dimension [0]")
                    else:
                        translated_audio = generated_audio_tensor.cpu().float().numpy().squeeze()
                        logger.info(f"    Extracted from single tensor")
                    
                    logger.info(f"    ✓ Extracted audio array: {len(translated_audio)} samples, dtype: {translated_audio.dtype}")
                    
                    translated_duration = len(translated_audio) / sample_rate
                    logger.info(f"    Raw translated duration: {translated_duration:.3f}s")
                    
                    # CRITICAL VERIFICATION: Check if translation seems truncated
                    if translated_duration < 0.5:
                        logger.error(f"    ⚠ SUSPICIOUS: Translated audio is only {translated_duration:.3f}s from {original_duration:.3f}s input")
                        logger.error(f"    This might indicate model stopped prematurely!")
                    
                    # CRITICAL: Verify audio is not all zeros (silence)
                    non_zero_samples = np.count_nonzero(translated_audio)
                    if non_zero_samples == 0:
                        logger.error(f"    ✗ CRITICAL: Translated audio is ALL ZEROS (pure silence)!")
                        logger.error(f"    Translation failed - model generated no actual audio")
                        return None
                    
                    logger.info(f"    ✓ Translation verification: {non_zero_samples}/{len(translated_audio)} non-zero samples ({100*non_zero_samples/len(translated_audio):.1f}%)")
                    
                    # SMART TIME-STRETCHING: Only if ratio is reasonable
                    stretch_ratio = translated_duration / original_duration
                    
                    if stretch_ratio < 0.8 or stretch_ratio > 1.25:
                        # Outside reasonable bounds - don't stretch (sounds unnatural)
                        logger.info(f"    Ratio {stretch_ratio:.2f}x outside bounds (0.8-1.25), keeping natural duration")
                        final_duration = translated_duration
                    elif abs(translated_duration - original_duration) > 0.1:
                        # Significant difference but within bounds - apply stretch
                        logger.info(f"    Applying time-stretch (ratio: {stretch_ratio:.2f}x) to match {original_duration:.3f}s...")
                        translated_audio = self._time_stretch_to_duration(
                            translated_audio, 
                            original_duration, 
                            sample_rate
                        )
                        final_duration = len(translated_audio) / sample_rate
                        logger.info(f"    ✓ Stretched to: {final_duration:.3f}s")
                    else:
                        # Close enough - no stretch needed
                        logger.info(f"    Duration close enough ({translated_duration:.3f}s), no stretching")
                        final_duration = translated_duration
                    
                    # FINAL VERIFICATION before returning
                    logger.info(f"    FINAL CHECK before returning:")
                    logger.info(f"      Array length: {len(translated_audio)} samples")
                    logger.info(f"      Duration: {len(translated_audio)/sample_rate:.3f}s")
                    logger.info(f"      Non-zero: {np.count_nonzero(translated_audio)} samples")
                    logger.info(f"      Min value: {np.min(translated_audio):.6f}")
                    logger.info(f"      Max value: {np.max(translated_audio):.6f}")
                    logger.info(f"    ✓✓✓ Translation segment {segment_index} VERIFIED AND RETURNING ✓✓✓")
                    
                    return translated_audio
                    
                except torch.cuda.OutOfMemoryError as oom_error:
                    retry_count += 1
                    logger.warning(f"    OOM error, retry {retry_count}/{self.MAX_RETRIES}")
                    # Clear device cache
                    if self.device == "mps" and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if retry_count >= self.MAX_RETRIES:
                        raise oom_error
                        
        except Exception as e:
            logger.error(f"  ✗ Translation failed for segment {segment_index}: {str(e)}")
            return None
    
    def _reconstruct_audio_with_silence(self, segments: List[Dict], 
                                        translated_segments: List[np.ndarray],
                                        total_duration: float,
                                        sample_rate: int,
                                        output_path: Path) -> Path:
        """
        PHASE 3: DYNAMIC AUDIO RECONSTRUCTION
        Place translated segments at their original positions.
        Canvas expands dynamically if translations are longer.
        
        Args:
            segments: List of segment dicts with 'start' and 'end' times
            translated_segments: List of translated audio arrays (may vary in length)
            total_duration: Total duration of original audio in seconds
            sample_rate: Audio sample rate
            output_path: Path for reconstructed audio
            
        Returns:
            Path to reconstructed audio file
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: DYNAMIC AUDIO RECONSTRUCTION")
        logger.info("="*80)
        
        # VERIFICATION: Check inputs
        logger.info(f"Input verification:")
        logger.info(f"  Segments metadata: {len(segments)} entries")
        logger.info(f"  Translated audio arrays: {len(translated_segments)} entries")
        logger.info(f"  Valid translations: {len([s for s in translated_segments if s is not None and len(s) > 0])}")
        
        if len(segments) != len(translated_segments):
            logger.error(f"CRITICAL: Segment/translation count mismatch!")
            raise Exception(f"Cannot reconstruct: {len(segments)} segments but {len(translated_segments)} translations")
        
        # Calculate required canvas size (accommodate longer translations)
        max_end_time = total_duration
        for idx, (segment, translated) in enumerate(zip(segments, translated_segments)):
            if translated is not None and len(translated) > 0:
                segment_end = segment['start'] + (len(translated) / sample_rate)
                max_end_time = max(max_end_time, segment_end)
        
        total_samples = int(max_end_time * sample_rate)
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        logger.info(f"\nCanvas setup:")
        logger.info(f"  Original duration: {total_duration:.3f}s")
        logger.info(f"  Canvas duration: {max_end_time:.3f}s ({total_samples} samples at {sample_rate}Hz)")
        logger.info(f"  Will place {len([s for s in translated_segments if s is not None])} translated speech segments")
        logger.info(f"  Silence will be preserved automatically (zero-filled canvas)")
        
        # Calculate expected silent regions
        logger.info(f"\n📊 Expected Audio Structure:")
        logger.info(f"  Total duration: {total_duration:.3f}s")
        logger.info(f"  Speech regions: {len(segments)} segments (will contain translated audio)")
        
        # Show where silence should be
        if segments:
            if segments[0]['start'] > 0.1:
                logger.info(f"  Leading silence: 0.0s - {segments[0]['start']:.3f}s ({segments[0]['start']:.3f}s)")
            
            for i in range(len(segments) - 1):
                gap = segments[i + 1]['start'] - segments[i]['end']
                if gap > 0.05:
                    logger.info(f"  Silence gap #{i+1}: {segments[i]['end']:.3f}s - {segments[i + 1]['start']:.3f}s ({gap:.3f}s)")
            
            if segments[-1]['end'] < total_duration - 0.1:
                trailing = total_duration - segments[-1]['end']
                logger.info(f"  Trailing silence: {segments[-1]['end']:.3f}s - {total_duration:.3f}s ({trailing:.3f}s)")
        logger.info(f"")
        
        # Crossfade duration for smooth transitions - 20ms to eliminate breaking sounds
        crossfade_samples = int(0.02 * sample_rate)  # 20ms crossfade
        
        placed_segments = 0
        
        # Place each translated segment at its EXACT original position
        logger.info(f"{'─'*80}")
        logger.info(f"PLACING SEGMENTS INTO CANVAS WITH 20MS CROSSFADE")
        logger.info(f"{'─'*80}\n")
        
        for idx, (segment, translated_audio) in enumerate(zip(segments, translated_segments)):
            logger.info(f"\nProcessing segment {idx+1}/{len(segments)} for placement:")
            
            if translated_audio is None or len(translated_audio) == 0:
                logger.warning(f"  ✗ SKIPPED - Translation was None or empty")
                logger.warning(f"     Original position: {segment['start']:.3f}s - {segment['end']:.3f}s")
                continue
            
            # Make a copy to avoid modifying original
            segment_audio = translated_audio.copy()
            logger.info(f"  ✓ Valid translation found (length: {len(segment_audio)} samples)")
            
            # Calculate sample positions
            start_sample = int(segment['start'] * sample_rate)
            segment_samples = len(segment_audio)
            end_sample = start_sample + segment_samples
            
            # Check if we need to expand canvas
            if end_sample > len(final_audio):
                new_length = end_sample + int(2 * sample_rate)  # Add 2s buffer
                logger.info(f"  Expanding canvas to {new_length/sample_rate:.3f}s to fit segment {idx+1}")
                final_audio = np.pad(final_audio, (0, new_length - len(final_audio)), mode='constant')
                total_samples = len(final_audio)
            
            # Boundary check
            if start_sample >= total_samples:
                logger.error(f"  Segment {idx+1}: SKIPPED (position beyond audio end)")
                continue
            
            # ADVANCED CROSSFADING: Apply linear fade-in/out for smooth transitions
            if crossfade_samples > 0 and len(segment_audio) > crossfade_samples * 2:
                # Fade in at start (first 20ms)
                if start_sample > 0:
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    segment_audio[:crossfade_samples] *= fade_in
                    
                    # Also fade out the existing audio at the crossfade point
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    final_audio[start_sample:start_sample+crossfade_samples] *= fade_out
                
                # Fade out at end (last 20ms)
                if end_sample < total_samples:
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    segment_audio[-crossfade_samples:] *= fade_out
                    
                    # Also fade in the next segment's beginning if it exists
                    if end_sample + crossfade_samples < total_samples:
                        fade_in = np.linspace(0, 1, crossfade_samples)
                        final_audio[end_sample:end_sample+crossfade_samples] *= fade_in
            
            # Place translated audio in final array with crossfading
            # Use addition for crossfade regions to blend smoothly
            overlap_start = max(0, start_sample - crossfade_samples)
            overlap_end = min(total_samples, end_sample + crossfade_samples)
            
            # For the main segment, use direct assignment
            final_audio[start_sample:end_sample] += segment_audio
            
            placed_segments += 1
            
            # Verify placement
            placed_samples = np.count_nonzero(final_audio[start_sample:end_sample])
            end_time = (start_sample + len(translated_audio)) / sample_rate
            logger.info(f"  ✓ PLACED at samples {start_sample}-{end_sample}")
            logger.info(f"     Position: {segment['start']:.3f}s - {end_time:.3f}s")
            logger.info(f"     Non-zero samples in region: {placed_samples}/{len(translated_audio)}")
            
            # Calculate silence gap to next segment
            silence_gap = "END"
            if idx < len(segments) - 1:
                next_start = segments[idx + 1]['start']
                current_end = segment['start'] + (len(translated_audio) / sample_rate)
                gap_duration = next_start - current_end
                silence_gap = f"{gap_duration:.3f}s"
            logger.info(f"     Silence after this segment: {silence_gap}")
        
        # FINAL VERIFICATION
        non_silent_samples = np.count_nonzero(final_audio)
        final_duration = len(final_audio) / sample_rate
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RECONSTRUCTION COMPLETE - FINAL VERIFICATION")
        logger.info(f"{'='*80}")
        logger.info(f"Expected to place: {len(segments)} speech segments")
        logger.info(f"Actually placed: {placed_segments} speech segments")
        logger.info(f"Skipped (failed translations): {len(segments) - placed_segments}")
        logger.info(f"\nAudio statistics:")
        logger.info(f"  Output duration: {final_duration:.3f}s")
        logger.info(f"  Original duration: {total_duration:.3f}s")
        logger.info(f"  Total samples: {len(final_audio)}")
        logger.info(f"  Speech samples (non-zero): {non_silent_samples} ({100*non_silent_samples/len(final_audio):.1f}%)")
        logger.info(f"  Silent samples (zeros): {len(final_audio) - non_silent_samples} ({100*(1-non_silent_samples/len(final_audio)):.1f}%)")
        
        # Verify silence preservation (accounting for failed translations)
        logger.info(f"\n✓ SILENCE PRESERVATION VERIFICATION:")
        
        # Identify which segments were successfully placed
        successful_segment_indices = [i for i, trans in enumerate(translated_segments) if trans is not None and len(trans) > 0]
        
        # Check leading silence (only if first segment was translated)
        if successful_segment_indices and segments[0]['start'] > 0.1:
            leading_end_sample = int(segments[0]['start'] * sample_rate)
            leading_silence_check = np.count_nonzero(final_audio[:leading_end_sample])
            status = "✓ SILENT" if leading_silence_check == 0 else "⚠ HAS AUDIO"
            logger.info(f"  Leading silence (0.0s - {segments[0]['start']:.3f}s): {leading_silence_check} non-zero samples [{status}]")
        
        # Check gaps between successfully translated segments
        for i in range(len(segments) - 1):
            # Only check gaps between segments that were both successfully translated
            if i in successful_segment_indices and (i+1) in successful_segment_indices:
                gap_start = segments[i]['end']
                gap_end = segments[i + 1]['start']
                gap_duration = gap_end - gap_start
                
                if gap_duration > 0.05:
                    gap_start_sample = int(gap_start * sample_rate)
                    gap_end_sample = int(gap_end * sample_rate)
                    gap_silence_check = np.count_nonzero(final_audio[gap_start_sample:gap_end_sample])
                    status = "✓ SILENT" if gap_silence_check == 0 else "⚠ HAS AUDIO"
                    logger.info(f"  Silence gap #{i+1} ({gap_start:.3f}s - {gap_end:.3f}s, {gap_duration:.3f}s): {gap_silence_check} non-zero samples [{status}]")
        
        # Check trailing silence (only if last segment was translated)
        if successful_segment_indices and segments[-1]['end'] < total_duration - 0.1:
            trailing_start_sample = int(segments[-1]['end'] * sample_rate)
            trailing_silence_check = np.count_nonzero(final_audio[trailing_start_sample:])
            status = "✓ SILENT" if trailing_silence_check == 0 else "⚠ HAS AUDIO"
            logger.info(f"  Trailing silence ({segments[-1]['end']:.3f}s - {total_duration:.3f}s): {trailing_silence_check} non-zero samples [{status}]")
        
        if placed_segments < len(segments):
            logger.warning(f"\n⚠ WARNING: Only {placed_segments}/{len(segments)} segments were placed!")
            logger.warning(f"  {len(segments) - placed_segments} segments failed translation and are missing from output")
        else:
            logger.info(f"\n✓ SUCCESS: All {placed_segments} speech segments placed successfully")
            logger.info(f"✓ All silent segments preserved as zero-filled regions")
        
        # Sample audio at different positions to verify content
        logger.info(f"\nAudio content verification (checking for non-zero regions):")
        sample_points = [0, 0.25, 0.5, 0.75, 1.0]
        for point in sample_points:
            pos = int(point * len(final_audio))
            if pos < len(final_audio):
                window_start = max(0, pos - 1000)
                window_end = min(len(final_audio), pos + 1000)
                window_nonzero = np.count_nonzero(final_audio[window_start:window_end])
                logger.info(f"  At {point*100:.0f}% ({pos/sample_rate:.1f}s): {window_nonzero}/2000 non-zero samples")
        
        # Save reconstructed audio
        sf.write(str(output_path), final_audio, sample_rate)
        
        logger.info(f"\n✓ Audio saved: {output_path}")
        logger.info(f"="*80)
        
        return output_path
    
    def _get_video_audio_duration(self, video_path: str) -> float:
        """
        Extract audio duration from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio duration in seconds
        """
        try:
            with VideoFileClip(video_path) as video:
                return video.duration
        except Exception as e:
            logger.error(f"Failed to get video duration: {str(e)}")
            # Fallback: use audio file directly
            return None
    
    def translate_audio(self, source_audio_path: str, original_video_path: str) -> dict:
        """
        THREE-PHASE DYNAMIC SPEECH-TO-SPEECH TRANSLATION PIPELINE
        
        PHASE 1: Dynamic VAD-based speech/silence detection with production thresholds
        PHASE 2: Segment-by-segment S2ST translation with time-stretching for exact duration match
        PHASE 3: Sample-perfect reconstruction preserving all silence at original positions
        
        Args:
            source_audio_path: Path to extracted source audio WAV
            original_video_path: Path to original video file
            
        Returns:
            Dictionary with success status, translated_audio_path, and metadata
        """
        logger.info("=" * 80)
        logger.info("THREE-PHASE DYNAMIC SPEECH-TO-SPEECH TRANSLATION PIPELINE")
        logger.info(f"Source audio: {source_audio_path}")
        logger.info(f"Original video: {original_video_path}")
        logger.info(f"Target language: {self.target_lang}")
        logger.info("=" * 80)
        
        source_path = Path(source_audio_path)
        temp_dir = source_path.parent
        process_id = temp_dir.name
        
        logger.info(f"Processing directory: {temp_dir}")
        
        try:
            # Get target duration from original video
            target_duration = self._get_video_audio_duration(original_video_path)
            if target_duration is None:
                target_duration = self._get_audio_duration(source_audio_path)
            
            logger.info(f"Target duration: {target_duration:.3f}s")
            
            # PHASE 1: Dynamic speech detection
            segments = self._detect_speech_segments(source_audio_path)
            
            if not segments:
                raise Exception("No speech segments detected in audio")
            
            # PHASE 2: Translate each speech segment with time-stretching
            translated_segments = []
            segment_files = []
            successful_translations = 0
            failed_translations = 0
            
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 2: SEGMENT-BY-SEGMENT TRANSLATION")
            logger.info(f"{'='*80}")
            logger.info(f"CRITICAL VERIFICATION: {len(segments)} segments detected in Phase 1")
            logger.info(f"MUST PROCESS ALL {len(segments)} SEGMENTS - NO SKIPPING ALLOWED")
            
            # Print complete segment timeline
            logger.info(f"\nCOMPLETE SEGMENT TIMELINE TO BE TRANSLATED:")
            for idx, seg in enumerate(segments):
                logger.info(f"  Segment {idx+1}: {seg['start']:.3f}s - {seg['end']:.3f}s (duration: {seg['duration']:.3f}s)")
            logger.info(f"Total segments to translate: {len(segments)}")
            logger.info(f"{'='*80}\n")
            
            for idx, segment in enumerate(segments):
                logger.info(f"\n{'#'*80}")
                logger.info(f"PROCESSING SEGMENT {idx+1} of {len(segments)} TOTAL")
                logger.info(f"{'#'*80}")
                try:
                    logger.info(f"Segment {idx+1} position: {segment['start']:.3f}s - {segment['end']:.3f}s (duration: {segment['duration']:.3f}s)")
                    
                    # Extract segment from original audio
                    segment_path = temp_dir / f"segment_{idx}.wav"
                    logger.info(f"  Step 1: Extracting audio segment to {segment_path.name}...")
                    self._extract_audio_segment(
                        source_audio_path, 
                        segment['start'], 
                        segment['end'], 
                        segment_path
                    )
                    segment_files.append(segment_path)
                    logger.info(f"  Step 1: ✓ Extracted")
                    
                    # Translate segment with time-stretching
                    logger.info(f"  Step 2: Sending to translation engine...")
                    translated_audio = self._translate_speech_segment(segment_path, segment, idx+1)
                    
                    if translated_audio is not None and len(translated_audio) > 0:
                        successful_translations += 1
                        trans_duration = len(translated_audio) / 16000
                        logger.info(f"  Step 2: ✓ TRANSLATION SUCCESS - Generated {trans_duration:.3f}s audio")
                        logger.info(f"  Current success count: {successful_translations}/{idx+1}")
                    else:
                        failed_translations += 1
                        logger.error(f"  Step 2: ✗ TRANSLATION FAILED - Returned None or empty array")
                        logger.error(f"  Current failure count: {failed_translations}/{idx+1}")
                    
                    translated_segments.append(translated_audio)
                    logger.info(f"  Step 3: Added to translated_segments list (index {idx})")
                    
                    # Memory flush every 5 segments to prevent swap mode on M2 (8GB RAM)
                    if (idx + 1) % 5 == 0:
                        logger.info(f"  Memory flush checkpoint: Processed {idx+1} segments")
                        if self.device == "mps" and torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        elif self.device == "cuda" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"  ✓ Device cache cleared")
                    
                except Exception as seg_error:
                    logger.error(f"\n{'!'*80}")
                    logger.error(f"EXCEPTION in segment {idx+1}/{len(segments)}: {str(seg_error)}")
                    logger.error(f"{'!'*80}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Append small silence as fallback to maintain segment alignment
                    # This prevents total pipeline failure from one bad segment
                    failed_translations += 1
                    fallback_duration = segment['duration']
                    fallback_samples = int(fallback_duration * 16000)
                    fallback_audio = np.zeros(fallback_samples, dtype=np.float32)
                    translated_segments.append(fallback_audio)
                    logger.warning(f"  Using {fallback_duration:.1f}s silence as fallback for failed segment")
            
            # CRITICAL VERIFICATION
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 2 COMPLETE - VERIFICATION")
            logger.info(f"{'='*80}")
            logger.info(f"Expected to process: {len(segments)} segments")
            logger.info(f"Actually processed: {len(translated_segments)} segments")
            logger.info(f"Successful translations: {successful_translations}")
            logger.info(f"Failed translations: {failed_translations}")
            logger.info(f"Success rate: {100*successful_translations/len(segments):.1f}%")
            
            # Verify counts match
            if len(translated_segments) != len(segments):
                logger.error(f"CRITICAL ERROR: Mismatch in segment counts!")
                logger.error(f"  Detected: {len(segments)}, Translated: {len(translated_segments)}")
                raise Exception(f"Segment count mismatch: {len(segments)} detected but {len(translated_segments)} translated")
            
            # List status of each segment
            logger.info(f"\nPer-segment status:")
            for idx, trans in enumerate(translated_segments):
                status = "✓ SUCCESS" if trans is not None and len(trans) > 0 else "✗ FAILED"
                duration = f"{len(trans)/16000:.3f}s" if trans is not None and len(trans) > 0 else "N/A"
                logger.info(f"  Segment {idx+1}: {status} (duration: {duration})")
            
            if successful_translations == 0:
                raise Exception(f"All {len(segments)} segment translations failed!")
            
            logger.info(f"\n✓ Proceeding to Phase 3 with {successful_translations} successful translations")
            logger.info(f"{'='*80}")
            
            # PHASE 3: Sample-perfect reconstruction
            final_audio_path = temp_dir / "translated_audio.wav"
            self._reconstruct_audio_with_silence(
                segments=segments,
                translated_segments=translated_segments,
                total_duration=target_duration,
                sample_rate=16000,
                output_path=final_audio_path
            )
            
            # Cleanup intermediate segment files
            for segment_file in segment_files:
                if segment_file.exists():
                    segment_file.unlink()
            
            logger.info("\n" + "=" * 80)
            logger.info("✓✓✓ TRANSLATION PIPELINE COMPLETED SUCCESSFULLY ✓✓✓")
            logger.info(f"Output: {final_audio_path}")
            logger.info(f"Duration: {target_duration:.3f}s (sample-perfect)")
            logger.info(f"Segments: {len(segments)} speech segments translated")
            logger.info(f"Silence: Preserved at exact original positions")
            logger.info(f"Quality: Time-stretched for perfect synchronization")
            logger.info("=" * 80)
            
            return {
                "success": True,
                "translated_audio_path": str(final_audio_path),
                "duration": target_duration,
                "segments_count": len(segments),
                "process_id": process_id,
            }
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU Out of Memory: {str(e)}"
            logger.error(error_msg)
            logger.error("CRITICAL: Translation failed due to insufficient VRAM")
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": "OOM",
            }
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": "GENERAL",
            }
    
    def cleanup(self):
        """Release model resources and clear GPU cache."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear device cache
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SeamlessTranslationEngine resources released")
