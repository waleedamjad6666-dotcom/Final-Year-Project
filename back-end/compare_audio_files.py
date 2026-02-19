"""
Compare the translated and cloned audio files to see if they're different
"""
import sys
import os
import django

# Set UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from videos.models import Video
from pathlib import Path
import numpy as np
import librosa

print("=" * 80)
print("COMPARING TRANSLATED vs CLONED AUDIO")
print("=" * 80)

video = Video.objects.latest('id')
print(f"\nVideo: {video.id}")
print(f"translated_audio_path: {video.translated_audio_path}")

# Find both files
if video.translated_audio_path:
    cloned_path = Path(video.translated_audio_path)
    translated_path = cloned_path.parent / "translated_audio.wav"
    
    print(f"\nFiles:")
    print(f"  Translated: {translated_path}")
    print(f"  Exists: {translated_path.exists()}")
    if translated_path.exists():
        print(f"  Size: {translated_path.stat().st_size / 1024:.2f} KB")
    
    print(f"\n  Cloned: {cloned_path}")
    print(f"  Exists: {cloned_path.exists()}")
    if cloned_path.exists():
        print(f"  Size: {cloned_path.stat().st_size / 1024:.2f} KB")
    
    # Load and compare audio
    if translated_path.exists() and cloned_path.exists():
        print(f"\nLoading audio files...")
        trans_audio, trans_sr = librosa.load(str(translated_path), sr=None)
        clone_audio, clone_sr = librosa.load(str(cloned_path), sr=None)
        
        print(f"\nAudio Analysis:")
        print(f"  Translated:")
        print(f"    Sample rate: {trans_sr} Hz")
        print(f"    Duration: {len(trans_audio)/trans_sr:.2f}s")
        print(f"    RMS energy: {np.sqrt(np.mean(trans_audio**2)):.4f}")
        print(f"    Mean: {np.mean(trans_audio):.6f}")
        
        print(f"\n  Cloned:")
        print(f"    Sample rate: {clone_sr} Hz")
        print(f"    Duration: {len(clone_audio)/clone_sr:.2f}s")
        print(f"    RMS energy: {np.sqrt(np.mean(clone_audio**2)):.4f}")
        print(f"    Mean: {np.mean(clone_audio):.6f}")
        
        # Check if they're identical
        if len(trans_audio) == len(clone_audio):
            diff = np.abs(trans_audio - clone_audio)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"\nDifference Analysis:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff < 0.0001:
                print(f"\n❌ FILES ARE IDENTICAL!")
                print(f"   Voice cloning did NOT modify the audio!")
            else:
                print(f"\n✅ FILES ARE DIFFERENT!")
                print(f"   Voice cloning successfully modified the audio!")
        else:
            print(f"\n✅ Files have different lengths - cloning worked!")
