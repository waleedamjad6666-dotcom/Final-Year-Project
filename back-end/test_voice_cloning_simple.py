"""
Simple test to verify voice cloning is working
"""
import sys
from pathlib import Path

# Add OpenVoice to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extras" / "OpenVoice"))

print("Testing Voice Cloning Setup...")
print("="*60)

# Test 1: OpenVoice import
try:
    from openvoice.api import ToneColorConverter
    print("✅ OpenVoice imported successfully")
except Exception as e:
    print(f"❌ OpenVoice import failed: {e}")
    sys.exit(1)

# Test 2: VoiceCloner import
try:
    from s2s_translator.voice_cloner import VoiceCloner
    print("✅ VoiceCloner imported successfully")
except Exception as e:
    print(f"❌ VoiceCloner import failed: {e}")
    sys.exit(1)

# Test 3: VoiceCloner initialization
try:
    cloner = VoiceCloner()
    print("✅ VoiceCloner initialized successfully")
    print(f"   Device: {cloner.device}")
    cloner.cleanup()
except Exception as e:
    print(f"❌ VoiceCloner initialization failed: {e}")
    sys.exit(1)

print("="*60)
print("🎉 ALL TESTS PASSED!")
print("\nVoice cloning is ready to use!")
print("\nNext steps:")
print("  1. python manage.py runserver")
print("  2. Upload a video")
print("  3. Download cloned audio after processing")
