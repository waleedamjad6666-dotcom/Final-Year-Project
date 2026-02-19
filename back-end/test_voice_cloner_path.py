"""
Quick test to verify VoiceCloner can find the model files
"""
import sys
from pathlib import Path

# Add OpenVoice to path
project_root = Path(__file__).parent.parent
openvoice_path = project_root / "extras" / "OpenVoice"
if openvoice_path.exists():
    sys.path.insert(0, str(openvoice_path))

print("Testing VoiceCloner Model Path...")
print("=" * 60)

# Check model path
model_path = project_root / "models" / "OpenVoiceV2"
print(f"Model path: {model_path}")
print(f"Exists: {model_path.exists()}")

if model_path.exists():
    converter_path = model_path / "converter"
    print(f"\nConverter path: {converter_path}")
    print(f"Exists: {converter_path.exists()}")
    
    if converter_path.exists():
        config_path = converter_path / "config.json"
        checkpoint_path = converter_path / "checkpoint.pth"
        print(f"\nConfig: {config_path.exists()} - {config_path}")
        print(f"Checkpoint: {checkpoint_path.exists()} - {checkpoint_path}")
        
        if config_path.exists() and checkpoint_path.exists():
            print("\n✅ All required files found!")
            print("\nNow testing VoiceCloner initialization...")
            
            try:
                from s2s_translator.voice_cloner import VoiceCloner
                print("✅ VoiceCloner imported successfully")
                
                # Try to initialize (may take a moment to load model)
                cloner = VoiceCloner()
                print("✅ VoiceCloner initialized successfully!")
                print(f"   Device: {cloner.device}")
                print(f"   Model path: {cloner.model_path}")
                
            except Exception as e:
                print(f"❌ VoiceCloner initialization failed: {e}")
        else:
            print("\n❌ Missing config.json or checkpoint.pth")
    else:
        print("\n❌ Converter directory not found")
else:
    print("\n❌ Model directory not found")
