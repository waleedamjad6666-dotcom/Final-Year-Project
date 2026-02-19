"""
Quick installation script for OpenVoice v2
Run this to set up voice cloning capabilities
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description, use_shell=False):
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run (string if use_shell=True, list otherwise)
        description: Description of what the command does
        use_shell: Whether to use shell=True (default False for better security)
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    # Display command nicely
    if isinstance(cmd, list):
        print(f"Running: {' '.join(cmd)}\n")
    else:
        print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=use_shell, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        if result.stderr:
            print(f"Error details: {result.stderr}")
        return False
    else:
        print(f"\n✅ {description} completed successfully!")
        return True


def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          OpenVoice v2 Installation Script                 ║
    ║        Voice Cloning for Translation Pipeline             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    project_root = Path(__file__).parent.parent
    openvoice_dir = project_root / "extras" / "OpenVoice"
    checkpoints_dir = openvoice_dir / "checkpoints_v2"
    
    print(f"Project root: {project_root}")
    print(f"OpenVoice directory: {openvoice_dir}")
    
    # Step 1: Install OpenVoice package
    print("\n📦 Step 1: Installing OpenVoice v2 package...")
    if not run_command(
        ["pip", "install", "git+https://github.com/myshell-ai/OpenVoice.git"],
        "OpenVoice v2 installation"
    ):
        print("\n⚠️  Trying alternative installation method...")
        if not run_command(
            ["pip", "install", "openvoice"],
            "OpenVoice v2 installation (PyPI)"
        ):
            print("\n❌ Failed to install OpenVoice!")
            print("Please install manually:")
            print("  pip install openvoice")
            print("  OR")
            print("  pip install git+https://github.com/myshell-ai/OpenVoice.git")
            return False
    
    # Step 2: Check for checkpoints
    print(f"\n📥 Step 2: Checking for OpenVoice v2 checkpoints...")
    
    if checkpoints_dir.exists() and (checkpoints_dir / "converter" / "checkpoint.pth").exists():
        print(f"✅ Checkpoints already exist at: {checkpoints_dir}")
    else:
        print(f"❌ Checkpoints not found at: {checkpoints_dir}")
        print("\n📥 Downloading checkpoints...")
        
        # Create directory
        openvoice_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using wget (Windows compatible)
        checkpoint_url = "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
        
        print(f"\nDownloading from: {checkpoint_url}")
        print(f"To: {openvoice_dir}")
        
        # Try different download methods
        download_success = False
        
        # Method 1: PowerShell (Windows)
        if sys.platform == "win32":
            ps_cmd = f'Invoke-WebRequest -Uri "{checkpoint_url}" -OutFile "{openvoice_dir / "checkpoints_v2_0417.zip"}"'
            if run_command(f'powershell -Command "{ps_cmd}"', "Downloading checkpoints (PowerShell)", use_shell=True):
                download_success = True
        
        # Method 2: curl (cross-platform - Mac/Linux)
        if not download_success:
            if run_command(
                ["curl", "-L", checkpoint_url, "-o", str(openvoice_dir / "checkpoints_v2_0417.zip")],
                "Downloading checkpoints (curl)"
            ):
                download_success = True
        
        if not download_success:
            print("\n❌ Automatic download failed!")
            print(f"\n📋 Manual download instructions:")
            print(f"1. Download: {checkpoint_url}")
            print(f"2. Save to: {openvoice_dir}")
            print(f"3. Extract the ZIP file")
            print(f"4. Ensure this structure exists:")
            print(f"   {checkpoints_dir}/")
            print(f"     ├─ converter/")
            print(f"     │   ├─ config.json")
            print(f"     │   └─ checkpoint.pth")
            print(f"     └─ base_speakers/")
            print(f"         └─ ses/")
            return False
        
        # Extract ZIP
        print("\n📦 Extracting checkpoints...")
        zip_path = openvoice_dir / "checkpoints_v2_0417.zip"
        
        if sys.platform == "win32":
            # Windows: Use PowerShell Expand-Archive
            ps_cmd = f'Expand-Archive -Path "{zip_path}" -DestinationPath "{openvoice_dir}" -Force'
            if not run_command(f'powershell -Command "{ps_cmd}"', "Extracting checkpoints", use_shell=True):
                print("❌ Extraction failed!")
                return False
        else:
            # Unix (Mac/Linux): Use unzip
            if not run_command(["unzip", str(zip_path), "-d", str(openvoice_dir)], "Extracting checkpoints"):
                print("❌ Extraction failed!")
                return False
        
        # Cleanup ZIP
        if zip_path.exists():
            zip_path.unlink()
            print("🗑️  Cleaned up ZIP file")
    
    # Step 3: Verify installation
    print("\n🔍 Step 3: Verifying installation...")
    
    try:
        # Test import
        print("Testing OpenVoice import...")
        import openvoice
        print("✅ OpenVoice package imported successfully")
        
        # Test voice cloner
        print("\nTesting VoiceCloner import...")
        sys.path.insert(0, str(project_root / "back-end"))
        from s2s_translator.voice_cloner import VoiceCloner
        print("✅ VoiceCloner imported successfully")
        
        # Test initialization
        print("\nTesting VoiceCloner initialization...")
        cloner = VoiceCloner(model_path=str(checkpoints_dir))
        cloner.cleanup()
        print("✅ VoiceCloner initialized successfully")
        
        print("\n" + "="*60)
        print("🎉 INSTALLATION COMPLETE!")
        print("="*60)
        print("\n✅ OpenVoice v2 is ready to use!")
        print(f"✅ Checkpoints location: {checkpoints_dir}")
        print("\n📖 Usage:")
        print("   from s2s_translator.voice_cloner import apply_voice_cloning")
        print("   cloned_audio = apply_voice_cloning(...)")
        print("\n📚 See VOICE_CLONING_SETUP.md for detailed documentation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if OpenVoice is installed:")
        print("   pip list | grep openvoice")
        print("2. Verify checkpoints exist:")
        print(f"   {checkpoints_dir}/converter/checkpoint.pth")
        print("3. Check Python path:")
        print(f"   sys.path should include: {project_root / 'back-end'}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
