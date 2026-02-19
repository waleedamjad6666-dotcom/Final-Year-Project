"""
Test script to check what URL the video serializer returns for translated audio
"""
import os
import sys
import django

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from videos.models import Video
from videos.serializers import VideoSerializer

print("=" * 80)
print("TESTING AUDIO URL SERIALIZATION")
print("=" * 80)

# Get the latest video
try:
    video = Video.objects.latest('id')
    print(f"\nLatest Video:")
    print(f"  ID: {video.id}")
    print(f"  Title: {video.title}")
    print(f"  Status: {video.status}")
    print(f"  Progress: {video.progress}%")
    print(f"  translated_audio_path: {video.translated_audio_path}")
    
    # Serialize it
    serializer = VideoSerializer(video)
    data = serializer.data
    
    print(f"\nSerialized Data:")
    print(f"  translated_audio_url: {data.get('translated_audio_url')}")
    print(f"  cloned_audio_url: {data.get('cloned_audio_url')}")
    
    # Check if file exists
    if video.translated_audio_path:
        from pathlib import Path
        file_path = Path(video.translated_audio_path)
        print(f"\nFile Check:")
        print(f"  Path: {file_path}")
        print(f"  Exists: {file_path.exists()}")
        if file_path.exists():
            print(f"  Size: {file_path.stat().st_size / 1024:.2f} KB")
            print(f"  Name: {file_path.name}")
        
        # Check the directory for all audio files
        if file_path.parent.exists():
            print(f"\n  Files in directory:")
            for f in file_path.parent.iterdir():
                if f.suffix in ['.wav', '.mp3']:
                    print(f"    - {f.name} ({f.stat().st_size / 1024:.2f} KB)")
    
    print(f"\n✅ Serializer returns: {data.get('translated_audio_url')}")
    print(f"   This is what the frontend download button will use!")
    
except Video.DoesNotExist:
    print("\n❌ No videos found in database")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
