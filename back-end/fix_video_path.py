import os
import django
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from videos.models import Video

# Get the most recent completed video
video = Video.objects.filter(status='completed').order_by('-updated_at').first()

if video:
    print(f'Video ID: {video.id}')
    print(f'Current processed_video: {video.processed_video}')
    
    # Check if final_video.mp4 exists in processing directory
    final_video_path = Path('media/processing') / str(video.id) / 'final_video.mp4'
    print(f'File exists at {final_video_path}: {final_video_path.exists()}')
    
    if final_video_path.exists():
        # Update the database with correct path
        video.processed_video.name = f'processing/{video.id}/final_video.mp4'
        video.save()
        print(f'✓ Updated processed_video to: {video.processed_video.name}')
        print('Now refresh the frontend page!')
    else:
        print('✗ Video file not found')
else:
    print('No completed videos found')
