import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from videos.models import Video
from videos.serializers import VideoSerializer

v = Video.objects.filter(status='completed').order_by('-updated_at').first()

if v:
    s = VideoSerializer(v)
    print('Video ID:', v.id)
    print('processed_video field:', v.processed_video)
    print('processed_video.name:', v.processed_video.name if v.processed_video else None)
    print('processed_video_url from serializer:', s.data.get('processed_video_url'))
    print('Status:', v.status)
    print('Progress:', v.progress)
else:
    print('No completed videos found')
