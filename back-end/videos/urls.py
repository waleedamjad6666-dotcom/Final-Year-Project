from django.urls import path
from .views import (
    VideoListCreateView, VideoDetailView,
    VideoDownloadView, VideoStatusView,
    VideoLogsView, UserVideosStatsView
)

app_name = 'videos'

urlpatterns = [
    # Video CRUD
    path('', VideoListCreateView.as_view(), name='video-list-create'),
    path('<uuid:pk>/', VideoDetailView.as_view(), name='video-detail'),
    path('<uuid:pk>/download/', VideoDownloadView.as_view(), name='video-download'),
    path('<uuid:pk>/status/', VideoStatusView.as_view(), name='video-status'),
    path('<uuid:pk>/logs/', VideoLogsView.as_view(), name='video-logs'),
    
    # Statistics
    path('stats/user/', UserVideosStatsView.as_view(), name='user-stats'),
]
