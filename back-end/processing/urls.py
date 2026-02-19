"""
URLs for video processing module.
"""
from django.urls import path
from . import views

app_name = 'processing'

urlpatterns = [
    path('extract-audio/', views.extract_audio, name='extract-audio'),
    path('get-metadata/', views.get_video_metadata, name='get-metadata'),
    path('extract-thumbnail/', views.extract_thumbnail, name='extract-thumbnail'),
]
