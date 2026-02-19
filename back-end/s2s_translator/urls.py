"""
URL configuration for S2S Translation API
"""

from django.urls import path
from . import views

app_name = 's2s_translator'

urlpatterns = [
    path('api/videos/<uuid:video_id>/translate/', views.start_s2s_translation, name='start_translation'),
    path('api/videos/<uuid:video_id>/translation-status/', views.get_translation_status, name='translation_status'),
]
