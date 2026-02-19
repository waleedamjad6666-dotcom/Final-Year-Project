"""
Lip Sync URL Configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import LipSyncJobViewSet

app_name = 'lip_sync'

router = DefaultRouter()
router.register(r'', LipSyncJobViewSet, basename='lip-sync-job')

urlpatterns = [
    path('', include(router.urls)),
]
