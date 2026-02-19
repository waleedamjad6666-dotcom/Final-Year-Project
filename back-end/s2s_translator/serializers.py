"""
Serializers for S2S Translation API
"""

from rest_framework import serializers
from videos.models import Video


class TranslationRequestSerializer(serializers.Serializer):
    """Serializer for translation request"""
    target_language = serializers.ChoiceField(
        choices=[('urd', 'Urdu'), ('eng', 'English')],
        required=False,
        help_text="Target language code (defaults to video's target_language)"
    )
    

class TranslationStatusSerializer(serializers.ModelSerializer):
    """Serializer for translation status response"""
    file_size_mb = serializers.ReadOnlyField()
    is_processing = serializers.ReadOnlyField()
    is_completed = serializers.ReadOnlyField()
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'status', 'progress', 'error_message',
            'source_language', 'target_language', 'duration',
            'file_size_mb', 'is_processing', 'is_completed',
            'created_at', 'updated_at', 'processed_at'
        ]
        read_only_fields = fields
