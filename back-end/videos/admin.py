from django.contrib import admin
from .models import Video, ProcessingLog


class ProcessingLogInline(admin.TabularInline):
    model = ProcessingLog
    extra = 0
    readonly_fields = ['step', 'level', 'message', 'timestamp']
    can_delete = False


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'status', 'progress', 'file_size_mb', 'duration', 'created_at']
    list_filter = ['status', 'created_at', 'source_language', 'target_language']
    search_fields = ['title', 'user__email', 'user__username']
    readonly_fields = ['id', 'file_size', 'duration', 'created_at', 'updated_at', 'processed_at']
    ordering = ['-created_at']
    inlines = [ProcessingLogInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'title', 'description')
        }),
        ('Files', {
            'fields': ('original_video', 'thumbnail', 'processed_video')
        }),
        ('Metadata', {
            'fields': ('duration', 'file_size', 'resolution')
        }),
        ('Processing', {
            'fields': ('status', 'progress', 'error_message', 'source_language', 'target_language')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'processed_at')
        }),
    )


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = ['video', 'step', 'level', 'timestamp']
    list_filter = ['level', 'timestamp']
    search_fields = ['video__title', 'step', 'message']
    readonly_fields = ['video', 'step', 'level', 'message', 'timestamp']
    ordering = ['-timestamp']
