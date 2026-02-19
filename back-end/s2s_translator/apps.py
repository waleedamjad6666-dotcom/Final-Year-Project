from django.apps import AppConfig


class S2STranslatorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 's2s_translator'
    verbose_name = 'Speech-to-Speech Translation'
    
    def ready(self):
        """
        App initialization hook.
        Called when Django starts.
        """
        # Import any signals or startup code here if needed
        pass
