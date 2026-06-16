from .LLMEnums import LLMEnums
from .providers.GroqProvider import GroqProvider

class LLMProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str):
        # إذا طلب النظام موديل Groq، المصنع سيقوم بتجهيزه فوراً
        if provider == LLMEnums.GROQ.value:
            return GroqProvider(
                api_key=self.config.GROQ_API_KEY,
                default_generation_temperature=self.config.LLM_TEMPERATURE
            )

        return None
    
    