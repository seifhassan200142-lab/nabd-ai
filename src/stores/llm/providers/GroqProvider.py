from ..LLMInterface import LLMInterface
from langchain_groq import ChatGroq

class GroqProvider(LLMInterface):
    def __init__(self, api_key: str, default_generation_temperature: float = 0.0):
        self.api_key = api_key
        self.default_temperature = default_generation_temperature
        self.llm = None

    # 1. إعداد وتجهيز الموديل
    def set_generation_model(self, model_id: str):
        self.llm = ChatGroq(
            temperature=self.default_temperature,
            groq_api_key=self.api_key,
            model_name=model_id
        )

    # 2. توليد الإجابة
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None, temperature: float = None):
        if not self.llm:
            raise ValueError("Groq model is not set. Please call set_generation_model() first.")
        
        # إرسال النص إلى الموديل واستلام الإجابة
        response = self.llm.invoke(prompt)
        return response.content

    # ----------------------------------------------------
    # دوال مطلوبة في كراسة الشروط (Interface) ولكننا لا نحتاجها
    # مع Groq لأن Groq متخصص في التوليد (Generation) وليس الـ Embeddings
    # لذا نضع فيها pass لكي لا يعترض بايثون
    # ----------------------------------------------------
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass

    def embed_text(self, text: str, document_type: str = None):
        pass

    def construct_prompt(self, prompt: str, role: str):
        pass

    