import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import Config

logger = logging.getLogger(__name__)

class ChatController:
    def __init__(self):
        # 1. Initialize the LLM Engine (all settings now read from config.py)
        if Config.CURRENT_LLM == "GROQ":
            self.llm = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model_name=Config.LLM_MODEL_GROQ,
                temperature=Config.LLM_TEMPERATURE
            )
        else:
            raise ValueError(f"Unsupported LLM engine: {Config.CURRENT_LLM}")

        # 2. In-Memory Store for Chat History
        # Format: {"session_id": [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]}
        self.chat_history_store = {}

    def get_session_history(self, session_id: str):
        """Retrieve history for a specific session, or create a new one."""
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = []
        return self.chat_history_store[session_id]

    def format_history_for_prompt(self, history_list):
        """Convert the history list into a single string to feed the LLM."""
        formatted = ""
        # Get only the last 4 interactions to save tokens and money!
        for msg in history_list[-4:]:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        return formatted

    def generate_answer(self, query: str, retrieved_chunks: list, session_id: str = "default_session"):
        """Generates an answer using Vector Context and Chat History."""

        # 1. Format the medical context from the Vector DB
        context_text = "\n".join([chunk.page_content for chunk in retrieved_chunks])

        # 2. Retrieve and format previous chat history for this user
        session_history = self.get_session_history(session_id)
        history_text = self.format_history_for_prompt(session_history)

        # 3. Build the Master Prompt (Context + History + New Question)
        master_prompt = ChatPromptTemplate.from_template("""
        You are an expert medical AI assistant. Answer the user's question accurately based ONLY on the provided medical context.
        If the answer is not in the context, say "I don't have enough medical data to answer this."

        PREVIOUS CONVERSATION HISTORY:
        {history}

        MEDICAL CONTEXT:
        {context}

        USER QUESTION: {question}
        EXPERT MEDICAL ANSWER (in Arabic):
        """)

        # 4. Create the final prompt with our variables
        formatted_prompt = master_prompt.format_messages(
            history=history_text,
            context=context_text,
            question=query
        )

        # 5. Generate Response
        logger.info(f"Generating answer for session: {session_id}")
        response = self.llm.invoke(formatted_prompt)
        final_answer = response.content

        # 6. Save this interaction to the memory store!
        session_history.append({"role": "user", "content": query})
        session_history.append({"role": "ai", "content": final_answer})

        return final_answer