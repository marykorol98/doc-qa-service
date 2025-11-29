from langchain_huggingface import HuggingFaceEndpoint
import os
import logging
from dotenv import load_dotenv

load_dotenv() 

logger = logging.getLogger(__name__)
MAX_LOG_LENGTH = 120


class DocLLM:
    def __init__(self):
        self._model = HuggingFaceEndpoint(
            repo_id=os.getenv("LLM_REPO_ID"),
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
            task="conversational"
        )
        self.context: dict[str, str] = {}

        self.questions: dict[str, str] = {}

    @property
    def model(self):
        return self._model

    def load_context_from_doc(self, doc_id, doc_text: str):
        self.context[doc_id] = doc_text
        logger.info(
            f"Saved context: {doc_text[:MAX_LOG_LENGTH] if len(doc_text) > MAX_LOG_LENGTH else doc_text}"
        )

    def prompt(self, context: str, question: str):
        return f"""Контекст:
        {context}

        Вопрос: {question}

        Ответь кратко и по делу.
        """

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        prompt = self.prompt(self.context.get(file_id, ""), question)

        answer = self.model.invoke(prompt)
        
        logger.info(
            f"Generated answer: {answer[:MAX_LOG_LENGTH] if len(answer) > MAX_LOG_LENGTH else answer}"
        )

        self.questions[q_id] = {
            "file_id": file_id,
            "question": question,
            "status": "completed",
            "answer": answer,
        }

        return answer
