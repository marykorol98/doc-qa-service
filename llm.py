import requests
from huggingface_hub import InferenceClient
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DocLLM:
    MAX_LOG_LENGTH = 120
    MAX_CONTEXT_CHARS = 20000

    def __init__(self):
        self.model_id = os.getenv("LLM_REPO_ID")
        self.token = os.getenv("HF_API_TOKEN")

        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        self._model = InferenceClient(
            model=os.getenv("LLM_REPO_ID"),
            token=os.getenv("HF_API_TOKEN"),
        )
        self.context: dict[str, str] = {}

        self.questions: dict[str, str] = {}

    @property
    def model(self):
        return self._model

    def load_context_from_doc(self, doc_id, doc_text: str):
        self.context[doc_id] = doc_text
        logger.info(
            f"Saved context: {doc_text[: self.MAX_LOG_LENGTH] if len(doc_text) > self.MAX_LOG_LENGTH else doc_text}"
        )

    def prompt(self, context: str, question: str):
        context = context[: self.MAX_CONTEXT_CHARS]

        return f"""Контекст:
        {context}

        Вопрос: {question}

        Ответь кратко и по делу.
        """

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        prompt = self.prompt(self.context.get(file_id, ""), question)

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            self.api_url, headers=self.headers, json=payload, timeout=60
        )
        data = response.json()

        print(data)

        # Формат:
        # {
        #   "choices": [
        #     {"message": {"role": "assistant", "content": "..."}}
        #   ]
        # }

        answer = data["choices"][0]["message"]["content"]

        logger.info(
            f"Generated answer: {answer[: self.MAX_LOG_LENGTH] if len(answer) > self.MAX_LOG_LENGTH else answer}"
        )

        self.questions[q_id] = {
            "file_id": file_id,
            "question": question,
            "status": "completed",
            "answer": answer,
        }

        return answer
