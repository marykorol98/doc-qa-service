import re
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
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

        # LangChain модель
        base_llm = HuggingFaceEndpoint(
            repo_id=self.model_id,
            huggingfacehub_api_token=self.token,
            task="conversational",
        )

        self.llm = ChatHuggingFace(llm=base_llm)

        self.context = {}
        self.questions = {}

    def clean_text(self, text: str) -> str:
        text = text.replace("\r", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text

    def load_context_from_doc(self, doc_id, doc_text: str):
        self.context[doc_id] = self.clean_text(doc_text)
        logger.info(
            f"Saved context: {doc_text[: self.MAX_LOG_LENGTH] if len(doc_text) > self.MAX_LOG_LENGTH else doc_text}"
        )

    @property
    def prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                ("system", "Ты отвечаешь кратко по делу."),
                ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
            ]
        )

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        context = self.context.get(file_id, "")

        chain = self.prompt | self.llm
        result = chain.invoke({"context": context, "question": question})

        answer = result.content
        print(result)
        self.questions[q_id] = {
            "file_id": file_id,
            "question": question,
            "status": "completed",
            "answer": answer,
        }

        return answer
