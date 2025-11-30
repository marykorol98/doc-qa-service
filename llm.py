import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
import logging
from dotenv import load_dotenv

load_dotenv()



class DocLLM:
    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger
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

        self.persist_dir: str = "chroma_store"
        
    def logger(self, msg: str):
        if self._logger:
            self._logger.info(msg)

    def clean_text(self, text: str) -> str:
        text = text.replace("\r", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text

    def text_splitter(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # ≈ tokens
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        return splitter.split_text(text)

    def enumerate_chunks(self, chunks: list[str]) -> list[str]:
        """Добавляет CHUNK i/n заголовки."""
        total = len(chunks)
        output = []
        for i, ch in enumerate(chunks, start=1):
            output.append(f"=== CHUNK {i}/{total} ===\n{ch}")
        return output

    def build_vector_store(self, chunks: list[str], persist_dir: str = "chroma_store"):
        """Создаёт Chroma векторное хранилище."""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
        )

        return Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    def load_context_from_doc(self, doc_id, doc_text: str):
        text_clean = self.clean_text(doc_text)
        text_split = self.text_splitter(text_clean)
        chunks_enumerated = self.enumerate_chunks(text_split)

        self.context[doc_id] = self.build_vector_store(
            chunks_enumerated, persist_dir=self.persist_dir
        )

    def prompt(self, context: str, question: str) -> list[tuple[str]]:
        return [
            (
                "system",
                f"Ты юридический ассистент. Отвечай строго по документу:\n{context}",
            ),
            ("human", question),
        ]

    @property
    def chain(self):
        """Вся схема целиком"""

        return self.prompt | self.llm

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        vector_store = self.context.get(file_id, "")

        if not vector_store:
            raise RuntimeError("Document is not loaded!")

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        texts = retriever._get_relevant_documents(question, run_manager=None)
        context = "\n\n".join(d.page_content for d in texts)

        self.logger(f"context:\n\n{context}")

        prompt_messages = self.prompt(context=context, question=question)

        # Генерируем ответ через LLM
        response = self.llm.invoke(prompt_messages)
        answer = response.content
        self.logger(answer)

        self.questions[q_id] = {
            "file_id": file_id,
            "question": question,
            "status": "completed",
            "answer": answer,
        }

        return answer
