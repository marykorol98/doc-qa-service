import re
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import unicodedata
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document

import os
import logging
from dotenv import load_dotenv

from schemas import PROMPT_TYPE

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

    def clean_text(self, raw_text: str) -> str:
        text = raw_text.replace("\ufeff", "")  # BOM
        text = text.replace("\r", "")  # возврат каретки
        text = text.replace("\\n", "\n")  # экранированные переносы
        text = text.replace('\\"', '"')  # экранированные кавычки

        text = re.sub(r"\\+", "", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # сжимаем лишние пустые строки

        # Сжимаем пробелы и табы **внутри строк**, но не убираем \n
        text = re.sub(r"[ \t]+", " ", text)

        text = unicodedata.normalize("NFKC", text)
        # убираем лишние пробелы вокруг \n, но не объединяем все строки
        text = re.sub(r"[ ]*\n[ ]*", "\n", text)

        text = re.sub(r"<[^>]+>", "", text)  # удаляем HTML-теги

        return text.strip()

    def text_splitter(self, text: str) -> list[str]:
        """
        Сплиттер для договора:
        - делит на основной текст + приложения
        - потом рекурсивно делит каждый блок на пункты/подпункты
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[
                # "\n\n",
                r"Приложение\s*№\d+",
                r"Договор\s*№[\w\-.\/]+",
                r"\d+\.",
                r"\d+\.\d+\.\d+",
                r"\d+\.\d+",
                r"[a-zA-Z]\)",
                " ",
                "",
            ],
        )

        split_parts = splitter.split_text(text)

        return split_parts

    def enumerate_chunks(self, chunks: list[str]) -> list[str]:
        """Добавляет CHUNK i/n заголовки."""
        total = len(chunks)
        output = []
        for i, ch in enumerate(chunks, start=1):
            output.append(f"=== CHUNK {i}/{total} ===\n{ch}")
        return output

    def build_vector_store(self, chunks: list[str], persist_dir: str = "chroma_store"):
        """Создаёт Chroma векторное хранилище."""
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

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

    def prompt(self, context: str, question: str) -> PROMPT_TYPE:
        return [
            (
                "system",
                f"""Ты юридический ассистент. 
                Отвечай строго на основе предоставленного документа. 
                - Используй только информацию из документа. 
                - Давай краткие и точные ответы, без лишней информации. 
                - Отвечай строго на русском языке.
                - Запрещено использовать другие языки, включая китайский, английский, латиницу, и смешанные строки.
                - Если информации недостаточно для точного ответа, скажи «информация в документе отсутствует». 
                - Используй из внешних источников знания о том, в каком месте документа (РФ) обычно распологается искомая информация.
                - Если нужно искать числовые данные, то не забывай распознавать и сопоставлять числа (суммы, идентификаторы)
                Документ:
                {context}""",
            ),
            ("human", question),
        ]

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        vector_store = self.context.get(file_id)

        if not vector_store:
            raise RuntimeError("Document is not loaded!")

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        mq_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "Ты — помощник, который помогает искать юридические данные.\n"
                "Сгенерируй 5 разных формулировок вопроса, включая варианты, которые ищут информацию "
                "в приложениях, в шапках, подписях, реквизитах, таблицах, примечаниях.\n\n"
                "Вопрос: {question}"
            ),
        )

        raw = vector_store.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = 5

        hybrid_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25],
            weights=[0.6, 0.4],
        )

        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=hybrid_retriever,
            llm=self.llm,
            prompt=mq_prompt,
        )
        texts = multi_retriever.invoke(question)
        # texts = retriever._get_relevant_documents(question, run_manager=None)
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
