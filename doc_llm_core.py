import re

from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import unicodedata
from langchain_classic.retrievers import MultiQueryRetriever
import torch
import ftfy

import os
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

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def logger(self, msg: str):
        if self._logger:
            self._logger.info(msg)

    def clean_text(self, raw_text: str) -> str:
        """
        Очистка текста.
        """
        text = ftfy.fix_text(raw_text)
        text = text.replace("\\n", "\n")  # экранированные переносы

        text = re.sub(r"\\+", "", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # сжимаем лишние пустые строки
        text = unicodedata.normalize("NFKC", text)

        # удаляем дублирующиеся строки, сохраняя порядок
        lines = text.split("\n")
        seen = set()
        unique_lines = []
        HAS_LETTERS = re.compile(r"[A-Za-zА-Яа-яЁё]")
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                if not HAS_LETTERS.search(line_stripped):
                    continue

                seen.add(line_stripped)
                unique_lines.append(line_stripped)

        return "\n".join(unique_lines).strip()

    def split_contract(self, text: str) -> list[str]:
        """
        Делит договор на смысловые разделы и по длине.
        Очень короткие чанки (< min_chunk_size) объединяются с предыдущими.
        """
        section_pattern = r"(?=(?:\n\d+\.\s+)|(?:\nПриложение\s*№\s*\d+))"
        raw_sections = re.split(section_pattern, text)

        return raw_sections

    # TODO: подумать, как реализовать
    # def build_summaries(self, chunks: list[str]) -> list[str]:
    #     """Создаёт краткие резюме для каждого фрагмента документа с прогресс-баром."""
    #     summaries = []
    #     logging.info("Начало создания резюме для %d фрагментов", len(chunks))

    #     for i, chunk in enumerate(tqdm(chunks, desc="Обработка фрагментов", unit="chunk"), start=1):
    #         try:
    #             prompt = [
    #                 ("system", "Ты юридический ассистент. Сделай краткое резюме следующего текста."),
    #                 ("human", chunk),
    #             ]
    #             summary = self.summary_llm_runable.invoke(prompt)
    #             summaries.append(summary)
    #         except Exception as e:
    #             logging.error("Ошибка при обработке фрагмента %d: %s", i, str(e))
    #             summaries.append("")  # добавляем пустое резюме при ошибке

    #     logging.info("Создание резюме завершено")
    #     return summaries

    def enumerate_chunks(self, chunks: list[str]) -> list[str]:
        """Добавляет CHUNK i/n заголовки."""
        total = len(chunks)
        output = []
        for i, ch in enumerate(chunks, start=1):
            output.append(f"=== CHUNK {i}/{total} ===\n{ch}")
        return output

    def build_vector_store(self, chunks: list[str], persist_dir: str = "chroma_store"):
        """Создаёт Chroma хранилище для текстов"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": self.device},
        )

        text_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=os.path.join(persist_dir),
        )

        return text_store

    def load_context_from_doc(self, doc_id, doc_text: str):
        text_clean = self.clean_text(doc_text)
        text_split = self.split_contract(text_clean)
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

    def get_retriever(self, vector_store):
        """Multihop: сначала summary, потом уточнение full_text."""
        summary_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        return MultiQueryRetriever.from_llm(llm=self.llm, retriever=summary_retriever)

    def get_rag_context(self, question: str, file_id: str) -> str:
        vector_store = self.context.get(file_id)
        if not vector_store:
            raise RuntimeError("Document is not loaded!")

        retriever = self.get_retriever(vector_store)
        texts = retriever.invoke(question)
        return "\n\n".join(d.page_content for d in texts)

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        context = self.get_rag_context(question, file_id)

        self.logger(f"context:\n\n{context}")

        prompt_messages = self.prompt(context=context, question=question)

        # Генерируем ответ через LLM
        answer = self.llm.invoke(prompt_messages).content
        self.logger(answer)

        self.questions[q_id] = {
            "file_id": file_id,
            "question": question,
            "status": "completed",
            "answer": answer,
        }

        return answer
