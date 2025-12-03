import re
import shutil
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import unicodedata
from langchain_classic.retrievers import MultiQueryRetriever
import torch
import ftfy

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
        
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def logger(self, msg: str):
        if self._logger:
            self._logger.info(msg)

    def clean_text(self, raw_text: str) -> str:
        """
        Чистит текст: исправляет кодировки, удаляет BOM, HTML-теги, лишние пробелы и дублирующиеся строки.
        """
        text = ftfy.fix_text(raw_text)
        text = text.replace("\ufeff", "")  # BOM
        text = text.replace("\r", "")  # возврат каретки
        text = text.replace("\\n", "\n")  # экранированные переносы
        text = text.replace('\\"', '"')  # экранированные кавычки

        text = re.sub(r"\\+", "", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # сжимаем лишние пустые строки
        text = re.sub(r"[ \t]+", " ", text)  # сжимаем пробелы и табы внутри строк
        text = re.sub(r"[ ]*\n[ ]*", "\n", text)  # убираем лишние пробелы вокруг \n
        text = re.sub(r"<[^>]+>", "", text)  # удаляем HTML-теги
        text = unicodedata.normalize("NFKC", text)

        # удаляем дублирующиеся строки, сохраняя порядок
        lines = text.split("\n")
        seen = set()
        unique_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                unique_lines.append(line_stripped)

        return "\n".join(unique_lines).strip()

    def split_contract(
        self, text: str, max_chunk_size: int = 2500, overlap: int = 200, min_chunk_size: int = 500
    ) -> list[str]:
        """
        Делит договор на смысловые разделы и по длине.
        Очень короткие чанки (< min_chunk_size) объединяются с предыдущими.
        """
        section_pattern = r"(?=(?:\n\d+\.\s+)|(?:\nПриложение\s*№\s*\d+))"
        raw_sections = re.split(section_pattern, text)

        sections = []
        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue
            lines = sec.split("\n", 1)
            header = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            sections.append(header + "\n" + body)

        chunks = []
        for sec in sections:
            start = 0
            while start < len(sec):
                end = start + max_chunk_size
                if end < len(sec):
                    end -= overlap
                part = sec[start:end].strip()
                if part:
                    # объединяем слишком короткие чанки с предыдущим
                    if chunks and len(part) < min_chunk_size:
                        chunks[-1] += "\n" + part
                    else:
                        chunks.append(part)
                start = end

        return chunks

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
            model_kwargs={"device": self.device},
        )

        return Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

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
        return vector_store.as_retriever(search_kwargs={"k": 5})
        # return MultiQueryRetriever.from_llm(
        #     llm=self.llm,
        #     retriever=vector_store.as_retriever(search_kwargs={"k": 5})
        # )

    def get_rag_context(self, question: str, file_id: str) -> str:
        """
        Получаем контекст из документа
        """
        vector_store = self.context.get(file_id)

        if not vector_store:
            raise RuntimeError("Document is not loaded!")

        retriever = self.get_retriever(vector_store)

        texts = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in texts)
        return context

    def ask(self, q_id: str, file_id: str, question: str) -> str:
        context = self.get_rag_context(question, file_id)

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
