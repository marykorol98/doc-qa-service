import logging
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from doc_llm_core import DocLLM
from utils import get_google_doc_text

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

app = FastAPI()

llm = DocLLM(logger=logger)


@app.post("/upload")
async def upload_file(url: str) -> dict[str, str]:
    file_id = str(uuid4())
    # Загружаем текст Google Docs в plain text
    text = get_google_doc_text(url)

    llm.load_context_from_doc(file_id, text)

    return {"file_id": file_id}


@app.post("/ask")
async def ask_question(file_id: str, question: str) -> dict[str, str]:
    q_id = str(uuid4())

    llm.ask(q_id, file_id, question)

    return {"question_id": q_id}


@app.get("/answer")
async def get_answer(q_id: str):
    if q_id not in llm.questions:
        return JSONResponse(
            {"status": "error", "error": "question not found"}, status_code=404
        )

    q = llm.questions[q_id]

    return {
        "status": q["status"],
        "answer": q["answer"],
    }
