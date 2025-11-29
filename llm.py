# llm.py
from langchain_community.llms import HuggingFaceEndpoint
import os

def get_llm():
    return HuggingFaceEndpoint(
        repo_id=os.getenv("LLM_REPO_ID"),
        huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
    )
