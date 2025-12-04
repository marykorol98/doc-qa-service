import requests
import re
import random
import numpy as np
import torch


def get_google_doc_text(doc_url: str) -> str:
    """
    Получает текст Google Docs по ссылке.

    :param doc_url: Ссылка на Google Docs
    :return: Текст документа
    """
    # Пытаемся извлечь DOC_ID из ссылки
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", doc_url)
    if not match:
        raise ValueError("Неверная ссылка на Google Docs")

    doc_id = match.group(1)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

    response = requests.get(export_url)
    if response.status_code != 200:
        raise Exception(
            f"Не удалось скачать документ, статус код: {response.status_code}"
        )

    return response.text


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
