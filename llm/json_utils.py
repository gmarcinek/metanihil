import json
import re
from typing import Optional


def parse_json_with_markdown_blocks(response: str) -> Optional[dict]:
    """
    Prosty parser JSON dla odpowiedzi LLM.
    Najpierw próbuje sparsować cały tekst jako JSON,
    potem szuka JSON w blokach markdown ```json ... ```,
    a na końcu szuka w dowolnym bloku ``` ... ```.
    Nie modyfikuje tekstu (bez zamiany cudzysłowów).
    """
    response = response.strip()

    # 1. Spróbuj wczytać cały tekst jako JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 2. Szukaj JSON w bloku markdown ```json ... ```
    json_blocks = re.findall(r"```json(.*?)```", response, re.DOTALL)
    for block in json_blocks:
        block = block.strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # 3. Szukaj JSON w dowolnym bloku ``` ... ```
    generic_blocks = re.findall(r"```(.*?)```", response, re.DOTALL)
    for block in generic_blocks:
        block = block.strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # 4. Spróbuj wyodrębnić JSON jako pierwszy obiekt JSON z tekstu (np. bez bloków)
    json_like = re.search(r"\{.*\}", response, re.DOTALL)
    if json_like:
        candidate = json_like.group(0).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None
