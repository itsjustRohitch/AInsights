import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests


def extract_code_block(text: str) -> str:
    """
    Extract the first fenced Python code block.
    Falls back to raw text if no fence exists.
    """
    if not text:
        return ""

    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text.strip()


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """
    Try to extract JSON from a model response.
    """
    if not text:
        return None

    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:1.5b-instruct"
    timeout: int = 120

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        num_predict: int = 512,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()