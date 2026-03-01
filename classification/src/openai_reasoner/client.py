from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


def get_openai_client() -> Any | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)
