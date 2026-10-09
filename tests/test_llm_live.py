"""Live check against the real Gemini API. Run with: uv run pytest -m network"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from mia.config import get_settings
from mia.llm import LLM

pytestmark = pytest.mark.network


class Capital(BaseModel):
    country: str
    capital: str


def test_live_extract(tmp_path):
    from dataclasses import replace

    s = get_settings()
    if not (s.gemini_api_key and s.model_bulk):
        pytest.skip("Set GEMINI_API_KEY and MIA_MODEL_BULK in .env")
    llm = LLM(replace(s, data_dir=tmp_path))
    result = llm.extract("What is the capital of Switzerland?", schema=Capital, model="bulk")
    assert result.capital.lower() == "bern"
    assert llm.usage.requests == 1
