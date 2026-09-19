"""
LLM provider router. Both providers expose the same call_llm() interface
so the rest of the app (and the eval script) can swap between them freely
and log latency/cost consistently - this is what makes the benchmark
possible.
"""
import time
from dataclasses import dataclass
from enum import Enum

from app.config import settings


class Provider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = settings.PRICING.get(model)
    if not pricing:
        return 0.0
    return (
        (input_tokens / 1000) * pricing["input"]
        + (output_tokens / 1000) * pricing["output"]
    )


def _call_gemini(prompt: str) -> LLMResponse:
    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)

    start = time.time()
    response = model.generate_content(prompt)
    latency = time.time() - start

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    return LLMResponse(
        text=response.text,
        provider=Provider.GEMINI.value,
        model=settings.GEMINI_MODEL,
        latency_seconds=round(latency, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=_estimate_cost(
            settings.GEMINI_MODEL, input_tokens, output_tokens
        ),
    )


def _call_groq(prompt: str) -> LLMResponse:
    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)

    start = time.time()
    completion = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    usage = completion.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return LLMResponse(
        text=completion.choices[0].message.content,
        provider=Provider.GROQ.value,
        model=settings.GROQ_MODEL,
        latency_seconds=round(latency, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=_estimate_cost(
            settings.GROQ_MODEL, input_tokens, output_tokens
        ),
    )


def call_llm(prompt: str, provider: Provider) -> LLMResponse:
    if provider == Provider.GEMINI:
        return _call_gemini(prompt)
    if provider == Provider.GROQ:
        return _call_groq(prompt)
    raise ValueError(f"Unknown provider: {provider}")
