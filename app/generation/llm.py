"""LLM providers.

* ``anthropic``  – Claude via the official Anthropic SDK (needs ``ANTHROPIC_API_KEY``).
* ``extractive`` – no LLM: answers by quoting the most relevant sentences of
  the retrieved chunks, each with its citation marker. Fully offline and
  deterministic; used when no API key is available and in tests.

Both return text that uses ``[S#]`` markers referring to the retrieved chunks
in order, so citation handling is identical for either provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.exceptions import LLMConfigurationError, LLMGenerationError
from app.core.logging import get_logger
from app.embeddings.embedder import Embedder
from app.generation.prompts import INSUFFICIENT_EVIDENCE_TAG, SYSTEM_PROMPT, build_messages
from app.ingestion.chunker import is_heading, split_sentences
from app.ingestion.metadata import DEPT_REF_RE, RBI_REF_RE
from app.retrieval.keyword_search import tokenize
from app.schemas.models import ChatTurn, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class LLMResult:
    text: str
    model: str | None
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    name: str
    model: str | None

    def generate(self, question: str, retrieved: list[RetrievedChunk], history: list[ChatTurn]) -> LLMResult: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 4000,
        timeout: float = 120.0,
        enable_refusal_fallback: bool = True,
        client=None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.enable_refusal_fallback = enable_refusal_fallback
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise LLMConfigurationError(
                    "ANTHROPIC_API_KEY is not set. Add it to .env or set LLM_PROVIDER=extractive."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def generate(self, question: str, retrieved: list[RetrievedChunk], history: list[ChatTurn]) -> LLMResult:
        import anthropic

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": build_messages(question, retrieved, history),
            "output_config": {"effort": self.effort},
        }
        if self.enable_refusal_fallback:
            # Server-side fallback: if the model declines on a safety
            # classifier, the API retries on a fallback model in the same call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMConfigurationError("Anthropic API key was rejected") from exc
        except anthropic.RateLimitError as exc:
            raise LLMGenerationError("LLM rate limit reached; retry shortly") from exc
        except anthropic.APIStatusError as exc:
            raise LLMGenerationError(f"LLM API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMGenerationError("Could not reach the LLM API") from exc

        if response.stop_reason == "refusal":
            raise LLMGenerationError("The model declined to answer this request")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMGenerationError(f"Empty LLM response (stop_reason={response.stop_reason})")
        usage = getattr(response, "usage", None)
        return LLMResult(
            text=text,
            model=getattr(response, "model", self.model),
            provider=self.name,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


_LOWER_WORD_RE = re.compile(r"\b[a-z]{2,}\b")


def _is_prose(sentence: str) -> bool:
    """Reject title blocks / reference lines: real provisions contain ordinary lower-case words."""
    if RBI_REF_RE.search(sentence) or DEPT_REF_RE.search(sentence):
        return False
    return len(_LOWER_WORD_RE.findall(sentence)) >= 5


class ExtractiveProvider:
    """Offline fallback: returns the best-matching source sentences verbatim.

    Sentences are scored by cosine similarity to the question (same embedding
    model as retrieval) plus keyword overlap; the top sentences are returned
    in source order with their ``[S#]`` marker. Nothing is paraphrased, so the
    output cannot contain claims absent from the sources.
    """

    name = "extractive"
    model = None

    def __init__(self, embedder: Embedder, max_sentences: int = 5, min_score: float = 0.35) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.min_score = min_score

    def generate(self, question: str, retrieved: list[RetrievedChunk], history: list[ChatTurn]) -> LLMResult:
        candidates: list[tuple[int, int, str]] = []  # (source_idx, sentence_idx, sentence)
        for s_idx, item in enumerate(retrieved, start=1):
            j = 0
            for paragraph in item.chunk.text.split("\n"):
                if is_heading(paragraph):
                    continue  # headings/banners are not claims
                for sentence in split_sentences(paragraph):
                    if 40 <= len(sentence) <= 800 and _is_prose(sentence):
                        candidates.append((s_idx, j, sentence))
                    j += 1
        if not candidates:
            return LLMResult(
                text=f"{INSUFFICIENT_EVIDENCE_TAG}\nThe retrieved passages contain no usable sentences.",
                model=None,
                provider=self.name,
            )

        q_vec = self.embedder.embed_query(question)
        s_vecs = self.embedder.embed_documents([c[2] for c in candidates])
        semantic = s_vecs @ q_vec
        q_tokens = set(tokenize(question))
        overlap = np.array(
            [len(q_tokens & set(tokenize(c[2]))) / max(1, len(q_tokens)) for c in candidates], dtype=np.float32
        )
        scores = 0.8 * semantic + 0.2 * overlap

        order = [i for i in np.argsort(-scores) if scores[i] >= self.min_score][: self.max_sentences]
        if not order:
            return LLMResult(
                text=f"{INSUFFICIENT_EVIDENCE_TAG}\nNo sentence in the retrieved passages directly addresses the question.",
                model=None,
                provider=self.name,
            )
        order.sort(key=lambda i: (candidates[i][0], candidates[i][1]))
        lines = [
            "Regulatory requirements (verbatim extracts; no LLM is configured, so this is not a synthesised answer):",
            "",
        ]
        lines += [f"- {candidates[i][2].strip()} [S{candidates[i][0]}]" for i in order]
        return LLMResult(text="\n".join(lines), model=None, provider=self.name)
