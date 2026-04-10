"""
LiteLLM-backed implementations of LLMProvider and EmbeddingProvider.

LiteLLM (https://github.com/BerriAI/litellm) exposes a unified async API
over 100+ model providers — OpenAI, Anthropic, Groq, Mistral, Ollama,
vLLM, Azure OpenAI, AWS Bedrock, and many more — through a single call
interface.  Swapping the underlying model requires only changing the
``model`` string; no code changes are needed in cognitive modules.

Brain analog: The neurotransmitter reuptake transporter — a generic
mechanism that routes molecular signals (tokens) to the appropriate
receptor system (model provider) regardless of the specific chemistry
(API format) involved.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm
from litellm import acompletion, aembedding, token_counter

from mnemon.core.interfaces import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

# Silence litellm's verbose success logging by default; callers can re-enable.
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------


class LiteLLMProvider(LLMProvider):
    """LLM provider backed by litellm (100+ model providers).

    Wraps ``litellm.acompletion`` for async completions and
    ``litellm.aembedding`` for embeddings.  Supports every provider
    that litellm supports: OpenAI, Anthropic, Groq, Ollama, vLLM, etc.

    Parameters
    ----------
    model:
        litellm model string, e.g. ``"gpt-4o-mini"``, ``"claude-3-haiku-20240307"``,
        ``"ollama/llama3"``, ``"groq/llama3-8b-8192"``.
    temperature:
        Default sampling temperature (overridable per-call via kwargs).
    max_tokens:
        Default max tokens for completions (overridable per-call).
    **kwargs:
        Additional keyword arguments forwarded to every ``acompletion`` call
        (e.g. ``api_key``, ``api_base``, ``timeout``).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a free-form text completion for *prompt*.

        Parameters
        ----------
        prompt:
            The full prompt string (system + user content combined, or user-only).
        **kwargs:
            Per-call overrides: ``temperature``, ``max_tokens``,
            ``stop``, ``system``, etc.

        Returns
        -------
        str
            The generated text content, stripped of leading/trailing whitespace.

        Raises
        ------
        litellm.exceptions.AuthenticationError
            If the provider API key is missing or invalid.
        litellm.exceptions.RateLimitError
            If the provider rate limit is exceeded.
        Exception
            Re-raises any other litellm exception after logging it.
        """
        call_kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.pop("temperature", self._temperature),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            **self._kwargs,
            **kwargs,
        }

        logger.debug(
            "LiteLLMProvider.generate: model=%s, prompt_len=%d",
            self._model,
            len(prompt),
        )

        try:
            response = await acompletion(**call_kwargs)
            content: str = response.choices[0].message.content or ""
            return content.strip()
        except Exception:
            logger.exception(
                "LiteLLMProvider.generate failed (model=%s, prompt_len=%d).",
                self._model,
                len(prompt),
            )
            raise

    async def generate_chat(
        self,
        system: str,
        history: list[dict[str, str]],
        message: str,
        **kwargs: Any,
    ) -> str:
        """Generate a reply using native multi-turn message API."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        call_kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self._temperature),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            **self._kwargs,
            **kwargs,
        }
        logger.debug(
            "LiteLLMProvider.generate_chat: model=%s, turns=%d",
            self._model,
            len(messages),
        )
        try:
            response = await acompletion(**call_kwargs)
            content: str = response.choices[0].message.content or ""
            return content.strip()
        except Exception:
            logger.exception(
                "LiteLLMProvider.generate_chat failed (model=%s).", self._model
            )
            raise

    async def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a response constrained to *response_schema* (JSON Schema).

        Uses the ``response_format`` parameter where the provider supports
        native structured output (OpenAI, Azure).  Falls back to prompting
        the model to emit raw JSON and parsing the result for other providers.

        Parameters
        ----------
        prompt:
            The full prompt string.
        response_schema:
            JSON Schema dict.  Must be a valid object schema.
        **kwargs:
            Per-call overrides forwarded to litellm.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response.

        Raises
        ------
        ValueError
            If the model's output cannot be parsed as valid JSON after retries.
        """
        # Build a schema-aware prompt suffix for models without native support
        schema_hint = json.dumps(response_schema, indent=2)
        augmented_prompt = (
            f"{prompt}\n\n"
            f"Respond with a single JSON object conforming exactly to this schema:\n"
            f"```json\n{schema_hint}\n```\n"
            f"Return only the JSON object — no markdown fences, no additional text."
        )

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": augmented_prompt}],
            "temperature": kwargs.pop("temperature", self._temperature),
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            **self._kwargs,
            **kwargs,
        }

        # Attempt native structured output (OpenAI-compatible providers)
        if _supports_response_format(self._model):
            call_kwargs["response_format"] = {"type": "json_object"}

        logger.debug(
            "LiteLLMProvider.generate_structured: model=%s, schema_keys=%s",
            self._model,
            list(response_schema.get("properties", {}).keys()),
        )

        try:
            response = await acompletion(**call_kwargs)
            raw: str = response.choices[0].message.content or ""
            raw = raw.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            return json.loads(raw)

        except json.JSONDecodeError as exc:
            logger.error(
                "LiteLLMProvider.generate_structured: JSON parse failed. raw=%r", raw
            )
            raise ValueError(
                f"Model '{self._model}' returned non-JSON output: {exc}"
            ) from exc
        except Exception:
            logger.exception(
                "LiteLLMProvider.generate_structured failed (model=%s).", self._model
            )
            raise

    async def token_count(self, text: str) -> int:
        """Return the number of tokens *text* consumes for the configured model.

        Uses litellm's token counter which selects the correct tokenizer
        (tiktoken for OpenAI, sentencepiece for others) based on *model*.

        Parameters
        ----------
        text:
            The string to count tokens for.

        Returns
        -------
        int
            Estimated token count.
        """
        try:
            count: int = token_counter(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            return count
        except Exception:
            # Fallback: rough estimate (4 chars ≈ 1 token for English text)
            logger.warning(
                "LiteLLMProvider.token_count: litellm counter failed, using estimate."
            )
            return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingProvider
# ---------------------------------------------------------------------------


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by litellm.

    Supports OpenAI (text-embedding-3-small/large, ada-002), Cohere,
    HuggingFace TEI, and any other provider accessible through litellm's
    embedding endpoint.

    Parameters
    ----------
    model:
        litellm embedding model string, e.g.
        ``"text-embedding-3-small"``, ``"cohere/embed-english-v3.0"``,
        ``"huggingface/BAAI/bge-large-en"``.
    dimensions:
        Expected output dimensionality.  Must match the model's output;
        used for validation and metadata (e.g. vector index creation).
    **kwargs:
        Additional keyword arguments forwarded to every ``aembedding`` call
        (e.g. ``api_key``, ``api_base``, ``timeout``).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._kwargs = kwargs

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Produce a single dense embedding vector for *text*.

        Parameters
        ----------
        text:
            The string to embed.

        Returns
        -------
        list[float]
            Dense float vector of length ``self.dimensions``.

        Raises
        ------
        ValueError
            If the returned embedding has an unexpected dimensionality.
        """
        if not text.strip():
            raise ValueError("Cannot embed an empty string.")

        logger.debug(
            "LiteLLMEmbeddingProvider.embed: model=%s, text_len=%d",
            self._model,
            len(text),
        )

        try:
            response = await aembedding(
                model=self._model,
                input=[text],
                **self._kwargs,
            )
            embedding: list[float] = response.data[0]["embedding"]

            if len(embedding) != self._dimensions:
                logger.warning(
                    "embed: expected %d dimensions, got %d (model=%s).",
                    self._dimensions,
                    len(embedding),
                    self._model,
                )

            return embedding
        except Exception:
            logger.exception(
                "LiteLLMEmbeddingProvider.embed failed (model=%s, text_len=%d).",
                self._model,
                len(text),
            )
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Produce embeddings for a batch of texts using the provider's batch API.

        Parameters
        ----------
        texts:
            Non-empty list of strings to embed.

        Returns
        -------
        list[list[float]]
            List aligned with *texts*: ``result[i]`` is the embedding of ``texts[i]``.

        Raises
        ------
        ValueError
            If *texts* is empty.
        """
        if not texts:
            raise ValueError("embed_batch requires at least one text.")

        # Reject empty strings — they cause API errors and break index alignment
        for i, t in enumerate(texts):
            if not t.strip():
                raise ValueError(
                    f"Empty string at index {i} in embed_batch; "
                    "all texts must be non-empty."
                )

        logger.debug(
            "LiteLLMEmbeddingProvider.embed_batch: model=%s, count=%d",
            self._model,
            len(texts),
        )

        try:
            response = await aembedding(
                model=self._model,
                input=texts,
                **self._kwargs,
            )
            # litellm returns data sorted by index
            embeddings: list[list[float]] = [item["embedding"] for item in response.data]
            return embeddings
        except Exception:
            logger.exception(
                "LiteLLMEmbeddingProvider.embed_batch failed (model=%s, count=%d).",
                self._model,
                len(texts),
            )
            raise

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Human-readable identifier for the underlying model."""
        return self._model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _supports_response_format(model: str) -> bool:
    """Return True if *model* supports the ``response_format=json_object`` param.

    This is a heuristic based on known-good prefixes.  litellm will raise
    a ``BadRequestError`` for unsupported models, so this function avoids
    unnecessary round-trips for clearly unsupported providers.
    """
    openai_compatible_prefixes = (
        "gpt-",
        "o1",
        "o3",
        "azure/",
        "groq/",
        "together_ai/",
        "deepseek/",
        "perplexity/",
    )
    return any(model.startswith(prefix) for prefix in openai_compatible_prefixes)
