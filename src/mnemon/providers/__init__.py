"""
Providers subpackage — LLM and embedding backend implementations.

Brain analog: The neurotransmitter synthesis machinery — the biochemical
substrate that transforms raw electrochemical potential into specific
signalling molecules.  Providers are the chemical factories that convert
text prompts into vector representations or language completions, enabling
all higher cognitive functions.

Currently available providers
------------------------------
- ``LiteLLMProvider``: LLM completions via litellm (100+ model backends)
- ``LiteLLMEmbeddingProvider``: Dense embeddings via litellm

All providers implement the abstract interfaces defined in
``mnemon.core.interfaces``, ensuring any provider can be swapped without
modifying calling code.
"""

from mnemon.providers.litellm_provider import LiteLLMEmbeddingProvider, LiteLLMProvider

__all__ = ["LiteLLMProvider", "LiteLLMEmbeddingProvider"]
