"""Fixed corpus for real-model promotion tests.

Covers:
  * Chat
  * Code
  * Structured JSON
  * Retrieval
  * Long context
  * Repeated-token stress
  * Activation outliers
  * Multi-turn conversations
"""
from __future__ import annotations

CORPUS: dict[str, str] = {
    "chat_short": "What is the capital of France?",
    "chat_medium": (
        "Explain the difference between supervised learning and reinforcement learning, "
        "including examples of each and when one might be preferred over the other."
    ),
    "code_python": (
        "Write a Python function that implements binary search on a sorted list. "
        "Include type hints and docstrings."
    ),
    "json_structured": (
        "Return a JSON object with fields: name (string), age (integer), hobbies (list of strings), "
        "and address (nested object with street and city)."
    ),
    "retrieval_fact": (
        "The Eiffel Tower is 330 meters tall. The Great Pyramid of Giza is 138 meters tall. "
        "The Empire State Building is 443 meters tall. "
        "Question: Which building is the tallest?"
    ),
    "long_context_summary": (
        "Summarize the following paragraph in one sentence: "
        "Machine learning is a subset of artificial intelligence that enables computers to learn "
        "from data without being explicitly programmed. It involves algorithms that improve through "
        "experience, identifying patterns in large datasets to make predictions or decisions. "
        "Deep learning, a branch of machine learning, uses neural networks with many layers to model "
        "complex patterns in data. These technologies power applications ranging from image recognition "
        "and natural language processing to autonomous vehicles and recommendation systems. "
        "The field continues to evolve rapidly, with new architectures and training techniques "
        "pushing the boundaries of what machines can learn and accomplish."
    ),
    "repeated_token_stress": (
        "Repeat the word 'token' 50 times in a single sentence that still makes grammatical sense."
    ),
    "outlier_math": (
        "Calculate 999999999 * 999999999 and explain why floating point arithmetic "
        "might give an approximate result."
    ),
    "multiturn_context": (
        "User: What is 2+2?\n"
        "Assistant: 2+2 equals 4.\n"
        "User: Now multiply that by 3.\n"
        "Assistant: 4 multiplied by 3 is 12.\n"
        "User: What was the original sum before multiplication?"
    ),
}


def get_corpus_prompts() -> list[tuple[str, str]]:
    """Return list of (prompt_id, prompt_text) tuples."""
    return list(CORPUS.items())


def get_corpus_hash() -> str:
    """Return SHA-256 hash of the canonical corpus."""
    import hashlib
    canonical = "\n".join(f"{k}:{v}" for k, v in sorted(CORPUS.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
