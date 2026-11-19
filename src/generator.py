"""Answer generation with Qwen/Qwen3-0.6B from retrieved sources."""
from __future__ import annotations

import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase using ONLY the context "
    "provided. Be concise and precise. If the context does not contain the "
    "answer, say so. Never invent file names, options or APIs."
)


def read_span(source: MinimalSource) -> str:
    """Re-read a chunk's text from its file using its character span."""
    try:
        with open(source.file_path, "r", encoding="utf-8",
                  errors="replace", newline="") as f:
            return f.read()[source.first_character_index:
                            source.last_character_index]
    except OSError:
        return ""


class AnswerGenerator:
    """Loads the model once and answers questions from sources."""

    def __init__(self, max_new_tokens: int = 256,
                 max_context_chars: int = 6000) -> None:
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model.eval()

    def build_prompt(self, question: str,
                     sources: List[MinimalSource]) -> str:
        """Best sources first, stopping when the character budget is used."""
        parts: List[str] = []
        used = 0
        for i, s in enumerate(sources, 1):
            text = read_span(s)
            if not text.strip():
                continue
            room = self.max_context_chars - used
            if room <= 0:
                break
            text = text[:room]
            parts.append(f"[{i}] {s.file_path}\n{text}")
            used += len(text)
        user = ("Context:\n" + "\n\n".join(parts)
                + f"\n\nQuestion: {question}\nAnswer:")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    def generate(self, question: str, sources: List[MinimalSource]) -> str:
        """Greedy-decode an answer grounded in the given sources."""
        prompt = self.build_prompt(question, sources)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()