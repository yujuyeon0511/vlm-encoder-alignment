"""Unified LLM loading and text embedding extraction.

Consolidates LLM loading (previously duplicated in 6 files) into a single module.
"""

import gc
import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoModel, AutoTokenizer

from vlm_alignment.config import load_config, get_device


class LLMManager:
    """Manages LLM loading, caching, and text embedding extraction.

    Replaces duplicated LLM loading code across experiments.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self._models: Dict[str, object] = {}
        self._tokenizers: Dict[str, object] = {}

    def load(self, llm_name: str):
        """Load an LLM by name.

        Args:
            llm_name: Short name (e.g., 'llama', 'llama3', 'qwen', 'gemma3', 'internlm')

        Returns:
            Tuple of (model, tokenizer)
        """
        key = llm_name.lower()
        if key in self._models:
            return self._models[key], self._tokenizers[key]

        cfg = load_config()
        llm_map = cfg["models"]["llms"]

        if key not in llm_map:
            raise ValueError(
                f"Unknown LLM: {llm_name}. Available: {list(llm_map.keys())}"
            )

        model_id = llm_map[key]
        print(f"Loading {llm_name.upper()} from {model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._models[key] = model
        self._tokenizers[key] = tokenizer
        return model, tokenizer

    @torch.no_grad()
    def extract_text_embeddings(
        self, llm_name: str, texts: List[str], max_length: int = 128
    ) -> np.ndarray:
        """Extract text embeddings from an LLM.

        This single function replaces duplicated text embedding extraction
        code across 6 files in the original codebase.

        Args:
            llm_name: LLM name
            texts: List of text strings
            max_length: Maximum token length

        Returns:
            Embeddings array of shape (n_texts, hidden_dim)
        """
        model, tokenizer = self.load(llm_name)

        all_embeddings = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # Use last hidden state, mean-pool over non-padding tokens
            hidden = outputs.last_hidden_state  # (batch, seq, hidden)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    def unload(self, llm_name: Optional[str] = None):
        """Free model memory."""
        if llm_name:
            key = llm_name.lower()
            self._models.pop(key, None)
            self._tokenizers.pop(key, None)
        else:
            self._models.clear()
            self._tokenizers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def available_llms() -> List[str]:
        cfg = load_config()
        return list(cfg["models"]["llms"].keys())
