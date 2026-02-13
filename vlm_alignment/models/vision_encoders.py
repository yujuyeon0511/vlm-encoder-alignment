"""Unified vision encoder loading and embedding extraction.

Consolidates model loading (previously duplicated in 7 files) and embedding
extraction (previously in utils/embedding_utils.py) into a single module.
"""

import gc
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoProcessor, AutoImageProcessor,
)

from vlm_alignment.config import get_model_id, get_device


class VisionEncoderManager:
    """Manages vision encoder loading, caching, and embedding extraction.

    Replaces duplicated loading code across experiments with a single source.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self._models: Dict[str, object] = {}
        self._processors: Dict[str, object] = {}

    def load(self, encoder_name: str, with_attention: bool = False):
        """Load a vision encoder by name.

        Args:
            encoder_name: One of 'clip', 'siglip', 'dinov2', 'internvit', 'paligemma'
            with_attention: If True, load with eager attention for attention extraction

        Returns:
            Tuple of (model, processor)
        """
        cache_key = f"{encoder_name}_attn" if with_attention else encoder_name

        if cache_key in self._models:
            return self._models[cache_key], self._processors[cache_key]

        model_id = get_model_id("vision_encoders", encoder_name)
        print(f"Loading {encoder_name.upper()} from {model_id}...")

        loader = {
            "clip": self._load_clip,
            "siglip": self._load_siglip,
            "dinov2": self._load_dinov2,
            "internvit": self._load_internvit,
            "paligemma": self._load_paligemma,
        }

        if encoder_name not in loader:
            raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(loader.keys())}")

        model, processor = loader[encoder_name](model_id, with_attention)
        model.eval()

        self._models[cache_key] = model
        self._processors[cache_key] = processor
        return model, processor

    def _load_clip(self, model_id: str, with_attention: bool):
        kwargs = {"attn_implementation": "eager"} if with_attention else {}
        model = CLIPModel.from_pretrained(model_id, **kwargs).to(self.device)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor

    def _load_siglip(self, model_id: str, with_attention: bool):
        kwargs = {"attn_implementation": "eager"} if with_attention else {}
        model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

    def _load_dinov2(self, model_id: str, with_attention: bool):
        kwargs = {"attn_implementation": "eager"} if with_attention else {}
        model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device)
        processor = AutoImageProcessor.from_pretrained(model_id)
        return model, processor

    def _load_internvit(self, model_id: str, with_attention: bool):
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)
        processor = AutoImageProcessor.from_pretrained(model_id)
        return model, processor

    def _load_paligemma(self, model_id: str, with_attention: bool):
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        full_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)
        model = full_model.vision_tower
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        return model, processor

    @torch.no_grad()
    def extract_embeddings(
        self, encoder_name: str, images: List[Image.Image]
    ) -> np.ndarray:
        """Extract normalized embeddings from a vision encoder.

        This is the single unified function replacing duplicated extraction
        code across 7 files in the original codebase.

        Args:
            encoder_name: Encoder name (e.g., 'clip', 'siglip', 'dinov2')
            images: List of PIL images

        Returns:
            Normalized embeddings array of shape (n_images, embed_dim)
        """
        model, processor = self.load(encoder_name)

        extractor = {
            "clip": self._extract_clip,
            "siglip": self._extract_siglip,
            "dinov2": self._extract_dinov2,
            "internvit": self._extract_dinov2,  # Same CLS extraction
            "paligemma": self._extract_paligemma,
        }

        embeddings = extractor[encoder_name](model, processor, images)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    def _extract_clip(self, model, processor, images):
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return model.get_image_features(**inputs).cpu().numpy()

    def _extract_siglip(self, model, processor, images):
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return model.get_image_features(**inputs).cpu().numpy()

    def _extract_dinov2(self, model, processor, images):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0].float().cpu().numpy()

    def _extract_paligemma(self, model, processor, images):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs.get("pixel_values", inputs.get("image")).to(self.device)
        outputs = model(pixel_values)
        return outputs.last_hidden_state[:, 0].float().cpu().numpy()

    @torch.no_grad()
    def extract_with_attention(
        self, encoder_name: str, image: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embedding and attention map from a single image.

        Args:
            encoder_name: 'clip', 'siglip', or 'dinov2'
            image: Single PIL image

        Returns:
            Tuple of (embedding [1, dim], attention [n_patches])
        """
        model, processor = self.load(encoder_name, with_attention=True)

        if encoder_name == "clip":
            return self._extract_clip_attention(model, processor, image)
        elif encoder_name in ("dinov2", "siglip", "internvit"):
            return self._extract_generic_attention(model, processor, image)
        else:
            raise ValueError(f"Attention extraction not supported for {encoder_name}")

    def _extract_clip_attention(self, model, processor, image):
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        vision_out = model.vision_model(
            pixel_values=inputs["pixel_values"], output_attentions=True
        )
        pooled = vision_out.pooler_output
        emb = model.visual_projection(pooled).cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        attn = vision_out.attentions[-1][0, :, 0, 1:].mean(dim=0).cpu().numpy()
        return emb, attn

    def _extract_generic_attention(self, model, processor, image):
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_attentions=True)
        emb = outputs.last_hidden_state[:, 0].float().cpu().numpy()
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        attn = outputs.attentions[-1][0, :, 0, 1:].mean(dim=0).cpu().numpy()
        return emb, attn

    def extract_multi_encoder(
        self, encoder_names: List[str], images: List[Image.Image]
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings from multiple encoders.

        Args:
            encoder_names: List of encoder names
            images: List of PIL images

        Returns:
            Dict mapping encoder name to embeddings array
        """
        results = {}
        for name in encoder_names:
            results[name] = self.extract_embeddings(name, images)
        return results

    def unload(self, encoder_name: Optional[str] = None):
        """Free model memory.

        Args:
            encoder_name: Specific encoder to unload, or None for all
        """
        if encoder_name:
            keys_to_remove = [k for k in self._models if k.startswith(encoder_name)]
            for k in keys_to_remove:
                del self._models[k]
                del self._processors[k]
        else:
            self._models.clear()
            self._processors.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def available_encoders() -> List[str]:
        return ["clip", "siglip", "dinov2", "internvit", "paligemma"]
