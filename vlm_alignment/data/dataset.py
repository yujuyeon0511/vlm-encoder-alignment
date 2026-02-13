"""VLM dataset loader with configurable paths.

Replaces hardcoded /NetDisk/ paths with config-driven paths.
Falls back to sample_data/ when real data is not available.
"""

import json
import os
import random
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass

from vlm_alignment.config import load_config, get_data_root, get_project_root


@dataclass
class VLMSample:
    """A single VLM data sample."""

    image_path: str
    image: Image.Image
    question: str
    answer: str
    data_type: str  # 'chart', 'table', 'text', etc.


class VLMDataset:
    """VLM dataset loader with automatic fallback to sample data.

    Data root priority:
    1. Explicit data_root parameter
    2. VLM_DATA_ROOT environment variable
    3. config.yaml data.root
    4. sample_data/ (built-in synthetic data)
    """

    def __init__(self, data_root: Optional[str] = None, seed: int = 42):
        self.seed = seed
        random.seed(seed)

        if data_root and os.path.exists(data_root):
            self.data_root = data_root
            self._use_sample = False
        else:
            resolved = get_data_root()
            self.data_root = str(resolved)
            self._use_sample = not self._has_real_data(resolved)

        self.cfg = load_config()

    def _has_real_data(self, path) -> bool:
        """Check if real dataset structure exists at path."""
        cfg = load_config()
        for ds_cfg in cfg["data"]["datasets"].values():
            label_path = os.path.join(str(path), ds_cfg["label_file"])
            if os.path.exists(label_path):
                return True
        return False

    def _parse_jsonl_line(self, line: str) -> Optional[dict]:
        """Parse a single JSONL line from the VLM dataset."""
        try:
            data = json.loads(line.strip())
            messages = data.get("messages", [])

            question, image_path, answer = "", None, ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                question = item.get("text", "")
                            elif item.get("type") == "image":
                                image_path = item.get("image", item.get("path", ""))
                    elif isinstance(content, str):
                        question = content
                elif msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                answer = item.get("text", "")
                    elif isinstance(content, str):
                        answer = content

            return {"question": question, "answer": answer, "image_path": image_path}
        except (json.JSONDecodeError, KeyError):
            return None

    def _find_image(self, image_dir: str, sample_data: dict, line_idx: int) -> Optional[str]:
        """Find image file for a sample."""
        if sample_data.get("image_path"):
            img_path = sample_data["image_path"]
            for candidate in [img_path, os.path.join(self.data_root, img_path),
                              os.path.join(image_dir, os.path.basename(img_path))]:
                if os.path.exists(candidate):
                    return candidate

        if os.path.exists(image_dir):
            files = sorted(
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            )
            if line_idx < len(files):
                return os.path.join(image_dir, files[line_idx])
        return None

    def load_samples(self, data_type: str, n_samples: int = 30) -> List[VLMSample]:
        """Load samples of a specific data type.

        Args:
            data_type: 'chart', 'table', 'text', 'visualization', 'math'
            n_samples: Number of samples to load

        Returns:
            List of VLMSample
        """
        if self._use_sample:
            return self._load_sample_data(data_type, n_samples)

        datasets = self.cfg["data"]["datasets"]
        if data_type not in datasets:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(datasets.keys())}")

        ds_cfg = datasets[data_type]
        label_path = os.path.join(self.data_root, ds_cfg["label_file"])
        image_dir = os.path.join(self.data_root, ds_cfg["image_dir"])

        print(f"Loading {data_type} data from {label_path}...")

        samples_data = []
        with open(label_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if len(samples_data) >= n_samples * 10:
                    break
                parsed = self._parse_jsonl_line(line)
                if parsed and parsed["question"]:
                    parsed["line_idx"] = idx
                    samples_data.append(parsed)

        random.shuffle(samples_data)

        samples = []
        for sd in samples_data:
            if len(samples) >= n_samples:
                break
            img_path = self._find_image(image_dir, sd, sd["line_idx"])
            if img_path and os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    if max(image.size) > 512:
                        ratio = 512 / max(image.size)
                        image = image.resize(
                            (int(image.size[0] * ratio), int(image.size[1] * ratio)),
                            Image.LANCZOS,
                        )
                    samples.append(VLMSample(
                        image_path=img_path,
                        image=image,
                        question=sd["question"][:500],
                        answer=sd["answer"][:500],
                        data_type=data_type,
                    ))
                except Exception:
                    continue

        print(f"Loaded {len(samples)} {data_type} samples")
        return samples

    def _load_sample_data(self, data_type: str, n_samples: int) -> List[VLMSample]:
        """Load from built-in sample_data directory."""
        from vlm_alignment.data.synthetic import DataGenerator

        print(f"Using synthetic sample data for '{data_type}' (real data not available)")
        gen = DataGenerator(seed=self.seed)

        # Check if pre-generated sample images exist
        sample_dir = os.path.join(str(get_project_root()), "sample_data", "images", data_type)
        if os.path.exists(sample_dir):
            files = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
            samples = []
            for f in files[:n_samples]:
                path = os.path.join(sample_dir, f)
                img = Image.open(path).convert("RGB")
                samples.append(VLMSample(
                    image_path=path, image=img,
                    question=f"Describe this {data_type} image.",
                    answer="", data_type=data_type,
                ))
            if samples:
                return samples

        # Generate on the fly
        if data_type == "chart":
            images = [gen.generate_bar_chart(title=f"Chart {i+1}") for i in range(n_samples)]
        elif data_type == "table":
            images = [gen.generate_table_image(title=f"Table {i+1}") for i in range(n_samples)]
        elif data_type in ("text", "document"):
            images = [gen.generate_document_image(f"Doc {i+1}") for i in range(n_samples)]
        else:
            images = gen.generate_simple_images(n_samples)

        return [
            VLMSample(
                image_path="synthetic",
                image=img,
                question=f"Describe this {data_type} image.",
                answer="",
                data_type=data_type,
            )
            for img in images
        ]

    def load_mixed(self, n_per_type: int = 10, data_types: List[str] = None) -> Tuple[List[VLMSample], List[str]]:
        """Load mixed samples from multiple data types.

        Args:
            n_per_type: Samples per data type
            data_types: Types to load (default: chart, table, text)

        Returns:
            (samples, labels)
        """
        if data_types is None:
            data_types = ["chart", "table", "text"]

        all_samples, all_labels = [], []
        for dt in data_types:
            samples = self.load_samples(dt, n_per_type)
            all_samples.extend(samples)
            all_labels.extend([dt] * len(samples))

        combined = list(zip(all_samples, all_labels))
        random.shuffle(combined)
        if combined:
            all_samples, all_labels = zip(*combined)
            return list(all_samples), list(all_labels)
        return [], []

    def get_images_and_texts(self, samples: List[VLMSample]) -> Tuple[List[Image.Image], List[str]]:
        """Extract images and text queries from samples."""
        return [s.image for s in samples], [s.question for s in samples]
