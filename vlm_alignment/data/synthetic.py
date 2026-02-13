"""Synthetic data generation for testing without external datasets."""

import io
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict


class DataGenerator:
    """Generate synthetic images and labels for VLM testing."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_simple_images(self, n_images: int = 20, image_size: int = 224) -> List[Image.Image]:
        """Generate geometric shape images."""
        images = []
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
        shapes = ["circle", "rectangle", "triangle"]

        for i in range(n_images):
            img = Image.new("RGB", (image_size, image_size), "white")
            draw = ImageDraw.Draw(img)
            color = colors[i % len(colors)]
            shape = shapes[i % len(shapes)]

            margin = image_size // 4
            x1 = np.random.randint(margin, image_size - margin * 2)
            y1 = np.random.randint(margin, image_size - margin * 2)
            size = np.random.randint(margin, margin * 2)
            x2, y2 = x1 + size, y1 + size

            if shape == "circle":
                draw.ellipse([x1, y1, x2, y2], fill=color, outline="black")
            elif shape == "rectangle":
                draw.rectangle([x1, y1, x2, y2], fill=color, outline="black")
            elif shape == "triangle":
                draw.polygon([(x1 + size // 2, y1), (x1, y2), (x2, y2)], fill=color, outline="black")

            images.append(img)
        return images

    def generate_categorized_images(self, n_per_category: int = 10, image_size: int = 224) -> Tuple[List[Image.Image], List[str]]:
        """Generate images with category labels."""
        images, labels = [], []
        categories = {
            "circles": lambda d, c: d.ellipse([50, 50, 174, 174], fill=c, outline="black"),
            "squares": lambda d, c: d.rectangle([50, 50, 174, 174], fill=c, outline="black"),
            "triangles": lambda d, c: d.polygon([(112, 50), (50, 174), (174, 174)], fill=c, outline="black"),
        }
        colors = ["red", "blue", "green", "yellow", "purple"]

        for category, draw_func in categories.items():
            for i in range(n_per_category):
                img = Image.new("RGB", (image_size, image_size), "white")
                draw_func(ImageDraw.Draw(img), colors[i % len(colors)])
                images.append(img)
                labels.append(category)
        return images, labels

    def generate_bar_chart(self, categories: List[str] = None, values: List[float] = None, title: str = "Bar Chart") -> Image.Image:
        """Generate a bar chart image."""
        if categories is None:
            categories = ["A", "B", "C", "D"]
        if values is None:
            values = [np.random.uniform(10, 100) for _ in categories]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        bars = ax.bar(categories, values, color=colors, edgecolor="black")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Value")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close()
        return img

    def generate_table_image(self, data: pd.DataFrame = None, title: str = "Table") -> Image.Image:
        """Generate a table image."""
        if data is None:
            data = pd.DataFrame({
                "Product": ["Apple", "Banana", "Orange", "Grape"],
                "Price": ["$1.00", "$0.50", "$0.75", "$2.00"],
                "Qty": ["100", "150", "80", "60"],
            })

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        table = ax.table(cellText=data.values, colLabels=data.columns,
                         cellLoc="center", loc="center",
                         colColours=["#4472C4"] * len(data.columns))
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("black")
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close()
        return img

    def generate_document_image(self, title: str = "Sample Document", sections: int = 3) -> Image.Image:
        """Generate a document-like image with text blocks."""
        width, height = 600, 800
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()

        y = 30
        draw.text((50, y), title, fill="black", font=title_font)
        y += 50
        draw.line([(50, y), (550, y)], fill="gray", width=2)
        y += 20

        section_titles = ["Introduction", "Methods", "Results", "Conclusion"]
        lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor."

        for i in range(min(sections, len(section_titles))):
            draw.text((50, y), section_titles[i], fill="darkblue", font=title_font)
            y += 35
            words = lorem.split()
            line = ""
            for word in words:
                test = line + " " + word if line else word
                if len(test) > 60:
                    draw.text((50, y), line, fill="black", font=body_font)
                    y += 20
                    line = word
                else:
                    line = test
            if line:
                draw.text((50, y), line, fill="black", font=body_font)
                y += 20
            y += 30

        return img

    def generate_sample_dataset(self) -> Tuple[List[Image.Image], List[str], List[str]]:
        """Generate a small mixed dataset with images, texts, and labels.

        Returns:
            (images, texts, labels)
        """
        images, texts, labels = [], [], []

        # Charts
        for i in range(3):
            cats = [f"Cat{j}" for j in range(4)]
            vals = [np.random.uniform(10, 100) for _ in cats]
            images.append(self.generate_bar_chart(cats, vals, f"Chart {i+1}"))
            texts.append(f"What is the highest value in chart {i+1}?")
            labels.append("chart")

        # Tables
        for i in range(3):
            images.append(self.generate_table_image(title=f"Table {i+1}"))
            texts.append(f"What is the price of Banana in table {i+1}?")
            labels.append("table")

        # Documents
        for i in range(3):
            images.append(self.generate_document_image(f"Document {i+1}"))
            texts.append(f"What is the main topic of document {i+1}?")
            labels.append("text")

        return images, texts, labels
