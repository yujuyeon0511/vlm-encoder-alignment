#!/usr/bin/env python3
"""Generate sample data for the repository.

Run once to populate sample_data/ with synthetic images and labels,
enabling users to test without external datasets.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_alignment.data.synthetic import DataGenerator


def main():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
    gen = DataGenerator(seed=42)

    # Charts
    chart_dir = os.path.join(base, "images", "chart")
    os.makedirs(chart_dir, exist_ok=True)
    for i in range(3):
        img = gen.generate_bar_chart(
            categories=[f"Item{j}" for j in range(4)],
            title=f"Sample Chart {i+1}",
        )
        img.save(os.path.join(chart_dir, f"chart_{i+1:02d}.png"))

    # Tables
    table_dir = os.path.join(base, "images", "table")
    os.makedirs(table_dir, exist_ok=True)
    for i in range(3):
        img = gen.generate_table_image(title=f"Sample Table {i+1}")
        img.save(os.path.join(table_dir, f"table_{i+1:02d}.png"))

    # Documents
    doc_dir = os.path.join(base, "images", "document")
    os.makedirs(doc_dir, exist_ok=True)
    for i in range(3):
        img = gen.generate_document_image(f"Sample Document {i+1}")
        img.save(os.path.join(doc_dir, f"document_{i+1:02d}.png"))

    # Labels
    labels = []
    for i in range(3):
        labels.append({
            "image": f"images/chart/chart_{i+1:02d}.png",
            "question": f"What is the highest value in this chart?",
            "answer": "The highest value varies by chart.",
            "data_type": "chart",
        })
    for i in range(3):
        labels.append({
            "image": f"images/table/table_{i+1:02d}.png",
            "question": f"What is the price of Banana?",
            "answer": "$0.50",
            "data_type": "table",
        })
    for i in range(3):
        labels.append({
            "image": f"images/document/document_{i+1:02d}.png",
            "question": f"What is the main topic of this document?",
            "answer": "The document covers introduction, methods, and results.",
            "data_type": "text",
        })

    with open(os.path.join(base, "labels.jsonl"), "w") as f:
        for item in labels:
            f.write(json.dumps(item) + "\n")

    print(f"Generated sample data in {base}/")
    print(f"  Charts: 3 images")
    print(f"  Tables: 3 images")
    print(f"  Documents: 3 images")
    print(f"  Labels: {len(labels)} entries")


if __name__ == "__main__":
    main()
