"""CKA vs Retrieval Performance 실제 검증 실험.

기존 e2e_validation.py의 문제점:
1. 하드코딩된 결과값 사용 (실제 실험 아님)
2. 합성 데이터의 텍스트가 모두 동일 ("Describe this text image.")
3. Train/Test 분리 없이 같은 데이터로 학습+평가
4. 인코더 3개로 Pearson 상관 (통계적 의미 없음)

이 스크립트는 위 문제를 모두 수정하여 실제 검증을 수행한다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from vlm_alignment.analysis.cka import CKA
from vlm_alignment.models.vision_encoders import VisionEncoderManager
from vlm_alignment.models.llm_loaders import LLMManager
from vlm_alignment.models.projectors import create_projector
from vlm_alignment.data.synthetic import DataGenerator


def generate_diverse_data(n_per_type=20):
    """텍스트가 모두 다른 다양한 합성 데이터 생성."""
    gen = DataGenerator(seed=42)
    images = []
    texts = []
    labels = []

    # 차트: 각각 다른 제목과 값
    chart_topics = [
        ("Monthly Revenue", ["Jan", "Feb", "Mar", "Apr"]),
        ("Quarterly Sales", ["Q1", "Q2", "Q3", "Q4"]),
        ("Product Comparison", ["A", "B", "C", "D"]),
        ("Annual Growth", ["2020", "2021", "2022", "2023"]),
        ("Department Budget", ["HR", "Eng", "Sales", "Ops"]),
        ("Customer Ratings", ["Poor", "Fair", "Good", "Great"]),
        ("Market Share", ["US", "EU", "Asia", "Other"]),
        ("Temperature Data", ["Spring", "Summer", "Fall", "Winter"]),
        ("Exam Scores", ["Math", "Sci", "Eng", "Art"]),
        ("Website Traffic", ["Mon", "Tue", "Wed", "Thu"]),
        ("Inventory Count", ["Apples", "Oranges", "Grapes", "Bananas"]),
        ("Energy Usage", ["Solar", "Wind", "Hydro", "Nuclear"]),
        ("Population Stats", ["City A", "City B", "City C", "City D"]),
        ("Profit Margins", ["Tech", "Retail", "Health", "Finance"]),
        ("Survey Results", ["Yes", "No", "Maybe", "N/A"]),
        ("CPU Benchmark", ["Core i5", "Core i7", "Core i9", "Ryzen 9"]),
        ("Storage Prices", ["HDD", "SSD", "NVMe", "Cloud"]),
        ("Flight Prices", ["NYC", "LAX", "ORD", "DFW"]),
        ("Movie Ratings", ["Action", "Comedy", "Drama", "Horror"]),
        ("Coffee Sales", ["Espresso", "Latte", "Mocha", "Americano"]),
    ]
    for i in range(min(n_per_type, len(chart_topics))):
        title, cats = chart_topics[i]
        vals = np.random.RandomState(i).randint(10, 100, size=len(cats)).tolist()
        img = gen.generate_bar_chart(categories=cats, values=vals, title=title)
        images.append(img)
        texts.append(f"A bar chart titled '{title}' showing values for {', '.join(cats)} with values {vals}")
        labels.append("chart")

    # 테이블: 각각 다른 내용
    table_topics = [
        ("Employee Records", {"Name": ["Alice", "Bob", "Charlie"], "Age": [28, 35, 42], "Dept": ["Eng", "HR", "Sales"]}),
        ("Product Prices", {"Item": ["Laptop", "Phone", "Tablet"], "Price": [999, 699, 499], "Stock": [50, 200, 80]}),
        ("City Population", {"City": ["Seoul", "Tokyo", "Beijing"], "Pop(M)": [9.7, 13.9, 21.5], "Area": [605, 2191, 16410]}),
        ("Test Results", {"Subject": ["Math", "Science", "English"], "Score": [85, 92, 78], "Grade": ["B+", "A-", "C+"]}),
        ("Food Menu", {"Dish": ["Pasta", "Pizza", "Salad"], "Price": [12, 15, 9], "Cal": [650, 800, 300]}),
        ("Book List", {"Title": ["Novel A", "Guide B", "Manual C"], "Pages": [320, 180, 450], "Year": [2020, 2021, 2019]}),
        ("Car Models", {"Model": ["Sedan", "SUV", "Truck"], "MPG": [35, 25, 18], "HP": [180, 250, 350]}),
        ("Weather Data", {"City": ["Miami", "Denver", "Seattle"], "Temp": [85, 65, 55], "Rain": [60, 15, 37]}),
        ("Fruit Prices", {"Fruit": ["Apple", "Banana", "Cherry"], "Price/kg": [3, 1, 8], "Origin": ["US", "Ecuador", "Turkey"]}),
        ("Course List", {"Course": ["ML101", "DB201", "WEB301"], "Credits": [3, 4, 3], "Prof": ["Kim", "Lee", "Park"]}),
        ("Hotel Rooms", {"Type": ["Single", "Double", "Suite"], "Rate": [80, 120, 250], "Size": [20, 35, 60]}),
        ("Airline Routes", {"Route": ["NYC-LAX", "LAX-CHI", "CHI-NYC"], "Miles": [2475, 1745, 714], "Time": [5.5, 3.5, 2.0]}),
        ("Smartphone Specs", {"Model": ["PhoneA", "PhoneB", "PhoneC"], "RAM": [8, 12, 16], "Battery": [4000, 5000, 4500]}),
        ("Country GDP", {"Country": ["US", "China", "Japan"], "GDP(T)": [25.5, 17.9, 4.2], "Growth": [2.1, 5.2, 1.7]}),
        ("Gym Equipment", {"Item": ["Treadmill", "Bike", "Rower"], "Price": [800, 400, 600], "Weight": [80, 25, 35]}),
        ("Movie Box Office", {"Movie": ["Film A", "Film B", "Film C"], "Revenue(M)": [500, 300, 800], "Rating": [7.5, 6.2, 8.9]}),
        ("Plant Growth", {"Week": [1, 2, 3], "Height(cm)": [5, 12, 20], "Leaves": [2, 6, 10]}),
        ("Battery Test", {"Brand": ["AA Corp", "BB Inc", "CC Ltd"], "mAh": [2800, 3000, 2600], "Life(hr)": [8, 10, 7]}),
        ("Pizza Orders", {"Size": ["Small", "Medium", "Large"], "Orders": [150, 300, 200], "Revenue": [1500, 4500, 5000]}),
        ("Server Stats", {"Node": ["US-East", "US-West", "EU"], "CPU%": [65, 80, 45], "RAM%": [72, 55, 60]}),
    ]
    for i in range(min(n_per_type, len(table_topics))):
        title, data = table_topics[i]
        import pandas as pd
        df = pd.DataFrame(data)
        img = gen.generate_table_image(data=df, title=title)
        images.append(img)
        cols = list(data.keys())
        first_row = {k: v[0] for k, v in data.items()}
        texts.append(f"A table titled '{title}' with columns {cols}. First row: {first_row}")
        labels.append("table")

    # 문서: 각각 다른 제목
    doc_titles = [
        "Machine Learning Fundamentals",
        "Climate Change Analysis",
        "Financial Market Report",
        "Medical Research Summary",
        "Software Architecture Design",
        "Quantum Computing Overview",
        "Renewable Energy Systems",
        "Urban Planning Guidelines",
        "Nutrition Science Review",
        "Cybersecurity Best Practices",
        "Space Exploration Timeline",
        "Genetics Research Paper",
        "Autonomous Vehicles Study",
        "Blockchain Technology Report",
        "Ocean Conservation Plan",
        "Educational Psychology Review",
        "Agricultural Innovation",
        "Digital Marketing Strategy",
        "Robotics Engineering Manual",
        "Public Health Policy Brief",
    ]
    for i in range(min(n_per_type, len(doc_titles))):
        img = gen.generate_document_image(doc_titles[i])
        images.append(img)
        texts.append(f"A document about '{doc_titles[i]}' containing sections on introduction, methods, and results")
        labels.append("document")

    print(f"Generated {len(images)} samples ({len(set(texts))} unique texts)")
    return images, texts, labels


def evaluate_retrieval(projected_vision, text_embeddings):
    """Retrieval 평가: Recall@1, Recall@5, MRR."""
    v_norm = projected_vision / (np.linalg.norm(projected_vision, axis=1, keepdims=True) + 1e-8)
    t_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    sim = v_norm @ t_norm.T
    n = sim.shape[0]

    recall_1, recall_5, mrr = 0, 0, 0.0
    for i in range(n):
        ranking = np.argsort(-sim[i])
        rank = np.where(ranking == i)[0][0] + 1
        if rank == 1:
            recall_1 += 1
        if rank <= 5:
            recall_5 += 1
        mrr += 1.0 / rank

    return {
        "recall_1": recall_1 / n,
        "recall_5": recall_5 / n,
        "mrr": mrr / n,
    }


def run_validation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_names = ["clip", "siglip", "dinov2"]
    llm_name = "llama"
    n_per_type = 20  # 총 60개 샘플
    epochs = 500
    n_runs = 3  # 안정적 결과를 위해 여러 번 실행

    print("=" * 70)
    print("CKA vs Retrieval Performance - 실제 검증 실험")
    print("=" * 70)
    print(f"Encoders: {encoder_names}")
    print(f"LLM: {llm_name}")
    print(f"Samples: {n_per_type} per type x 3 types = {n_per_type * 3}")
    print(f"Projector epochs: {epochs}")
    print(f"Runs per encoder: {n_runs}")
    print()

    # 1. 다양한 데이터 생성
    print("[1/4] Generating diverse data...")
    images, texts, labels = generate_diverse_data(n_per_type=n_per_type)
    n_total = len(images)

    # 2. Vision embedding 추출
    print(f"\n[2/4] Extracting vision embeddings...")
    ve_mgr = VisionEncoderManager(device=device)
    vision_embs = {}
    for enc in encoder_names:
        vision_embs[enc] = ve_mgr.extract_embeddings(enc, images)
        print(f"  {enc}: shape={vision_embs[enc].shape}")
        ve_mgr.unload(enc)
    ve_mgr.unload()

    # 3. Text embedding 추출
    print(f"\n[3/4] Extracting text embeddings from {llm_name}...")
    llm_mgr = LLMManager(device=device)
    text_embs = llm_mgr.extract_text_embeddings(llm_name, texts)
    print(f"  text: shape={text_embs.shape}")
    llm_mgr.unload()

    # 4. 평가
    print(f"\n[4/4] Running CKA + Retrieval evaluation...")
    print("=" * 70)

    # Train/Test 분리 (70/30)
    n_train = int(n_total * 0.7)
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples\n")

    results = {}
    for enc in encoder_names:
        print(f"--- {enc.upper()} ---")

        v_emb = vision_embs[enc]
        v_train, v_test = v_emb[train_idx], v_emb[test_idx]
        t_train, t_test = text_embs[train_idx], text_embs[test_idx]

        # CKA (전체 데이터)
        cka_full = CKA.compute_cka(v_emb, text_embs, kernel="linear")
        cka_rbf = CKA.compute_cka(v_emb, text_embs, kernel="rbf")
        # CKA (테스트 데이터만)
        cka_test = CKA.compute_cka(v_test, t_test, kernel="linear")
        print(f"  CKA (linear, full): {cka_full:.4f}")
        print(f"  CKA (rbf, full):    {cka_rbf:.4f}")
        print(f"  CKA (linear, test): {cka_test:.4f}")

        # 여러 번 Projector 학습 + Retrieval 평가
        run_results = []
        for run in range(n_runs):
            v_dim, t_dim = v_emb.shape[1], text_embs.shape[1]
            projector = create_projector("2layer_mlp", v_dim, t_dim).to(device)
            optimizer = torch.optim.Adam(projector.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            v_tensor = torch.FloatTensor(v_train).to(device)
            t_tensor = torch.FloatTensor(t_train).to(device)

            # 학습
            projector.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                projected = projector(v_tensor)
                loss = criterion(projected, t_tensor)
                loss.backward()
                optimizer.step()

            train_loss = loss.item()

            # Test set에서 평가
            projector.eval()
            with torch.no_grad():
                v_test_tensor = torch.FloatTensor(v_test).to(device)
                projected_test = projector(v_test_tensor).cpu().numpy()
                test_loss = nn.MSELoss()(
                    projector(v_test_tensor),
                    torch.FloatTensor(t_test).to(device)
                ).item()

            retrieval = evaluate_retrieval(projected_test, t_test)
            run_results.append(retrieval)

        # 평균 성능
        avg_r1 = np.mean([r["recall_1"] for r in run_results])
        avg_r5 = np.mean([r["recall_5"] for r in run_results])
        avg_mrr = np.mean([r["mrr"] for r in run_results])
        std_mrr = np.std([r["mrr"] for r in run_results])

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        print(f"  Recall@1:   {avg_r1:.4f} (avg over {n_runs} runs)")
        print(f"  Recall@5:   {avg_r5:.4f}")
        print(f"  MRR:        {avg_mrr:.4f} +/- {std_mrr:.4f}")
        print()

        results[enc] = {
            "cka_linear": cka_full,
            "cka_rbf": cka_rbf,
            "cka_test": cka_test,
            "recall_1": avg_r1,
            "recall_5": avg_r5,
            "mrr": avg_mrr,
            "mrr_std": std_mrr,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }

    # 요약
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Encoder':<10} {'CKA(lin)':<10} {'CKA(rbf)':<10} {'R@1':<10} {'R@5':<10} {'MRR':<15} {'TrainLoss':<10} {'TestLoss':<10}")
    print("-" * 85)
    for enc in encoder_names:
        r = results[enc]
        print(f"{enc:<10} {r['cka_linear']:<10.4f} {r['cka_rbf']:<10.4f} {r['recall_1']:<10.4f} {r['recall_5']:<10.4f} {r['mrr']:<10.4f}+/-{r['mrr_std']:<4.4f} {r['train_loss']:<10.4f} {r['test_loss']:<10.4f}")

    # 상관관계 (n=3이라 참고용)
    cka_vals = [results[e]["cka_linear"] for e in encoder_names]
    mrr_vals = [results[e]["mrr"] for e in encoder_names]
    r1_vals = [results[e]["recall_1"] for e in encoder_names]

    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(cka_vals, mrr_vals)
    r_spearman, p_spearman = spearmanr(cka_vals, mrr_vals)
    r_r1, p_r1 = pearsonr(cka_vals, r1_vals)

    print(f"\n--- Correlation Analysis (n={len(encoder_names)}, 참고용) ---")
    print(f"CKA vs MRR:      Pearson r={r_pearson:.4f} (p={p_pearson:.4f}), Spearman r={r_spearman:.4f}")
    print(f"CKA vs Recall@1: Pearson r={r_r1:.4f} (p={p_r1:.4f})")
    print()
    print("NOTE: n=3으로는 통계적 유의성을 판단할 수 없습니다.")
    print("인코더 5개 이상 + 실제 데이터셋에서의 추가 검증이 필요합니다.")

    return results


if __name__ == "__main__":
    run_validation()
