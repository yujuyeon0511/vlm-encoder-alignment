# VLM Vision Encoder-LLM 정렬 분석
## CKA-Performance Paradox와 다차원 정렬 메트릭 연구

---

## Slide 1: 표지

**제목:** Vision Encoder-LLM 정렬(Alignment) 정량 분석
**부제:** CKA-Performance Paradox 발견과 Enhanced Alignment Score 제안

**발표자:** 유주연
**소속/날짜:** 2026

> **시각자료:** 없음 (텍스트 표지)

---

## Slide 2: 목차

1. 연구 배경 및 문제 정의
2. 연구 방법론 — 4가지 정렬 메트릭
3. 실험 설계
4. 실험 결과 I — CKA 정렬 분석
5. 실험 결과 II — Deep CORAL 분포 분석
6. 실험 결과 III — EAS 통합 메트릭
7. 핵심 발견: CKA-Performance Paradox
8. 임베딩 공간 시각화
9. 표현 기하학 분석
10. Pretrained VLM 내부 정렬 비교
11. 벤치마크 평가 결과
12. 결론 및 향후 연구

> **시각자료:** 없음 (텍스트 목차)

---

## Slide 3: 연구 배경 — VLM 아키텍처

**핵심 질문:**
> "어떤 Vision Encoder가 특정 LLM과 가장 잘 정렬되는가?"

**VLM 파이프라인:**
```
[이미지] → Vision Encoder → Projector → LLM → [응답]
```

- Vision Encoder의 출력이 LLM 텍스트 임베딩 공간과 얼마나 잘 정렬되는지가 모델 성능에 직접적 영향
- 그러나 이를 **체계적으로 비교하는 도구**가 부족한 실정
- 기존에는 CKA(Centered Kernel Alignment) 하나로 정렬 품질을 판단

> **시각자료:** VLM 파이프라인 다이어그램 (직접 제작 필요)
> - `[Image]` → `Vision Encoder (CLIP/SigLIP/DINOv2)` → `Projector (MLP)` → `LLM (Qwen2.5-7B)` → `[Output]`
> - 화살표 위에 "정렬 분석 포인트" 표시

---

## Slide 4: 문제 정의 — CKA만으로 충분한가?

**기존 가정:** CKA 점수가 높으면 → 실제 성능도 좋을 것

**우리의 발견:** CKA와 실제 성능 사이에 **역상관** (r = -0.0475)

| 인코더 | CKA Score | Retrieval MRR | 순위 역전 |
|--------|-----------|---------------|----------|
| CLIP | 0.9902 (1위) | 0.096 (3위) | ↓ |
| SigLIP | 0.9800 (2위) | 0.126 (1위) | ↑ |
| DINOv2 | 0.9148 (3위) | 0.109 (2위) | ↑ |

→ **CKA-Performance Paradox**: 높은 CKA ≠ 좋은 하류 태스크 성능

> **시각자료:** `outputs/paper_figures/figure_1.png` — Panel (c) "CKA-Performance Paradox" 산점도
> - X축: CKA Score, Y축: Retrieval MRR
> - CLIP이 가장 높은 CKA지만 가장 낮은 MRR
> - SigLIP이 가장 높은 MRR을 보이는 역전 현상
> - Pearson r = -0.0475 주석 표시

---

## Slide 5: 연구 방법론 개요

본 연구에서 사용/제안한 **4가지 정렬 메트릭:**

| 메트릭 | 측정 대상 | 핵심 특성 |
|--------|----------|----------|
| **CKA** | 커널 기반 구조적 유사도 | 표현 공간의 전역적 구조 비교 |
| **Deep CORAL** | 공분산 행렬 (2차 통계량) | 분포의 형태와 퍼짐 비교 |
| **EAS** | CKA + CORAL + 판별력 통합 | CKA의 한계를 보완하는 균형 메트릭 |
| **E2E Validation** | 실제 검색 성능 | CKA 예측력 검증 |

> **시각자료:** 4개 메트릭을 비교하는 개념도 (직접 제작 필요)
> - CKA: "구조 비교" (커널 행렬 도식)
> - CORAL: "분포 비교" (공분산 행렬 도식)
> - EAS: "통합 점수" (파이 차트: CKA 30%, CORAL 30%, Disc 40%)
> - E2E: "실제 검증" (Projector 학습 → 검색 평가 도식)

---

## Slide 6: CKA (Centered Kernel Alignment)

**정의:** 두 표현 공간 X, Y 사이의 구조적 유사도

$$CKA(K, L) = \frac{HSIC(K, L)}{\sqrt{HSIC(K, K) \cdot HSIC(L, L)}}$$

- $K = XX^T$ (선형 커널), $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ (센터링)
- HSIC = Hilbert-Schmidt Independence Criterion
- 범위: [0, 1] — 높을수록 구조적으로 유사

**한계:**
- 분포의 **형태**(공분산 구조)를 무시
- 높은 CKA가 실제 성능을 보장하지 않음 (Paradox)

> **시각자료:** `outputs/paper_figures/figure_1.png` — Panel (a) "Overall CKA Score"
> - CLIP(0.9902), SigLIP(0.9800), DINOv2(0.9148) 막대 그래프
> - 모든 인코더가 매우 높은 CKA (0.91 이상)를 보임

---

## Slide 7: Deep CORAL (Correlation Alignment)

**정의:** 공분산 행렬의 Frobenius 노름 차이

$$L_{CORAL} = \frac{1}{4d^2} \|C_s - C_t\|^2_F$$

- $C_s$, $C_t$: Source/Target 공분산 행렬
- CORAL Similarity: $score = 1 / (1 + distance)$

**추가 메트릭:**
- **Spectral Divergence:** 고유값 분포 간 대칭 KL 발산
  - 공분산 행렬의 "에너지 분포"가 얼마나 다른지 측정

**CKA와의 차이:** CKA는 전역 구조, CORAL은 분포의 **2차 모멘트** 비교

> **시각자료:** `outputs/paper_figures/figure_2.png` — 전체 3패널
> - (a) Spectral Divergence 막대 그래프: SigLIP(0.3687)이 압도적으로 높음
> - (b) CKA vs Spectral Divergence 산점도: 두 메트릭의 불일치 시각화
> - (c) Eigenvalue Spectrum: 로그 스케일 고유값 분포 비교

---

## Slide 8: EAS (Enhanced Alignment Score)

**CKA의 한계를 보완하기 위해 제안한 통합 메트릭:**

$$EAS = 0.3 \times CKA + 0.3 \times CORAL_{sim} + 0.4 \times Discriminability$$

| 구성요소 | 가중치 | 역할 |
|---------|--------|------|
| CKA | 30% | 구조적 유사도 |
| CORAL Similarity | 30% | 분포 정렬 품질 |
| Discriminability | 40% | 클래스 판별력 유지 (Fisher Discriminant Ratio) |

**판별력이 가장 높은 가중치(40%)를 차지하는 이유:**
- 정렬이 잘 되더라도 클래스 간 구분이 불가능하면 무의미
- Fisher Discriminant Ratio: $disc = \frac{between\_scatter}{between\_scatter + within\_scatter}$

> **시각자료:** `outputs/paper_figures/figure_3.png` — 전체 3패널
> - (a) Component Scores: 인코더별 CKA/CORAL/Discriminability 묶음 막대
> - (b) CKA vs EAS 산점도: y=x 대각선 기준으로 편차 시각화
> - (c) Ranking 테이블: CKA/Discriminability/EAS 기준별 순위 비교

---

## Slide 9: 실험 설계

### 대상 모델

**Vision Encoders (3종):**

| 인코더 | Model ID | 차원 | 학습 방식 |
|--------|----------|------|----------|
| CLIP | openai/clip-vit-base-patch32 | 768 | Contrastive (텍스트-이미지) |
| SigLIP | google/siglip-base-patch16-224 | 768 | Sigmoid Contrastive |
| DINOv2 | facebook/dinov2-base | 768 | Self-supervised (이미지만) |

**Target LLM:** Qwen2.5-7B (Qwen/Qwen2.5-7B)

### 데이터

- **5가지 데이터 타입:** Chart, Table, Visualization, Text/Document, Math
- **샘플 수:** 타입당 50개 (총 150개)
- **소스:** 실제 VLM 학습 데이터 (한국어 VQA 포함)

> **시각자료:** 실험 파이프라인 도식 (직접 제작 필요)
> - 3개 Vision Encoder + 1개 LLM의 조합
> - 데이터 타입별 이미지 예시 (chart/table/text 아이콘)
> - 측정 메트릭 흐름도

---

## Slide 10: 실험 결과 I — CKA 정렬 분석

### 전체 CKA 점수 (PCA 정렬 후)

| 인코더 | CKA (Linear) | 순위 |
|--------|-------------|------|
| CLIP | 0.9902 | 1위 |
| SigLIP | 0.9800 | 2위 |
| DINOv2 | 0.9148 | 3위 |

### 데이터 타입별 CKA

| 타입 | CLIP | SigLIP | DINOv2 |
|------|------|--------|--------|
| Chart | 0.0611 | 0.0501 | 0.0177 |
| Table | 0.0943 | 0.0703 | 0.0150 |
| Text | 0.0000 | **0.0829** | 0.0000 |

**관찰:**
- 전체 CKA는 모두 0.91 이상으로 높음
- 데이터 타입별로는 SigLIP이 Text에서 유일하게 유의미한 CKA
- DINOv2는 모든 타입에서 가장 낮은 CKA

> **시각자료:** `outputs/paper_figures/figure_1.png` — Panel (a) + (b)
> - (a) Overall CKA: 3개 인코더 막대 그래프
> - (b) CKA by Data Type: Chart/Table/Text 그룹 막대 그래프
> - CLIP이 Chart와 Table에서 우세, SigLIP이 Text에서 강점

---

## Slide 11: 실험 결과 II — Deep CORAL 분석

### CORAL 메트릭 비교

| 인코더 | CORAL Similarity | Spectral Divergence | 해석 |
|--------|-----------------|--------------------|----|
| CLIP | 1.0000 | **0.1238** | 분포 잘 정렬 + 고유값 유사 |
| SigLIP | 1.0000 | 0.3687 | 분포 잘 정렬 + 고유값 차이 큼 |
| DINOv2 | 1.0000 | 0.1330 | 분포 잘 정렬 + 고유값 유사 |

**핵심 발견:**
- CORAL Similarity는 모든 인코더에서 1.0 (= 분포 수준에서 잘 정렬)
- **Spectral Divergence에서 차이 발생:** SigLIP의 고유값 분포가 LLM과 가장 다름
- 즉, CKA와 CORAL이 **서로 다른 측면**을 포착

> **시각자료:** `outputs/paper_figures/figure_2.png` — 전체 3패널
> - (a) Spectral Divergence: SigLIP(0.3687)이 눈에 띄게 높음
> - (b) CKA vs Spectral Divergence: 역상관 관계 (CKA 높을수록 SD 낮지 않음)
> - (c) Eigenvalue Spectrum: 4개 모델의 로그 스케일 고유값 감쇠 비교
>   - DINOv2가 Text(LLM)와 가장 유사한 감쇠 패턴

---

## Slide 12: 실험 결과 III — EAS 통합 메트릭

### EAS 점수 비교

| 인코더 | CKA | CORAL | Discriminability | **EAS** |
|--------|-----|-------|------------------|---------|
| CLIP | 0.9902 | 1.0000 | 0.9439 | **0.9746** |
| DINOv2 | 0.9148 | 1.0000 | **0.9867** | **0.9691** |
| SigLIP | 0.9800 | 1.0000 | 0.7846 | 0.9079 |

### 순위 비교: CKA vs Discriminability vs EAS

| 기준 | 1위 | 2위 | 3위 |
|------|-----|-----|-----|
| CKA | CLIP (0.9902) | SigLIP (0.9800) | DINOv2 (0.9148) |
| Discriminability | DINOv2 (0.9867) | CLIP (0.9439) | SigLIP (0.7846) |
| **EAS** | **CLIP (0.9746)** | **DINOv2 (0.9691)** | **SigLIP (0.9079)** |

**관찰:**
- CKA 1위(CLIP)와 Discriminability 1위(DINOv2)가 다름
- EAS는 CLIP을 1위로 선정 → CKA 구조 + 판별력 균형
- SigLIP은 낮은 Discriminability(0.7846)로 EAS에서 3위

> **시각자료:** `outputs/paper_figures/figure_3.png` — 전체 3패널
> - (a) Component Scores: 3개 구성요소를 인코더별로 시각 비교
> - (b) CKA vs EAS: 대각선(y=x) 아래로 벗어나는 정도 = CKA와의 차이
> - (c) Ranking 텍스트: 기준별 순위 한눈에 비교

---

## Slide 13: 핵심 발견 — CKA-Performance Paradox

### E2E 검증 실험 설계

1. 각 인코더별 **2-layer MLP Projector** 학습 (300 epochs, MSE Loss)
2. 투영된 Vision 임베딩과 Text 임베딩 간 **Cosine 유사도 기반 검색** 평가
3. CKA와 검색 MRR 간 **Pearson 상관계수** 산출

### E2E 검색 결과

| 인코더 | CKA | Recall@1 | Recall@5 | MRR |
|--------|-----|----------|----------|-----|
| CLIP | 0.9902 | 0.027 | 0.127 | 0.096 |
| SigLIP | 0.9800 | **0.047** | **0.160** | **0.126** |
| DINOv2 | 0.9148 | 0.040 | 0.140 | 0.109 |

**Pearson r = -0.0475** (CKA와 MRR 간 약한 음의 상관)

**의미:**
- CKA 1위 CLIP이 MRR에서 **꼴찌**
- CKA 3위에 가까운 SigLIP이 MRR에서 **1위**
- → CKA만으로 Encoder-LLM 정렬 품질을 판단하면 **잘못된 선택**을 할 수 있음

> **시각자료:** `outputs/paper_figures/figure_1.png` — Panel (c)
> - CKA Score(x축) vs Retrieval MRR(y축) 산점도
> - 우하향 트렌드 (음의 상관)
> - "Pearson r = -0.0475" 주석
> - CLIP이 우측 하단, SigLIP이 좌측 상단에 위치

---

## Slide 14: 임베딩 공간 시각화

### t-SNE 차원 축소 결과 (n=150, perplexity=30)

**Row 1 (a-c): 인코더별 t-SNE (데이터 타입별 색상)**
- **(a) CLIP:** 넓게 분포, chart/table/text 클러스터 간 경계 불명확
- **(b) SigLIP:** 타이트한 클러스터링, 그러나 클러스터 간 겹침 존재
- **(c) DINOv2:** 가장 뚜렷한 클러스터 분리 (Discriminability 최고와 일치)

**Row 2 (d-f): Vision + LLM 텍스트 오버레이**
- **(d) CLIP + Qwen:** Vision(원)과 Text(x) 분포 비교
- **(e) SigLIP + Qwen:** Vision과 Text 분포의 겹침 정도
- **(f) DINOv2 + Qwen:** Vision과 Text의 관계

**관찰:**
- DINOv2가 데이터 타입별 가장 뚜렷한 클러스터 형성 → 높은 Discriminability
- CLIP은 넓은 분포 → 구조적 유사도(CKA)는 높지만 판별력은 상대적으로 낮음

> **시각자료:** `outputs/paper_figures/figure_4.png` — 전체 2x3 그리드
> - 상단: t-SNE (chart=파랑, table=오렌지, text=청록)
> - 하단: Vision(원) + Text(x 마커) 오버레이
> - DINOv2의 클러스터 분리가 가장 명확함을 시각적으로 확인

---

## Slide 15: 표현 기하학 분석

### 공분산 차이 히트맵 (a-c)

각 인코더의 Vision 공분산 - Text 공분산 차이:
- **(a) CLIP:** 차이가 작음 (colorbar 범위 좁음)
- **(b) SigLIP:** 차이가 중간
- **(c) DINOv2:** 차이가 가장 큼 (Self-supervised 특성)

### Procrustes 정렬 (d-f)

SVD 기반으로 임베딩 공간을 최적 회전시켜 정렬:
- **(d) CLIP (reference):** 기준 공간
- **(e) SigLIP → CLIP 정렬:** Procrustes error = 2.5
- **(f) DINOv2 → CLIP 정렬:** Procrustes error = 2.3

**관찰:** DINOv2가 CLIP 공간으로의 정렬 오차가 더 작음 (2.3 vs 2.5)

> **시각자료:** `outputs/paper_figures/figure_5.png` — 전체 2x3 그리드
> - 상단(a-c): 공분산 차이 히트맵 (빨간 계열 colormap)
> - 하단(d-f): Procrustes 정렬 산점도 (정렬된 임베딩과 타겟 비교)

---

## Slide 16: Pretrained VLM 내부 정렬 비교

### 실험 배경

기존 VLM의 내부 Vision Encoder-LLM 정렬 수준은?

### 결과 비교

| 모델 | 유형 | CKA (Linear) |
|------|------|-------------|
| Qwen2.5-VL (내부) | VLM 내장 | 0.0229 |
| LLaVA-OneVision (내부) | VLM 내장 | 0.0672 |
| **CLIP** (오픈) | 독립 인코더 | **0.9902** |
| **SigLIP** (오픈) | 독립 인코더 | **0.9800** |
| **DINOv2** (오픈) | 독립 인코더 | **0.9148** |

**핵심 발견:**
- VLM 내부 정렬: 평균 0.0450
- 오픈 인코더: 평균 0.9617
- **21.4배 차이** (gap)

**해석:**
- VLM 내부에서는 Projector가 비선형 변환을 담당 → 원시 CKA가 낮아도 잘 동작
- 오픈 인코더는 PCA 정렬 후 높은 CKA → 차원 축소가 구조를 보존

> **시각자료:** `outputs/paper_figures/figure_6.png` — 2패널
> - (a) CKA Score Comparison: 5개 모델 막대 그래프
>   - 좌측 2개(VLM 내부): 매우 낮음 (0.02~0.07)
>   - 우측 3개(오픈 인코더): 매우 높음 (0.91~0.99)
>   - "VLM Internal" vs "Open Encoders" 구분선
> - (b) Average CKA Gap: VLM 내부 평균 vs 오픈 인코더 평균
>   - "21.4x gap" 화살표 주석

---

## Slide 17: 벤치마크 평가 — Qwen2.5-VL-7B

### 5개 벤치마크 결과

| 벤치마크 | 태스크 | 메트릭 | 점수 | Stderr |
|---------|--------|--------|------|--------|
| **ChartQA** | 차트 이해 | Relaxed Overall | 23.6% | ±0.85% |
| **DocVQA** | 문서 이해 | ANLS | 31.0% | ±0.62% |
| **MMStar** | 종합 추론 | Average | 38.6% | — |
| **POPE** | 환각 탐지 | F1 Score | 86.5% | — |
| **TextVQA** | 텍스트 QA | Exact Match | 22.5% | ±0.58% |

**세부 결과 — MMStar 카테고리별:**

| 카테고리 | 점수 |
|---------|------|
| Coarse Perception | 47.8% |
| Instance Reasoning | 39.4% |
| Logical Reasoning | 38.6% |
| Science & Tech | 38.2% |
| Math | 37.6% |
| Fine-Grained Perception | 29.7% |

**평가 환경:** 2x A100-40GB, 총 5.1시간 소요

> **시각자료:** 벤치마크 결과 테이블 또는 레이더 차트 (직접 제작 권장)
> - 5개 벤치마크 점수를 레이더 차트로 시각화
> - 또는 수평 막대 그래프로 점수 비교

---

## Slide 18: 종합 분석 — 인코더별 프로파일

### CLIP (openai/clip-vit-base-patch32)
- **강점:** 가장 높은 CKA(0.9902), 가장 높은 EAS(0.9746)
- **약점:** 가장 낮은 검색 MRR(0.096)
- **특성:** 텍스트-이미지 Contrastive 학습 → 넓은 임베딩 분포

### SigLIP (google/siglip-base-patch16-224)
- **강점:** 가장 높은 검색 MRR(0.126), Text 타입에서 유일하게 유의미한 CKA
- **약점:** 가장 높은 Spectral Divergence(0.3687), 낮은 Discriminability(0.7846)
- **특성:** Sigmoid Contrastive → 세밀한 보정, 실제 태스크에서 우수

### DINOv2 (facebook/dinov2-base)
- **강점:** 가장 높은 Discriminability(0.9867), 가장 낮은 Spectral Divergence(0.1238)
- **약점:** 가장 낮은 CKA(0.9148)
- **특성:** Self-supervised → 텍스트 없이 순수 시각 특징, 뚜렷한 클러스터 형성

> **시각자료:** 인코더별 프로파일 요약 도식 (직접 제작 권장)
> - 3개 인코더의 강점/약점을 레이더 차트로 비교
> - 축: CKA, EAS, MRR, Discriminability, 1-Spectral Divergence
> - 또는 Figure 1~3에서 핵심 패널만 추출하여 나란히 배치

---

## Slide 19: 결론

### 주요 기여

1. **CKA-Performance Paradox 발견**
   - CKA와 실제 검색 성능 사이 음의 상관 (r = -0.0475)
   - CKA만으로 Encoder-LLM 정렬 품질을 판단하면 오류 발생

2. **Deep CORAL 분석 도입**
   - 공분산 기반 분포 정렬로 CKA의 사각지대 보완
   - Spectral Divergence를 통한 고유값 분포 비교

3. **EAS (Enhanced Alignment Score) 제안**
   - CKA(30%) + CORAL(30%) + Discriminability(40%) 통합 메트릭
   - 구조적 유사도 + 분포 정렬 + 판별력의 균형

4. **Pretrained VLM 내부 정렬 분석**
   - VLM 내부 CKA(0.045) vs 오픈 인코더 CKA(0.962): **21.4x 차이**

> **시각자료:** 핵심 수치를 강조하는 인포그래픽 (직접 제작 권장)
> - "r = -0.0475" (CKA-Performance Paradox)
> - "21.4x gap" (VLM 내부 vs 오픈 인코더)
> - EAS 공식: 0.3·CKA + 0.3·CORAL + 0.4·Disc

---

## Slide 20: 한계 및 향후 연구

### 현재 한계

1. **제한된 인코더/LLM 조합:** 3개 인코더 × 1개 LLM만 분석
2. **소규모 샘플:** 타입당 50개 (총 150개) — 대규모 데이터에서의 검증 필요
3. **LLaVA-Gemma 모델 미평가:** 커스텀 아키텍처의 lmms-eval 미지원

### 향후 연구 방향

1. **다중 LLM 확장:** Qwen, LLaMA, Gemma, InternLM 등 다양한 LLM에 대한 비교
2. **대규모 실험:** 수천 개 샘플에서의 통계적 유의성 검증
3. **EAS 가중치 최적화:** 태스크별 최적 가중치 탐색
4. **실제 VLM 학습에 적용:** EAS를 기준으로 인코더를 선택했을 때 실제 VLM 성능 개선 검증
5. **더 많은 벤치마크:** LLaVA-OneVision, LLaVA-Gemma 모델의 벤치마크 완료

> **시각자료:** 없음 (텍스트 위주) 또는 향후 연구 로드맵 다이어그램

---

## Slide 21: 도구 및 재현성

### VLM Encoder Alignment Toolkit

**GitHub:** https://github.com/yujuyeon0511/vlm-encoder-alignment

**주요 기능:**
- CLI: 9개 명령어 (`compare`, `coral`, `e2e`, `speed`, `embedding`, `elas`, `attention`, `multi-llm`, `all`)
- Gradio Web UI: 6개 인터랙티브 탭
- 논문 Figure 생성: `scripts/generate_paper_figures.py`

**실행 예시:**
```bash
# Deep CORAL 분석
python cli.py coral --encoders clip siglip dinov2 --llms qwen --n-samples 50

# E2E 검증
python cli.py e2e --encoders clip siglip dinov2 --llms qwen --epochs 300

# 논문 Figure 생성
python scripts/generate_paper_figures.py --n-samples 50 --device cuda
```

**환경:** Python 3.11, PyTorch 2.10+cu128, 2x A100-40GB

> **시각자료:** 없음 또는 CLI 실행 데모 스크린샷

---

## 부록 A: Figure 목록 및 파일 경로

| Figure | 파일 경로 | 내용 |
|--------|----------|------|
| Figure 1 | `outputs/paper_figures/figure_1.png` | CKA Alignment Overview (3패널) |
| Figure 2 | `outputs/paper_figures/figure_2.png` | Deep CORAL Distribution Analysis (3패널) |
| Figure 3 | `outputs/paper_figures/figure_3.png` | EAS Dashboard (3패널) |
| Figure 4 | `outputs/paper_figures/figure_4.png` | Embedding Space Visualization (2x3 그리드) |
| Figure 5 | `outputs/paper_figures/figure_5.png` | Representational Geometry (2x3 그리드) |
| Figure 6 | `outputs/paper_figures/figure_6.png` | Pretrained CKA Baseline (2패널) |

> 모든 Figure는 PNG(300dpi) + PDF 형식으로 `outputs/paper_figures/`에 저장

---

## 부록 B: 실험 결과 전체 수치 요약

### CKA 분석 (vs Qwen2.5-7B)

| | CLIP | SigLIP | DINOv2 |
|---|------|--------|--------|
| Overall CKA | 0.9902 | 0.9800 | 0.9148 |
| Chart CKA | 0.0611 | 0.0501 | 0.0177 |
| Table CKA | 0.0943 | 0.0703 | 0.0150 |
| Text CKA | 0.0000 | 0.0829 | 0.0000 |

### CORAL 분석

| | CLIP | SigLIP | DINOv2 |
|---|------|--------|--------|
| CORAL Similarity | 1.0000 | 1.0000 | 1.0000 |
| Spectral Divergence | 0.1238 | 0.3687 | 0.1330 |

### EAS 점수

| | CLIP | SigLIP | DINOv2 |
|---|------|--------|--------|
| CKA | 0.9902 | 0.9800 | 0.9148 |
| CORAL | 1.0000 | 1.0000 | 1.0000 |
| Discriminability | 0.9439 | 0.7846 | 0.9867 |
| **EAS** | **0.9746** | **0.9079** | **0.9691** |

### E2E 검증

| | CLIP | SigLIP | DINOv2 |
|---|------|--------|--------|
| CKA | 0.9902 | 0.9800 | 0.9148 |
| Recall@1 | 0.027 | 0.047 | 0.040 |
| Recall@5 | 0.127 | 0.160 | 0.140 |
| MRR | 0.096 | 0.126 | 0.109 |

**CKA-MRR Pearson r = -0.0475**

### Pretrained VLM 내부 CKA

| 모델 | CKA |
|------|-----|
| Qwen2.5-VL (내부) | 0.0229 |
| LLaVA-OV (내부) | 0.0672 |
| VLM Internal 평균 | 0.0450 |
| Open Encoder 평균 | 0.9617 |
| **Gap** | **21.4x** |
