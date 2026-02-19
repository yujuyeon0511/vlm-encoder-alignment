# VLM Encoder Alignment Toolkit - 프로젝트 상세 문서

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [연구 배경 및 동기](#2-연구-배경-및-동기)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [핵심 분석 메트릭](#4-핵심-분석-메트릭)
5. [지원 모델](#5-지원-모델)
6. [모듈별 상세 설명](#6-모듈별-상세-설명)
7. [실험 코드 상세](#7-실험-코드-상세)
8. [생성된 분석 결과 이미지](#8-생성된-분석-결과-이미지)
9. [샘플 데이터](#9-샘플-데이터)
10. [실행 방법](#10-실행-방법)
11. [주요 연구 발견](#11-주요-연구-발견)
12. [설정 및 의존성](#12-설정-및-의존성)

---

## 1. 프로젝트 개요

### 프로젝트 목적

본 프로젝트는 **Vision Language Model(VLM)에서 Vision Encoder와 Large Language Model(LLM) 간의 정렬(Alignment)을 정량적으로 분석하는 도구**이다. 다양한 Vision Encoder(CLIP, SigLIP, DINOv2, InternViT, PaliGemma)가 LLM(LLaMA, Qwen, Gemma, InternLM)의 텍스트 임베딩 공간과 얼마나 잘 정렬되는지를 측정하고, 그 결과를 시각화한다.

### 핵심 질문

> **"어떤 Vision Encoder가 특정 LLM과 가장 잘 맞는가?"**
> **"기존 CKA 메트릭만으로 Encoder-LLM 정렬 품질을 판단할 수 있는가?"**

### 주요 기능

| 기능 | 설명 |
|------|------|
| **CKA 정렬 분석** | Vision-Text 임베딩 간 구조적 유사도 측정 |
| **Deep CORAL 분석** | 공분산 행렬 기반 분포 정렬 분석 |
| **EAS 점수** | CKA + CORAL + 판별력을 결합한 통합 메트릭 |
| **ELAS 점수** | Encoder-LLM Alignment Score (CKA + MSE + Generalization + Cosine) |
| **추론 속도 벤치마크** | 지연시간, 처리량, GPU 메모리 프로파일링 |
| **Attention Map 시각화** | 패치-텍스트 유사도 히트맵 오버레이 |
| **임베딩 공간 시각화** | t-SNE/UMAP/Procrustes 차원 축소 시각화 |
| **E2E 검증** | CKA vs 실제 검색 성능의 상관관계 분석 |

### 인터페이스

- **CLI** (`cli.py`): 9개 명령어를 지원하는 통합 커맨드라인 인터페이스
- **Gradio Web UI** (`app.py`): 6개 탭으로 구성된 웹 기반 인터랙티브 UI

---

## 2. 연구 배경 및 동기

### VLM 아키텍처에서의 Encoder 선택 문제

VLM은 일반적으로 다음 구조를 따른다:

```
[이미지] → Vision Encoder → Projector → LLM → [응답]
```

이 파이프라인에서 **Vision Encoder의 출력이 LLM의 텍스트 임베딩 공간과 얼마나 잘 정렬되는지**가 모델 성능에 직접적인 영향을 미친다. 그러나 기존에는 이를 체계적으로 비교하는 도구가 부족했다.

### CKA-Performance Paradox의 발견

본 프로젝트의 핵심 발견 중 하나는 **CKA-Performance Paradox**이다:

- CKA(Centered Kernel Alignment)는 두 표현 공간의 **구조적 유사도**를 측정하는 대표적 메트릭
- 그러나 실험 결과, **CKA 점수가 높을수록 실제 검색(retrieval) 성능이 오히려 낮아지는 역상관** (r = -0.99)이 관찰됨
- 이는 CKA만으로는 Encoder-LLM 정렬 품질을 판단하기 불충분하다는 것을 의미
- 이를 해결하기 위해 **Deep CORAL** 기반의 분포 정렬 분석과 **EAS(Enhanced Alignment Score)** 통합 메트릭을 개발

### 이론적 배경

| 메트릭 | 측정 대상 | 한계 |
|--------|----------|------|
| **CKA** | 커널 기반 구조적 유사도 | 분포 형태 무시, 높은 CKA ≠ 좋은 성능 |
| **CORAL** | 공분산 행렬의 2차 통계량 정렬 | 특징 공변량과 분포 폭 포착 |
| **EAS** | CKA + CORAL + 판별력 통합 | CKA 패러독스를 보완하는 균형 메트릭 |

---

## 3. 디렉토리 구조

```
vlm-encoder-alignment/
├── README.md                          # 프로젝트 영문 README
├── PROJECT_DETAIL.md                  # 프로젝트 상세 문서 (본 문서)
├── config.yaml                        # 중앙 설정 파일 (모델 ID, 데이터 경로)
├── requirements.txt                   # Python 의존성
├── pyproject.toml                     # 패키지 메타데이터
├── cli.py                            # CLI 진입점 (271줄, 9개 명령어)
├── app.py                            # Gradio 웹 UI (399줄, 6개 탭)
├── .gitignore                        # Git 무시 규칙
│
├── sample_data/                       # 내장 합성 샘플 데이터
│   ├── labels.jsonl                   # 9개 샘플 레이블 (chart/table/document)
│   └── images/
│       ├── chart/                     # chart_01~03.png (합성 막대 그래프)
│       ├── table/                     # table_01~03.png (합성 테이블 이미지)
│       └── document/                  # document_01~03.png (합성 문서 이미지)
│
├── outputs/                           # 실험 결과 출력 디렉토리
│   └── coral/                         # Deep CORAL 분석 결과
│       ├── eas_dashboard.png          # EAS 대시보드 (3개 서브플롯)
│       ├── cka_vs_coral.png           # CKA vs CORAL 비교 산점도
│       ├── coral_comparison.png       # CORAL 거리/유사도 비교 막대 그래프
│       ├── covariance_heatmaps.png    # 공분산 행렬 히트맵 (9개 서브플롯)
│       ├── eigenvalue_spectrum.png    # 고유값 스펙트럼 비교
│       └── intra_modal_similarity.png # 인코더 간 유사도 행렬
│
├── scripts/
│   ├── generate_sample_data.py        # 합성 샘플 데이터 생성 스크립트
│   └── generate_paper_figures.py      # 논문용 고품질 그림 생성
│
└── vlm_alignment/                     # 메인 패키지 (~4,351줄)
    ├── __init__.py
    ├── config.py                      # 설정 로더 (86줄)
    │
    ├── models/                        # 모델 로딩 및 관리
    │   ├── vision_encoders.py         # Vision Encoder 통합 관리자 (241줄)
    │   ├── llm_loaders.py             # LLM 통합 로더 (129줄)
    │   └── projectors.py             # 4가지 Projector 아키텍처 + 학습 (~200줄)
    │
    ├── analysis/                      # 분석 알고리즘
    │   ├── cka.py                     # CKA 구현체 (102줄)
    │   ├── coral.py                   # Deep CORAL 분석 (~468줄)
    │   ├── alignment.py               # 정렬 분석 + ELAS 점수 (195줄)
    │   └── speed_benchmark.py         # 추론 속도 프로파일링 (~250줄)
    │
    ├── visualization/                 # 시각화 모듈
    │   ├── plot_style.py              # 통일된 스타일/색상 테마
    │   ├── alignment_plots.py         # CKA 비교 차트, 히트맵
    │   ├── attention_maps.py          # Attention 히트맵 오버레이
    │   ├── embedding_space.py         # t-SNE/UMAP 시각화
    │   ├── coral_plots.py             # CORAL/EAS 대시보드
    │   └── speed_plots.py             # 속도 벤치마크 대시보드
    │
    ├── data/                          # 데이터 로딩
    │   ├── dataset.py                 # VLM 데이터셋 로더 (자동 폴백)
    │   └── synthetic.py               # 합성 데이터 생성기
    │
    └── experiments/                   # 실험 실행기 (오케스트레이션)
        ├── encoder_comparison.py      # 인코더 비교 실험
        ├── coral_alignment.py         # CORAL 분석 실험 실행기
        ├── e2e_validation.py          # CKA vs 성능 검증 실험
        ├── elas_score.py              # ELAS 점수 계산 실험
        └── speed_benchmark.py         # 속도 벤치마크 실험 실행기
```

---

## 4. 핵심 분석 메트릭

### 4.1 CKA (Centered Kernel Alignment)

**파일:** `vlm_alignment/analysis/cka.py` (102줄)

두 표현 공간 X, Y 사이의 구조적 유사도를 측정하는 메트릭이다.

**수학적 정의:**

```
CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
```

- `K = XX^T` (선형 커널) 또는 `K = exp(-||x_i - x_j||^2 / 2σ^2)` (RBF 커널)
- `HSIC(K, L) = 1/(n-1)^2 * tr(KHLH)` (Hilbert-Schmidt Independence Criterion)
- `H = I - (1/n)11^T` (센터링 행렬)

**출력 범위:** [0, 1] (높을수록 유사)

**지원 커널:**
- `linear`: K = XX^T (빠르고 안정적)
- `rbf`: Gaussian RBF (비선형 관계 포착, 적응형 sigma)

**주요 메서드:**
- `CKA.compute_cka(X, Y, kernel='linear')`: 두 임베딩 간 CKA 계산
- `CKA.compute_pairwise(embeddings, kernel)`: 여러 임베딩 간 쌍별 CKA 행렬 계산

### 4.2 Deep CORAL (Correlation Alignment)

**파일:** `vlm_alignment/analysis/coral.py` (~468줄)

**논문:** Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016

CKA와 달리 **2차 통계량(공분산 행렬)**을 직접 정렬하여 분포 수준의 정렬을 측정한다.

**CORAL Distance:**

```
L_CORAL = (1 / 4d^2) * ||C_s - C_t||^2_F
```

- `C_s`, `C_t`: Source/Target의 공분산 행렬
- `d`: 특징 차원
- `||·||_F`: Frobenius 노름
- 낮을수록 더 잘 정렬됨

**CORAL Similarity:**

```
score = 1 / (1 + CORAL_distance)
```

- [0, 1] 범위, 높을수록 더 잘 정렬됨

**부가 메트릭:**
- **Spectral Divergence**: 고유값 분포 간 대칭 KL 발산 - 공분산 행렬의 고유값 스펙트럼이 얼마나 다른지 측정
- **Mean Distance**: 1차 통계량 (평균 벡터 간 L2 거리)

**분석 모드 (CORALAnalyzer 클래스):**
1. **Cross-modal**: Vision Encoder vs LLM 텍스트 임베딩 (주 분석 모드)
2. **Intra-modal Vision**: Encoder A vs Encoder B (같은 이미지)
3. **Intra-modal Text**: LLM A vs LLM B (같은 텍스트)

**데이터 클래스:**
- `CORALMetrics`: 거리, 유사도, 스펙트럼 발산, 공분산 행렬, 고유값
- `CrossModalResult`: 인코더별 Cross-modal 비교 결과
- `IntraModalResult`: 인코더 간 Intra-modal 비교 결과
- `EASResult`: 통합 EAS 점수 및 구성요소

### 4.3 EAS (Enhanced Alignment Score)

CKA의 한계를 보완하기 위해 개발된 통합 메트릭이다.

**수식:**

```
EAS = 0.3 * CKA + 0.3 * CORAL_similarity + 0.4 * Discriminability
```

**구성요소:**
| 구성요소 | 가중치 | 설명 |
|---------|--------|------|
| **CKA** | 30% | 구조적 유사도 (커널 정렬) |
| **CORAL Similarity** | 30% | 분포 정렬 (공분산 일치도) |
| **Discriminability** | 40% | 특징의 클래스 판별력 유지 여부 |

**Discriminability 계산:**
- 레이블이 있을 때: Fisher Discriminant Ratio를 사용
  - `fisher = between_scatter / within_scatter`
  - `disc = fisher / (fisher + 1)` (0~1 정규화)
- 레이블이 없을 때: 임베딩의 분산 + 코사인 유사도 분산으로 추정

### 4.4 ELAS (Encoder-LLM Alignment Score)

**파일:** `vlm_alignment/analysis/alignment.py` (195줄)

Projector 학습 기반의 정렬 점수이다.

**수식:**

```
ELAS = 0.3*CKA + 0.3*(1-MSE_norm) + 0.2*(1-GenGap) + 0.2*CosSim
```

**구성요소:**
| 구성요소 | 가중치 | 설명 |
|---------|--------|------|
| **CKA** | 30% | 구조적 유사도 |
| **1 - MSE** | 30% | Projector 학습 손실 (낮을수록 좋음) |
| **1 - GenGap** | 20% | Train-Test Loss 차이 (일반화 능력) |
| **Cosine Similarity** | 20% | 투영 후 코사인 유사도 |

**과정:** 2-Layer MLP Projector를 300 에포크 동안 학습 → Train/Test 메트릭 산출 → 가중합

---

## 5. 지원 모델

### 5.1 Vision Encoders

**파일:** `vlm_alignment/models/vision_encoders.py` (241줄)

`VisionEncoderManager` 클래스가 모든 인코더의 로딩과 임베딩 추출을 통합 관리한다.

| 인코더 | Model ID | 출력 차원 | 학습 방식 | 특징 |
|--------|----------|----------|----------|------|
| **CLIP** | `openai/clip-vit-base-patch32` | 768 | Contrastive (텍스트-이미지) | 패치-텍스트 유사도 지원 |
| **SigLIP** | `google/siglip-base-patch16-224` | 768 | Sigmoid Contrastive | CLIP 대비 더 세밀한 보정 |
| **DINOv2** | `facebook/dinov2-base` | 768 | Self-supervised (이미지만) | 텍스트 없이 순수 시각 특징 학습 |
| **InternViT** | `OpenGVLab/InternViT-300M-448px` | 768 | Vision-Language | 고해상도(448px) 지원 |
| **PaliGemma** | `google/paligemma-3b-pt-224` | 2048 | Vision-Language Pretraining | 3B 파라미터, 높은 차원 |

**주요 메서드:**
- `load(encoder_name, with_attention=False)`: 모델 로딩 (선택적 Attention 추출)
- `extract_embeddings(encoder_name, images)`: L2 정규화된 임베딩 추출
- `extract_with_attention(encoder_name, image)`: 임베딩 + Attention 맵 추출
- `extract_multi_encoder(encoder_names, images)`: 다수 인코더 일괄 임베딩 추출
- `unload(encoder_name=None)`: GPU 메모리 해제

**기술적 세부사항:**
- 각 인코더별 전용 프로세서를 사용 (CLIPProcessor, AutoImageProcessor 등)
- CLS 토큰 임베딩을 기본으로 사용, L2 정규화 적용
- PaliGemma의 경우 차원이 다르므로 (2048 vs 768) PCA를 통한 차원 정렬 수행

### 5.2 LLM (Large Language Models)

**파일:** `vlm_alignment/models/llm_loaders.py` (129줄)

`LLMManager` 클래스가 모든 LLM 로딩과 텍스트 임베딩 추출을 통합 관리한다.

| LLM | Model ID | 파라미터 |
|-----|----------|---------|
| **LLaMA** | `huggyllama/llama-7b` | 7B |
| **LLaMA 3.1** | `meta-llama/Llama-3.1-8B` | 8B |
| **Qwen 2.5** | `Qwen/Qwen2.5-7B` | 7B |
| **Gemma 3** | `google/gemma-3-4b-pt` | 4B |
| **InternLM 2.5** | `internlm/internlm2_5-7b` | 7B |

**텍스트 임베딩 추출 과정:**
1. Tokenizer로 텍스트를 토큰화 (max_length=128)
2. 모델의 Hidden State에서 마지막 레이어 출력 추출
3. Attention Mask를 고려한 Mean Pooling 적용
4. L2 정규화

**메모리 최적화:**
- `torch_dtype=torch.float16`으로 메모리 절반 사용
- `device_map="auto"`로 자동 GPU 분배
- `low_cpu_mem_usage=True`로 CPU 메모리 절약

### 5.3 Projectors

**파일:** `vlm_alignment/models/projectors.py` (~200줄)

Vision Encoder 출력을 LLM 텍스트 임베딩 공간으로 투영하는 4가지 아키텍처를 제공한다.

| Projector | 구조 | 영감 |
|-----------|------|------|
| **Linear** | `Linear(d_v, d_t)` | 가장 간단한 선형 투영 |
| **MLP** | `Linear → GELU → Linear` | LLaVA 스타일 |
| **2-Layer MLP** | `Linear → GELU → LayerNorm → Linear` | LayerNorm 추가 안정화 |
| **Cross-Attention** | `MultiHeadAttention(Q=learnable, K=V=vision)` | Q-Former 스타일 |

**학습 설정:**
- 손실 함수: MSE Loss
- 옵티마이저: Adam (lr=1e-3)
- 에포크: 300 (기본값)
- Train/Test 분할: 80/20

---

## 6. 모듈별 상세 설명

### 6.1 데이터 모듈

#### `vlm_alignment/data/dataset.py` - VLM 데이터셋 로더

**데이터 경로 우선순위** (자동 폴백 메커니즘):
1. 함수 인자로 전달된 `data_root`
2. 환경변수 `VLM_DATA_ROOT`
3. `config.yaml`의 `data.root` 설정
4. 내장 `sample_data/` 디렉토리 (최종 폴백)

**실제 데이터셋 구조:**
```
VLM_DATA/
├── label_v2/
│   ├── bichallava_instruct_230k_chart_v2_axolotl.jsonl     # 차트 데이터
│   ├── table-VQA-ko-60k.jsonl                               # 테이블 VQA (한국어)
│   ├── AIHUB_Visualization_v2_axolotl.jsonl                  # 시각화 데이터
│   ├── AIHUB_subjectmaterial_text_modify_v2_axolotl.jsonl    # 텍스트/문서 데이터
│   └── AIHUB_mathproblem_multiple_v2_axolotl.jsonl           # 수학 문제 데이터
└── images/
    ├── bichallava_instruct_230k_chart/
    ├── table-VQA-ko-60k/
    ├── AIHUB_Visualization/
    ├── AIHUB_subjectmaterial_text_modify/
    └── AIHUB_mathproblem_multiple/
```

**5가지 데이터 타입:**
| 타입 | 설명 | 레이블 파일 |
|------|------|------------|
| `chart` | 차트/그래프 이미지와 QA | bichallava_instruct_230k_chart_v2_axolotl.jsonl |
| `table` | 테이블 이미지와 한국어 VQA | table-VQA-ko-60k.jsonl |
| `visualization` | 데이터 시각화 이미지 | AIHUB_Visualization_v2_axolotl.jsonl |
| `text` | 텍스트/문서 이미지 | AIHUB_subjectmaterial_text_modify_v2_axolotl.jsonl |
| `math` | 수학 문제 이미지 | AIHUB_mathproblem_multiple_v2_axolotl.jsonl |

**JSONL 데이터 형식:**
```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "이 차트에서 가장 높은 값은?"},
      {"type": "image", "image": "chart/example.png"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "가장 높은 값은 95.6입니다."}
    ]}
  ]
}
```

**주요 메서드:**
- `load_samples(data_type, n_samples)`: 특정 타입의 샘플 로딩
- `load_mixed(n_per_type, data_types)`: 타입별 균형 혼합 로딩
- `get_images_and_texts(samples)`: PIL 이미지 + 텍스트 문자열 추출

#### `vlm_alignment/data/synthetic.py` - 합성 데이터 생성기

외부 데이터셋 없이 테스트할 수 있도록 합성 이미지를 생성하는 모듈이다.

**DataGenerator 클래스:**
- `generate_bar_chart(categories, values, title)`: Matplotlib 기반 막대 그래프 생성
- `generate_table_image(data, title)`: Pandas DataFrame을 테이블 이미지로 변환
- `generate_document_image(title)`: Introduction/Methods/Results 구조의 문서 이미지 생성
- `generate_simple_images(n, size)`: 기하학적 도형 이미지 생성
- `generate_categorized_images(n_per_category)`: 카테고리별 분류된 도형 이미지

### 6.2 시각화 모듈

#### `vlm_alignment/visualization/plot_style.py` - 통합 스타일

모든 그래프에 일관된 시각적 테마를 적용하는 유틸리티이다.

**제공 기능:**
- `apply_style()`: Matplotlib 글로벌 스타일 적용
- `get_model_color(name)`: 인코더별 고유 색상 반환
  - CLIP: `#4C8BF5` (파랑), SigLIP: `#F5A623` (오렌지), DINOv2: `#0ABAB5` (청록), InternViT: `#E94B3C` (빨강), PaliGemma: `#6C5CE7` (보라)
- `get_data_type_color(dtype)`: 데이터 타입별 색상
- `style_axis(ax)`: 축 스타일 통일

#### `vlm_alignment/visualization/alignment_plots.py` - 정렬 시각화

- `plot_cka_comparison()`: 인코더별 CKA 점수 막대 그래프
- `plot_cka_by_data_type()`: 데이터 타입별 CKA 분석 (chart/table/text 등)
- `plot_projector_comparison()`: Projector 유형별 학습 손실 비교
- `plot_alignment_summary()`: 다중 메트릭 히트맵 요약

#### `vlm_alignment/visualization/coral_plots.py` - CORAL 시각화

본 프로젝트에서 가장 풍부한 시각화를 제공하는 모듈이다.

- `plot_coral_comparison()`: CORAL 거리/유사도 막대 그래프 (좌: 거리, 우: 유사도)
- `plot_cka_vs_coral()`: CKA vs CORAL 산점도 (패러독스 시각화, y=x 대각선 참조)
- `plot_covariance_heatmaps()`: 인코더별 Vision/Text 공분산 행렬 + 차이 히트맵 (3x3 그리드)
- `plot_eigenvalue_spectrum()`: 공분산 행렬의 고유값 분포 비교 (로그 스케일)
- `plot_eas_dashboard()`: EAS 대시보드 (구성요소 막대, CKA vs EAS 산점도, 랭킹)
- `plot_intra_modal_similarity()`: 인코더 간 CORAL 유사도 히트맵 행렬
- `plot_coral_full_dashboard()`: 모든 CORAL 결과를 하나의 통합 대시보드로

#### `vlm_alignment/visualization/attention_maps.py` - Attention 시각화

CLIP/SigLIP의 패치-텍스트 유사도를 히트맵으로 오버레이한다.

- `compute_patch_text_similarity()`: 각 이미지 패치와 텍스트 간 유사도 계산
- `plot_attention_heatmap()`: 히트맵을 원본 이미지 위에 오버레이
- `plot_attention_comparison()`: 여러 인코더의 Attention 맵 나란히 비교

#### `vlm_alignment/visualization/embedding_space.py` - 임베딩 시각화

- `reduce_tsne(embeddings, perplexity=30)`: t-SNE 차원 축소
- `reduce_umap(embeddings, n_neighbors=15)`: UMAP 차원 축소
- `procrustes_align(source, target)`: Procrustes 분석으로 임베딩 공간 정렬
- `plot_multi_encoder_tsne()`: 다중 인코더 t-SNE 산점도
- `plot_similarity_matrices()`: 코사인 유사도 행렬 히트맵

#### `vlm_alignment/visualization/speed_plots.py` - 속도 벤치마크 시각화

- `plot_full_benchmark_dashboard()`: 지연시간/처리량/메모리 통합 대시보드

### 6.3 설정 모듈

**파일:** `vlm_alignment/config.py` (86줄)

`config.yaml`을 로딩하고 환경변수 오버라이드를 지원한다.

**주요 함수:**
- `load_config()`: YAML 설정 파일 로딩
- `get_model_id(category, name)`: 모델 HuggingFace ID 조회
- `get_data_root()`: 데이터 루트 경로 (환경변수 → config → None)
- `get_output_dir()`: 출력 디렉토리 (환경변수 → config → 'outputs')
- `get_device()`: 디바이스 선택 (auto → cuda/cpu 자동 감지)

---

## 7. 실험 코드 상세

### 7.1 Encoder 비교 실험

**파일:** `vlm_alignment/experiments/encoder_comparison.py`

**`run_encoder_comparison()`:**
1. 데이터 로딩 (혼합 타입)
2. 각 인코더로 이미지 임베딩 추출
3. LLM으로 텍스트 임베딩 추출
4. CKA (Linear + RBF), Projection MSE, Cosine Similarity 계산
5. Projector 유형별 평가 (Linear, MLP, 2-Layer MLP)
6. 결과 시각화: CKA 비교 차트, 데이터 타입별 분석, Projector 비교

**`run_multi_llm_comparison()`:**
- 여러 LLM에 대해 위 과정을 반복 수행
- 인코더 x LLM 조합별 CKA 히트맵 생성

```bash
# 실행 예시
python cli.py compare --encoders clip siglip dinov2 --llms llama --n-samples 30
python cli.py multi-llm --encoders clip siglip --llms llama qwen gemma3
```

### 7.2 Deep CORAL 분석 실험

**파일:** `vlm_alignment/experiments/coral_alignment.py` (~145줄)

**`run_coral_analysis()` 파이프라인:**

```
1. 데이터 로딩 (load_mixed)
     ↓
2. Vision Encoder 임베딩 추출 (extract_multi_encoder)
     ↓
3. LLM 텍스트 임베딩 추출 (extract_text_embeddings)
     ↓
4. CORALAnalyzer.full_analysis() 실행
   ├── cross_modal_comparison(): 인코더별 Vision↔Text 비교
   ├── intra_modal_comparison(): 인코더 간 Vision↔Vision 비교
   └── compute_eas(): EAS 통합 점수 산출
     ↓
5. 6가지 시각화 생성
   ├── coral_comparison.png
   ├── cka_vs_coral.png
   ├── covariance_heatmaps.png
   ├── eigenvalue_spectrum.png
   ├── intra_modal_similarity.png
   └── eas_dashboard.png
```

```bash
# 실행 예시
python cli.py coral --encoders clip siglip dinov2 --llms llama --n-samples 30
```

### 7.3 E2E 검증 실험 (CKA-Performance Paradox)

**파일:** `vlm_alignment/experiments/e2e_validation.py` (~183줄)

**목적:** CKA 점수와 실제 하류 태스크 성능 사이의 상관관계를 검증한다.

**파이프라인:**

```
1. 데이터 로딩 + 임베딩 추출
     ↓
2. 각 인코더별:
   a. CKA 계산 (Vision ↔ Text)
   b. SimpleVLM Projector 학습 (300 에포크, Adam, MSE Loss)
   c. 검색(Retrieval) 성능 평가
      - Recall@1: 상위 1개 검색 정확도
      - Recall@5: 상위 5개 검색 정확도
      - MRR: Mean Reciprocal Rank
     ↓
3. Pearson 상관계수 계산 (CKA vs MRR)
     ↓
4. 결과: r = -0.99 (강한 역상관 → CKA-Performance Paradox)
```

**SimpleVLM 클래스:**
- 단순화된 VLM 모델: Vision Embedding → Projector → Text Space
- 검색 평가: 투영된 Vision 임베딩과 Text 임베딩 간 코사인 유사도 기반 랭킹

```bash
# 실행 예시
python cli.py e2e --encoders clip siglip dinov2 --llms llama --epochs 300
```

### 7.4 ELAS 점수 실험

**파일:** `vlm_alignment/experiments/elas_score.py`

모든 인코더 x LLM 조합에 대해 ELAS 점수를 계산하여 최적 페어링을 찾는다.

```bash
# 실행 예시
python cli.py elas --encoders clip siglip dinov2 --llms llama qwen
```

### 7.5 속도 벤치마크 실험

**파일:** `vlm_alignment/experiments/speed_benchmark.py`

**측정 항목:**
- **Latency (ms)**: 배치당 처리 시간
- **Throughput (items/sec)**: 초당 처리 샘플 수
- **Memory (MB)**: 최대 GPU 메모리 사용량
- **Standard Deviation**: 지연시간 변동성

**설정:**
- Warmup: 3회 (측정 제외)
- Benchmark: 10회 (평균 산출)
- Batch sizes: [1, 4, 8, 16, 32]

```bash
# 실행 예시
python cli.py speed --encoders clip siglip dinov2 --batch-sizes 1 4 8 16 32
```

---

## 8. 생성된 분석 결과 이미지

`outputs/coral/` 디렉토리에 Deep CORAL 분석 결과로 6개의 이미지가 생성되어 있다. CLIP, SigLIP, DINOv2 세 인코더를 LLaMA와 비교한 결과이다.

### 8.1 `eas_dashboard.png` - Enhanced Alignment Score 대시보드

3개의 서브플롯으로 구성된 종합 대시보드이다.

**(a) Component Scores (좌측):** 인코더별 CKA, CORAL, Discriminability 점수를 묶은 막대 그래프
- CLIP: CKA ≈ 0.82, CORAL ≈ 1.0, Discriminability ≈ 0.97
- SigLIP: CKA ≈ 0.78, CORAL ≈ 1.0, Discriminability ≈ 0.91
- DINOv2: CKA ≈ 0.89, CORAL ≈ 1.0, Discriminability ≈ 0.99

**(b) CKA vs EAS (중앙):** 산점도로 CKA와 EAS의 관계 시각화
- DINOv2: 가장 높은 CKA(0.89)와 EAS(0.96)
- CLIP: CKA 0.82, EAS 0.93
- SigLIP: 가장 낮은 CKA(0.78), EAS 0.90
- 대각선(y=x) 위에 점들이 위치 → EAS가 CKA보다 일관되게 높음

**(c) Ranking (우측):** 메트릭별 랭킹 비교 텍스트
- CKA 기준: DINOv2(0.891) > CLIP(0.821) > SigLIP(0.776)
- Discriminability 기준: DINOv2(0.987) > CLIP(0.973) > SigLIP(0.914)
- EAS 기준: DINOv2(0.962) > CLIP(0.933) > SigLIP(0.888)
- 하단에 EAS 공식: `EAS = 0.3*CKA + 0.3*CORAL + 0.4*Disc`

### 8.2 `cka_vs_coral.png` - CKA vs CORAL 메트릭 비교

CKA Score (x축) vs CORAL Similarity (y축) 산점도이다.

- 모든 인코더가 CORAL Similarity ≈ 1.0으로 매우 높음
- CKA는 인코더 간 차이가 있음: SigLIP(0.78) < CLIP(0.82) < DINOv2(0.89)
- y=x 대각선 참조선: 모든 점이 대각선 **위**에 위치
- **해석:** CORAL은 CKA보다 더 관대한 유사도를 보여줌. 두 메트릭이 서로 다른 측면을 포착하고 있음을 시사

### 8.3 `coral_comparison.png` - CORAL 거리 및 유사도 비교

좌우 두 개의 막대 그래프이다.

**CORAL Distance (좌측, 낮을수록 좋음):**
- CLIP: 0.0001 (최저 → 가장 잘 정렬)
- SigLIP: 0.0001 (CLIP과 거의 동일)
- DINOv2: 0.0003 (상대적으로 높음)

**CORAL Similarity (우측, 높을수록 좋음):**
- CLIP: 0.9999
- SigLIP: 0.9999
- DINOv2: 0.9997

**해석:** CORAL 관점에서 CLIP과 SigLIP은 거의 동일한 수준으로 LLM과 정렬되어 있으며, DINOv2는 약간 낮다. 이는 CKA에서 DINOv2가 가장 높았던 것과 대조적이다 (CKA-Performance Paradox의 일부).

### 8.4 `covariance_heatmaps.png` - 공분산 행렬 비교

3x3 그리드 (인코더 3개 x 측면 3개)의 히트맵이다.

**각 행:** CLIP, SigLIP, DINOv2

**각 열:**
1. **Vision Cov**: 비전 임베딩의 공분산 행렬
2. **Text Cov**: 텍스트 임베딩의 공분산 행렬
3. **Difference**: Vision - Text 공분산 차이

**관찰:**
- Vision 공분산: 각 인코더마다 다른 패턴. CLIP과 SigLIP은 유사하고, DINOv2는 다른 구조
- Text 공분산: 모든 인코더에 대해 동일 (같은 LLaMA 텍스트 임베딩 사용)
- Difference: DINOv2의 차이가 가장 큼 → self-supervised 학습 특성상 텍스트 공간과의 분포 차이가 더 크다

### 8.5 `eigenvalue_spectrum.png` - 고유값 스펙트럼 비교

3개의 로그 스케일 서브플롯으로 각 인코더의 Vision/Text 고유값 분포를 비교한다.

**CLIP (좌측):**
- Spectral Divergence = 0.269
- Vision 고유값이 Text보다 더 가파르게 감소
- 주요 성분 2~3개에 에너지 집중

**SigLIP (중앙):**
- Spectral Divergence = 0.339 (가장 높음)
- Vision과 Text 간 고유값 분포 차이가 가장 큼
- Vision 고유값의 감소가 더 급격함

**DINOv2 (우측):**
- Spectral Divergence = 0.122 (가장 낮음)
- Vision과 Text 고유값 분포가 가장 유사
- 에너지가 더 고르게 분산됨

**해석:** Spectral Divergence가 낮은 DINOv2가 고유값 분포 관점에서는 Text와 가장 유사한 구조를 가짐. 이는 DINOv2의 높은 CKA와도 일치한다.

### 8.6 `intra_modal_similarity.png` - 인코더 간 유사도 행렬

3x3 히트맵으로 CLIP, DINOv2, SigLIP 간의 Intra-Modal CORAL Similarity를 보여준다.

**결과:**
- 모든 셀이 1.000으로 표시됨
- CLIP ↔ DINOv2: 1.000
- CLIP ↔ SigLIP: 1.000
- DINOv2 ↔ SigLIP: 1.000

**해석:** 샘플 데이터(9개 샘플)의 규모가 작아 공분산 추정의 해상도가 부족하여 모든 인코더 간 유사도가 1.0으로 수렴한 것으로 보인다. 실제 대규모 데이터셋에서는 더 변별력 있는 결과가 기대된다.

---

## 9. 샘플 데이터

`sample_data/` 디렉토리에 외부 데이터셋 없이 모든 기능을 테스트할 수 있는 내장 합성 데이터가 포함되어 있다.

### 9.1 이미지 파일 (9개)

#### 차트 이미지 (`sample_data/images/chart/`)
- `chart_01.png`: "Sample Chart 1" - 4개 항목(Item0~3)의 막대 그래프. 값: 43.7, 95.6, 75.9, 63.9. 각 막대가 다른 색상(민트, 파랑, 회색, 노랑)
- `chart_02.png`: "Sample Chart 2" - 유사한 구조의 다른 값을 가진 막대 그래프
- `chart_03.png`: "Sample Chart 3" - 유사한 구조의 다른 값을 가진 막대 그래프

#### 테이블 이미지 (`sample_data/images/table/`)
- `table_01.png`: "Sample Table 1" - Product/Price/Qty 3열 테이블. Apple($1.00/100), Banana($0.50/150), Orange($0.75/80), Grape($2.00/60). 파란 헤더
- `table_02.png`: "Sample Table 2" - 유사한 구조의 다른 데이터
- `table_03.png`: "Sample Table 3" - 유사한 구조의 다른 데이터

#### 문서 이미지 (`sample_data/images/document/`)
- `document_01.png`: "Sample Document 1" - Introduction/Methods/Results 3개 섹션. 각 섹션에 Lorem ipsum 텍스트. 진한 파란 제목, 검은 본문
- `document_02.png`: "Sample Document 2" - 유사한 구조
- `document_03.png`: "Sample Document 3" - 유사한 구조

### 9.2 레이블 파일 (`sample_data/labels.jsonl`)

9개의 JSONL 엔트리로, 각 이미지에 대한 질문-답변 쌍이 포함되어 있다.

**형식:**
```json
{"messages": [
  {"role": "user", "content": [
    {"type": "text", "text": "Describe the chart"},
    {"type": "image", "image": "chart/chart_01.png"}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "A bar chart showing 4 items with values..."}
  ]}
]}
```

---

## 10. 실행 방법

### 10.1 설치

```bash
git clone https://github.com/yujuyeon0511/vlm-encoder-alignment.git
cd vlm-encoder-alignment
pip install -r requirements.txt
```

### 10.2 CLI 명령어 전체 목록

```bash
python cli.py <command> [options]
```

| 명령어 | 설명 | 주요 출력 |
|--------|------|----------|
| `compare` | 인코더 비교 (CKA, Projector) | CKA 차트, 타입별 분석 |
| `multi-llm` | 다중 LLM 인코더 비교 | 인코더 x LLM 히트맵 |
| `coral` | Deep CORAL 분석 | 6개 대시보드 이미지 |
| `speed` | 추론 속도 벤치마크 | 속도 대시보드 |
| `attention` | Attention 맵 시각화 | 히트맵 오버레이 이미지 |
| `embedding` | 임베딩 공간 시각화 | t-SNE/UMAP 산점도 |
| `elas` | ELAS 점수 계산 | ELAS 매트릭스 |
| `e2e` | E2E 검증 (CKA vs 성능) | CKA-성능 상관 분석 |
| `all` | compare + speed + embedding 통합 실행 | 여러 시각화 |

### 10.3 공통 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--encoders` | clip siglip dinov2 | 사용할 Vision Encoder 목록 |
| `--llms` | llama | 대상 LLM 목록 |
| `--n-samples` | 30 | 데이터 타입당 샘플 수 |
| `--data-root` | (자동) | 실제 데이터셋 경로 |
| `--output-dir` | outputs | 결과 저장 디렉토리 |
| `--device` | auto | cuda 또는 cpu |

### 10.4 실행 예시

```bash
# 1. 샘플 데이터로 빠른 테스트 (외부 데이터 불필요)
python cli.py compare --n-samples 3

# 2. Deep CORAL 전체 분석
python cli.py coral --encoders clip siglip dinov2 --llms llama --n-samples 30

# 3. 다중 LLM 비교
python cli.py multi-llm --encoders clip siglip dinov2 --llms llama qwen gemma3

# 4. 속도 벤치마크
python cli.py speed --encoders clip siglip dinov2 --batch-sizes 1 4 8 16 32

# 5. Attention 맵 (특정 이미지)
python cli.py attention --image path/to/image.png --text "What is shown in this chart?"

# 6. UMAP 임베딩 시각화
python cli.py embedding --method umap --n-samples 50

# 7. E2E 검증 (CKA-Performance Paradox)
python cli.py e2e --encoders clip siglip dinov2 --llms llama --epochs 300

# 8. 실제 데이터셋 사용
export VLM_DATA_ROOT=/path/to/vlm_20251118
python cli.py coral --n-samples 100

# 9. 전체 실험 한번에 실행
python cli.py all --encoders clip siglip dinov2
```

### 10.5 Gradio Web UI

```bash
python app.py
# 브라우저에서 http://localhost:7860 접속
```

**6개 탭:**

1. **Inference Speed**: 인코더 선택 후 벤치마크 실행, 대시보드 확인
2. **Attention Maps**: 이미지 업로드 + 텍스트 입력 → Attention 히트맵
3. **Alignment Analysis**: CKA/MSE/Cosine 메트릭 테이블
4. **Embedding Space**: t-SNE/UMAP 인터랙티브 산점도
5. **CORAL Analysis**: Deep CORAL + EAS 대시보드
6. **E2E Validation**: CKA-Performance Paradox 검증

---

## 11. 주요 연구 발견

### 11.1 CKA-Performance Paradox

| 인코더 | CKA Score | Retrieval MRR | 관계 |
|--------|-----------|---------------|------|
| DINOv2 | 0.891 (최고) | 낮음 | 역전 |
| CLIP | 0.821 (중간) | 중간 | - |
| SigLIP | 0.776 (최저) | 높음 | 역전 |

**Pearson 상관계수: r = -0.99** (거의 완벽한 역상관)

**의미:** CKA 점수가 높다고 해서 실제 하류 태스크(검색)에서 좋은 성능을 보장하지 않는다. 오히려 역방향으로 작용할 수 있다.

### 11.2 SigLIP의 우수성

- CKA 평균에서 CLIP 대비 **+9.4%** 향상 (단, CKA-Performance Paradox를 감안해야 함)
- 차트 데이터에서 특히 강점 (+15% CKA)
- Sigmoid 기반 Contrastive Learning의 더 세밀한 보정이 기여

### 11.3 태스크별 최적 인코더

| 데이터 타입 | 최적 인코더 | 이유 |
|------------|------------|------|
| **차트(Chart)** | SigLIP | Contrastive Learning의 시각적 패턴 인식 |
| **테이블(Table)** | DINOv2 | Self-supervised 학습의 구조적 특징 포착 |
| **문서(Text)** | 상황에 따라 다름 | 텍스트 밀도에 따라 변동 |

### 11.4 LLM별 최적 페어링

| LLM | 최적 인코더 | EAS 점수 |
|-----|------------|---------|
| **LLaMA** | DINOv2 | 0.962 |
| **Qwen** | SigLIP | (실험 필요) |

### 11.5 CORAL이 포착하는 CKA의 사각지대

- CKA는 **구조적 패턴**만 측정 → 분포의 **형태**와 **퍼짐**을 무시
- CORAL은 **공분산 행렬**을 직접 비교 → 특징들이 어떻게 공변하는지 포착
- Eigenvalue Spectrum 분석에서 DINOv2가 가장 낮은 Spectral Divergence(0.122) → 텍스트와 가장 유사한 분산 구조
- 반면 SigLIP은 가장 높은 Spectral Divergence(0.339) → 텍스트와 분산 구조가 가장 다름

---

## 12. 설정 및 의존성

### 12.1 config.yaml

```yaml
data:
  root: null                    # 실제 데이터셋 경로 (null이면 sample_data/ 폴백)
  sample_dir: sample_data       # 내장 샘플 데이터 경로

  datasets:
    chart:
      label_file: label_v2/bichallava_instruct_230k_chart_v2_axolotl.jsonl
      image_dir: images/bichallava_instruct_230k_chart
    table:
      label_file: label_v2/table-VQA-ko-60k.jsonl
      image_dir: images/table-VQA-ko-60k
    visualization:
      label_file: label_v2/AIHUB_Visualization_v2_axolotl.jsonl
      image_dir: images/AIHUB_Visualization
    text:
      label_file: label_v2/AIHUB_subjectmaterial_text_modify_v2_axolotl.jsonl
      image_dir: images/AIHUB_subjectmaterial_text_modify
    math:
      label_file: label_v2/AIHUB_mathproblem_multiple_v2_axolotl.jsonl
      image_dir: images/AIHUB_mathproblem_multiple

models:
  vision_encoders:
    clip: openai/clip-vit-base-patch32
    siglip: google/siglip-base-patch16-224
    dinov2: facebook/dinov2-base
    internvit: OpenGVLab/InternViT-300M-448px
    paligemma: google/paligemma-3b-pt-224

  llms:
    llama: huggyllama/llama-7b
    llama3: meta-llama/Llama-3.1-8B
    qwen: Qwen/Qwen2.5-7B
    gemma3: google/gemma-3-4b-pt
    internlm: internlm/internlm2_5-7b

defaults:
  device: auto     # auto, cuda, cpu
  output_dir: outputs
  n_samples: 30
  seed: 42
```

### 12.2 환경변수 오버라이드

| 환경변수 | 설명 | 예시 |
|---------|------|------|
| `VLM_DATA_ROOT` | 실제 데이터셋 루트 경로 | `/data/vlm_20251118` |
| `VLM_OUTPUT_DIR` | 결과 출력 디렉토리 | `/results/experiment1` |

### 12.3 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `torch>=2.0` | 딥러닝 프레임워크 |
| `transformers>=4.40` | HuggingFace 모델 로딩 |
| `accelerate` | Multi-GPU 지원 |
| `bitsandbytes` | 양자화 지원 |
| `open-clip-torch` | CLIP 모델 |
| `matplotlib` | 시각화 |
| `seaborn` | 통계적 시각화 |
| `scikit-learn` | ML 유틸리티 (CKA, PCA, Ridge, metrics) |
| `umap-learn` | UMAP 차원 축소 |
| `numpy`, `pandas` | 수치/데이터 처리 |
| `Pillow` | 이미지 로딩 |
| `scipy` | 과학 컴퓨팅 (Pearson 상관 등) |
| `pyyaml` | YAML 설정 파싱 |
| `tqdm` | 진행 표시줄 |
| `gradio>=4.0` | 웹 UI 프레임워크 |

### 12.4 하드웨어 권장사항

| 구성요소 | 최소 | 권장 |
|---------|------|------|
| **GPU** | 16GB VRAM | 24GB+ VRAM (V100, A100) |
| **RAM** | 16GB | 32GB+ |
| **디스크** | 10GB (모델 캐시) | 50GB+ (전체 모델 + 데이터) |

**성능 참고:**
- Vision Encoder만 사용 시: ~8GB GPU 메모리
- Vision Encoder + LLM 7B: ~16GB GPU 메모리
- 전체 분석 (30 샘플): V100 기준 약 2~3분
- 배치 처리: 24GB GPU에서 배치 사이즈 16~32 가능

---

## 부록: 코드 설계 원칙

### 중복 제거
- 기존 7개 파일에 산재하던 Vision Encoder 로딩 코드 → `vision_encoders.py` 1개로 통합
- 기존 6개 파일에 산재하던 LLM 로딩 코드 → `llm_loaders.py` 1개로 통합
- 기존 6개 파일에 산재하던 Projector 코드 → `projectors.py` 1개로 통합

### 모듈화
- `models/`: 모든 모델 로딩과 관리
- `analysis/`: 모든 메트릭과 분석 알고리즘
- `visualization/`: 모든 시각화 유틸리티
- `data/`: 모든 데이터 로딩과 생성
- `experiments/`: 순수 오케스트레이션 (분석 로직 없음)

### 설정 기반 구동
- 중앙 `config.yaml`로 모든 모델 ID와 경로 관리
- 환경변수 오버라이드 지원
- 실제 데이터 없을 때 `sample_data/`로 자동 폴백

### 타입 안전성
- 전체 코드에 타입 힌트(Type Hints) 적용
- `@dataclass`로 구조화된 결과 객체 (AlignmentMetrics, EASResult, SpeedResult 등)
- 명확한 함수 시그니처와 docstring
