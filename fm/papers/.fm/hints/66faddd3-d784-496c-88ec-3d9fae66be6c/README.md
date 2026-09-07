# 모델 크기와 outlier(high-norm 토큰) 발생의 관계

**출처**: Darcet et al., *Vision Transformers Need Registers* (arXiv:2309.16588), §2.1 "Outliers appear during the training of large models", Fig. 4c

---

## 1. 카드 한 줄 요약

DINOv2를 Tiny / Small / Base / Large / Huge / giant 6가지 크기로 나눠 학습시킨 뒤 출력 패치 토큰의 norm 분포를 보면,
**가장 큰 세 모델 — ViT-Large, ViT-Huge, ViT-giant — 에서만 high-norm outlier가 나타난다.**
T / S / B에서는 분포가 단일 봉우리(unimodal)로 깔끔하고, L 이상에서 비로소 위쪽에 두 번째 봉우리(bimodal)가 생긴다.
즉 outlier는 "DINOv2라서" 생기는 게 아니라 **모델이 충분히 커졌을 때 창발(emergent)하는 현상**이다.

---

## 2. 그 전에: outlier가 뭐였는지 (Fig. 3)

![Fig. 3 — DINO vs DINOv2의 패치 토큰 norm](fig-2.jpeg)

- 왼쪽 두 히트맵: 같은 강아지 이미지에 대한 패치별 feature norm. DINO(ViT-B/16)는 전체가 고르게 낮은데,
  **DINOv2(ViT-g/14)는 배경 쪽 몇 개 패치만 노란색(norm 100 이상)으로 튄다.** 이게 attention map에 보이는 artifact의 정체다.
- 오른쪽 두 히스토그램: 작은 이미지 데이터셋 전체의 $L_2$ norm 분포. DINO는 봉우리 하나뿐인데,
  DINOv2는 **0~100 근처의 큰 봉우리 + 400~500 부근의 두 번째 봉우리**로 명확히 bimodal이다.
- 그래서 논문은 **norm > 150을 "high-norm = outlier"의 기준선**으로 삼는다(모델마다 컷오프는 달라질 수 있음).
  DINOv2 ViT-g에서 이 비율은 **2.37%** 정도.

이 카드가 묻는 것은 "그럼 이 두 번째 봉우리가 **언제** 생기냐"이고, 그 답 중 하나가 모델 크기 축이다.

---

## 3. 핵심 그림: Fig. 4 (세 개의 축)

![Fig. 4 — layer / iteration / model size 축의 norm 분포](fig-1.jpeg)

세 패널 모두 **세로축 = 토큰 norm(로그 스케일, 3 / 30 / 300 눈금)**, **색 = 해당 norm을 갖는 토큰의 비율(proportion, 로그 컬러바)** 인 밀도 히트맵이다.
"위쪽에 별도의 밝은 띠가 하나 더 생기는가"가 곧 "outlier가 있는가"이다.

| 패널 | 가로축 | 관찰 | 결론 |
|---|---|---|---|
| (a) | layer (1~40) | 초반 층에서는 norm이 낮게 뭉쳐 있다가, **layer 15 근처부터 위쪽으로 두 번째 띠가 갈라져 나온다.** 이후 300 부근까지 올라가 끝까지 유지 | outlier는 **모델 중반 층**(40층 ViT-g 기준 15층 이후)에서 분화 |
| (b) | pretrain iter (~112k / 312k / 512k) | 학습 초반에는 하나의 띠. **112k~ 이후(전체 학습의 약 1/3 지점)부터 위쪽 띠가 분리**되어 두 갈래로 벌어짐 | outlier는 **학습 1/3 이후**에 등장 |
| **(c)** | **arch (T, S, B, L, H, g)** | **T, S, B는 하나의 띠만 있고 크기에 따라 조금씩 위로 올라갈 뿐. L에서 처음으로 위쪽에 별도의 밝은 띠가 생기고, H, g에서 그 띠가 더 뚜렷/더 높아진다** | **outlier는 ViT-L 이상에서만 발생** ← 이 카드 |

> 논문 캡션 원문: *"The outliers appear around the middle of the model during training; they appear with models larger than and including ViT-Large."*
> 본문: *"only the three largest models exhibit outliers (Fig. 4c)."*

### (c) 패널을 볼 때 헷갈리기 쉬운 점

- T→g로 가면서 **띠 전체가 조금씩 위로 올라가는 것**(모델이 크면 feature norm 자체가 커짐)과,
  **L부터 위에 띠가 하나 더 갈라지는 것**은 다른 이야기다. 카드가 말하는 outlier는 **후자**(bimodality)다.
- 즉 "큰 모델은 norm이 크다"가 아니라 **"큰 모델에서만 norm 분포가 두 갈래로 쪼개진다"**가 포인트.

---

## 4. 세 조건을 한 문장으로 묶기

outlier가 관찰되려면 다음이 **동시에** 만족돼야 한다:

1. **모델 크기**: ViT-Large 이상 (L / H / g) — 6개 중 가장 큰 3개
2. **학습 진행도**: 전체 pretraining의 약 1/3 이후
3. **깊이**: 40층 ViT-g 기준 layer 15 이후 (모델 중반)

암기 팁: **"큰 모델이, 충분히 오래 학습된 뒤, 중간 층부터"** — 크기 · 시간 · 깊이 3축 모두 "어느 정도 지나야" 나타나는 임계(threshold)형 현상.

---

## 5. 왜 이게 중요한가 (해석)

논문의 해석은 이렇다:

- 충분히 큰 모델은 학습 중 **"정보가 중복된 패치(주변 이웃과 cosine similarity가 높은 배경 패치)를 재활용해 전역 정보를 담는 임시 저장소(register)로 쓰는 내부 메커니즘"** 을 스스로 획득한다.
- 이 재활용된 토큰은 **자기 위치·픽셀 정보를 잃고(local info ↓)**, 대신 **이미지 전역 정보를 담아 linear probing 정확도가 오히려 높다(global info ↑)**.
- 이런 "여유 용량을 활용한 저장소 만들기"는 **모델에 충분한 capacity가 있을 때만 발현되는 창발적 최적화**이므로, 작은 T/S/B에서는 나타나지 않는다.
- 그래서 해법이 **register 토큰을 명시적으로 추가**하는 것이다. 실제 ablation은 DINOv2 **ViT-L**(= outlier가 나타나는 최소 크기)로 수행했고, register 1개만 넣어도 artifact가 사라지며 4개를 기본값으로 채택했다.

여담: 큰 모델일수록 잘 나타난다는 점은 LLM의 **massive activations / attention sink** 현상과도 같은 결로 자주 비교된다.

---

## 6. 참고: ViT 크기별 스펙

일반적인 ViT 스케일 (patch 14 기준, DINOv2 계열 값 병기). 파라미터 수는 backbone 기준 근사치.

| Arch | 표기 | Embed dim | Heads | Blocks(depth) | 대략 파라미터 | Fig. 4c에서 outlier |
|---|---|---|---|---|---|---|
| ViT-Tiny | T | 192 | 3 | 12 | ~5M | ✕ |
| ViT-Small | S | 384 | 6 | 12 | ~21M | ✕ |
| ViT-Base | B | 768 | 12 | 12 | ~86M | ✕ |
| **ViT-Large** | **L** | **1024** | **16** | **24** | **~300M** | **○ (여기서부터)** |
| ViT-Huge | H | 1280 | 16 | 32 | ~630M | ○ |
| ViT-giant | g | **1536** | **24** | **40** | **~1.1B** | ○ |

- DINOv2의 ViT-g는 원래 Zhai et al.(2022)의 ViT-G(dim 1408 / 16 heads)와 달리
  **FlashAttention 효율(head dim 64, 전체 dim 256의 배수)을 위해 dim 1536 / 24 heads / 40 blocks**로 바꿨고, 총 **1.1B 파라미터**다.
  Fig. 4a의 가로축이 40층인 이유가 이것.
- DINOv2 논문 Table 17은 S/B/L(distilled, MLP FFN)과 L/g(from scratch, SwiGLU FFN)만 정리하고 있으며,
  T와 H는 registers 논문의 이 크기 스윕에서 추가로 등장한다.
- DINOv2에서 S/B/L(distilled)은 ViT-g로부터 **distillation**으로 만든다 — 그럼에도 L급에서는 outlier가 관찰된다는 점이 시사적이다.

---

## 7. 자주 틀리는 포인트

- ✕ "ViT-Base부터 나타난다" → ○ **ViT-Large부터**. 6개 중 **위에서 세 개(L, H, g)**.
- ✕ "모델이 클수록 outlier가 점점 많아지는 연속적 경향" → ○ 논문 그림은 **L 이전에는 사실상 없다가 L에서 나타나는 임계적 양상**에 가깝다.
- ✕ "outlier는 DINOv2만의 문제" → ○ 크기 스윕은 DINOv2로 했지만, 논문은 **OpenCLIP, DeiT-III**에서도 동일한 high-norm outlier를 확인했다(Fig. 7). 학습 알고리즘과 무관하게 큰 ViT에서 공통으로 나타나는 현상.
- ✕ "outlier = 쓸모없는 노이즈" → ○ 오히려 **전역 정보를 담고 있어** classification linear probing 성능은 일반 패치보다 높다. 문제는 dense task용 local feature가 망가진다는 것.
