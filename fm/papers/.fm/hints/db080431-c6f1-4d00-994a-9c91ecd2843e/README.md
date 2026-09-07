# LOST 실험에서 모델별로 어떤 feature를 사용했는가?

**답**: DeiT-III와 OpenCLIP은 **value**를, DINOv2는 **key**를 사용했다. 출력 feature의 conditioning(스케일·중심 위치)이 모델마다 다를 수 있으므로 feature의 **gram matrix에 bias를 수동으로 더했다**.

논문 원문(Sec. 3.3):

> We use values for DeiT and OpenCLIP, and for DINOv2, we use keys. Because the output features may have different conditioning, we manually add a bias to the gram matrix of features.

이 한 줄이 왜 필요한지 이해하려면 LOST가 무엇을 계산하는지부터 봐야 한다.

---

## 1. 먼저: LOST 알고리즘이란 (Siméoni et al., BMVC 2021)

LOST("Localizing Objects with Self-supervised Transformers and no labels")는 **레이블 없이 이미지 하나에서 물체 박스 하나를 뽑아내는** unsupervised object discovery 방법이다. 학습이 전혀 없고, frozen ViT의 patch feature만 가지고 순수한 선형대수 연산으로 마스크를 만든다.

### 1-1. 입력: patch feature와 gram matrix

ViT 마지막 self-attention layer에서 전체 head를 concat한 patch feature를 뽑는다. CLS 토큰 행은 버린다.

- $F \in \mathbb{R}^{N \times d}$ ($N$ = patch 개수, ViT-S/16이면 $d = 384$)
- **gram matrix** $A = F F^\top$, 즉 $A_{pq} = f_p^\top f_q$ — 모든 patch 쌍의 유사도 행렬

중요: LOST 원 구현은 feature를 **L2 정규화하지 않는다**. cosine similarity가 아니라 **정규화 안 된 내적(dot product)** 이다. 이 점이 뒤에 나올 bias 이야기의 핵심 전제다.

### 1-2. Seed 선택 — "가장 적게 상관된 patch"

내적의 **부호**로 이진화한다:

$$a_{pq} = \begin{cases} 1 & \text{if } f_p^\top f_q \ge 0 \\ 0 & \text{otherwise}\end{cases}$$

각 patch의 degree(자기와 양의 상관을 갖는 patch 수)를 세고, **degree가 최소인 patch를 seed로 고른다**:

$$d_p = \sum_q a_{pq}, \qquad p^* = \arg\min_p d_p$$

근거가 되는 두 가정:

- (a) 물체 내부 patch끼리는 서로 양의 상관, 물체 patch와 배경 patch는 음의 상관을 갖는다.
- (b) 개별 물체는 배경보다 **면적이 작다**.

따라서 "다른 patch와 가장 적게 양의 상관을 갖는 patch"는 배경이 아니라 물체에 속할 확률이 높다. 이게 LOST의 inverse-degree seed selection이다.

### 1-3. Seed expansion — seed와 양의 상관인 patch로 확장

seed 하나만 쓰면 물체의 가장 판별적인 일부(예: 새의 머리)만 잡히므로, 확장을 한다.

1. degree가 낮은 순으로 상위 $k$개 patch 집합 $D_k$를 만든다 (기본 $k = 100$).
2. 그중 seed와 양의 상관인 것만 남긴다: $S = \{q \in D_k \mid f_q^\top f_{p^*} \ge 0\}$.
3. 모든 patch $q$에 대해 **seed expansion score**를 계산한다 — 이건 이진화 전의 raw 내적 합이다:

   $$f(q) = \sum_{s \in S} f_q^\top f_s$$

4. 마스크: $m_q = 1 \iff f(q) \ge 0$. 즉 "$S$의 patch들과 평균적으로 양의 상관이면 물체".

### 1-4. Box 추출

마스크 $m$에서 **seed가 포함된 connected component**를 고르고, 그 component의 bounding box를 최종 검출 결과로 낸다. 평가는 **corloc** — 예측 박스가 GT 박스 중 하나와 IoU ≥ 0.5면 정답으로 치는 이미지 비율.

원 논문 성능: DINO ViT-S/16 keys 기준 VOC07 **61.9** corloc. 반면 supervised DeiT-S로 바꾸면 16.9로 붕괴 — LOST의 성능이 backbone feature의 성질에 극도로 민감하다는 증거다.

> **여기서 반드시 기억할 점**: LOST의 판정(1-2의 degree, 1-3의 마스크)은 전부 **내적의 부호 = 0을 임계값으로 하는 비교**로 이루어져 있다.

---

## 2. 왜 모델마다 다른 feature(key vs value)를 쓰는가

LOST 원 논문의 ablation(DINO ViT-S/16, VOC07 corloc):

| 사용 feature | corloc |
|---|---|
| queries | 30.8 |
| values | 50.5 |
| **keys** | **61.9** |

DINO에서는 key가 압도적이라 LOST의 기본값이 key다. Registers 논문도 DINOv2에 대해서는 이 기본 설정 그대로 **key**를 썼다.

하지만 DeiT-III와 OpenCLIP에서는 사정이 다르다. 이 모델들의 key/query 공간에는 **high-norm outlier(artifact) patch**가 그대로 살아 있어서, gram matrix가 오염되고 seed 선택이 배경의 outlier로 끌려간다. 반면 **value projection을 거치면 그 outlier가 상당 부분 걸러진다**. 그래서 두 모델은 value를 쓰는 편이 실용적이었다.

아래 Fig. 13은 LOST 중간 계산(LOST score / seed와의 dot product / seed expansion)을 모델별·register 유무별로 보여준다. DeiT-III w/o REG 열의 밝은 점 몇 개가 전형적인 outlier patch이고, register를 넣으면 그것이 사라지며 중간 맵이 극적으로 깨끗해진다.

![LOST 중간 계산 결과: 모델별·register 유무별 비교 (Fig. 13)](fig-1.jpeg)

Table 3의 결과(corloc):

| | VOC 2007 | VOC 2012 | COCO 20k |
|---|---|---|---|
| DeiT-III | 11.7 | 13.1 | 10.7 |
| DeiT-III+reg | 27.1 | 32.7 | 25.1 |
| OpenCLIP | 38.8 | 44.3 | 31.0 |
| OpenCLIP+reg | 37.1 | 42.0 | 27.9 |
| DINOv2 | 35.3 | 40.2 | 26.9 |
| DINOv2+reg | **55.4** | **60.0** | **42.0** |

DINOv2는 register로 +20.1 corloc, DeiT-III도 크게 개선. 그런데 DINOv2+reg(55.4)조차 원조 DINO의 61.9에는 못 미친다. **OpenCLIP만 register를 넣었더니 오히려 약간 나빠졌다** — 이 예외가 부록 C의 주제다.

---

## 3. gram matrix에 bias를 더하는 이유

LOST의 모든 판정이 **"내적 ≥ 0인가"** 라는 부호 비교라는 점을 다시 떠올리자. 이 임계값 0이 의미를 가지려면, feature가 **원점 근처에 중심화(centered)** 되어 있어야 한다.

문제는 모델마다, 그리고 key/value 중 무엇을 쓰냐에 따라 feature 분포의 **conditioning**이 다르다는 것이다. 특히 feature 전체에 공통 평균 성분(DC offset) $\mu$가 크게 실려 있으면:

$$f_p^\top f_q = (\mu + \tilde f_p)^\top(\mu + \tilde f_q) = \underbrace{\|\mu\|^2}_{\text{항상 큰 양수}} + \mu^\top(\tilde f_p + \tilde f_q) + \tilde f_p^\top \tilde f_q$$

$\|\mu\|^2$ 항이 지배하면 **거의 모든 쌍의 내적이 양수**가 되어버린다. 그러면

- degree $d_p \approx N$ 으로 모든 patch가 똑같아져 seed 선택이 무의미해지고,
- seed expansion score $f(q)$도 전부 양수가 되어 마스크가 이미지 전체를 덮는다.

반대로 offset이 음의 방향이면 마스크가 텅 비어버린다. 즉 **부호 기준이 무너진다.**

해결책이 "gram matrix에 bias를 수동으로 더한다"이다. $A' = FF^\top + b$ 로 두면 판정 조건이

$$f_p^\top f_q \ge 0 \quad\longrightarrow\quad f_p^\top f_q \ge -b$$

로 바뀐다. 즉 **0이라는 고정 임계값을, 그 feature 공간에 맞는 임계값으로 옮기는 재보정(recalibration)** 이다. feature를 중심화하는 것과 사실상 같은 효과이며, 스칼라 하나만 조정하면 되니 "manually add a bias"라는 표현이 붙었다.

**요점**: bias는 성능을 짜내는 트릭이 아니라, LOST를 DINO 아닌 backbone에 옮겨 쓰기 위한 **최소한의 전제 조건 맞추기**다. 이게 없으면 DeiT/OpenCLIP/DINOv2에서 LOST가 아예 동작하지 않는다(마스크가 전부 켜지거나 전부 꺼진다).

---

## 4. 부록 C와의 연결 — OpenCLIP은 왜 register 효과가 없었나

Table 3에서 OpenCLIP만 register를 넣으니 살짝 나빠졌다. 부록 C의 설명은 **"feature 선택 때문"** 이다.

- OpenCLIP은 출력 feature에 분명히 high-norm outlier patch가 존재한다(Fig. 7).
- 그런데 LOST 실험에서 OpenCLIP은 feature를 직접 쓰지 않고 **attention 계산의 value를 쓴다.**
- Fig. 14를 보면: register 없는 모델에서 **keys와 queries에는 배경에 점 형태의 artifact가 뚜렷**하지만, **values로 계산한 seed expansion score는 register 유무와 무관하게 이미 깨끗하다.**

![OpenCLIP의 keys / queries / values별 seed expansion score (Fig. 14)](fig-2.jpeg)

> 논문의 관찰: "the value projection filters out the outliers even without registers. This means that the outliers appear to live in the null space of the value projection layer."

즉 **outlier가 value projection layer의 null space에 살고 있어서, $W_V$를 통과하는 순간 지워진다.** OpenCLIP에는 register가 없어도 이미 "공짜 artifact 필터"가 걸려 있었던 셈이다. 그러니 register를 추가해 artifact를 실제로 제거해도(Fig. 20에서 확인됨) LOST score에는 변화가 거의 없고, 남는 것은 미세한 노이즈뿐 — 그래서 수치가 소폭 하락한 것.

### 인과의 사슬로 정리하면

1. LOST는 gram matrix의 **부호**에 전적으로 의존한다 → outlier가 있으면 seed 선택이 망가진다.
2. 그래서 각 모델에서 **outlier가 가장 덜 오염시킨 feature**를 골라야 한다 → DINOv2는 key(전통적 최적), DeiT/OpenCLIP은 value(value projection이 outlier를 걸러줌).
3. 그러나 feature마다 conditioning이 달라 임계값 0이 안 맞는다 → **gram matrix에 bias를 더해 임계값을 재보정**한다.
4. 그 결과, OpenCLIP은 value 덕분에 **register 도입 전부터 이미 artifact 면역** 상태였고, 그래서 유일하게 register의 이득이 나타나지 않았다.

---

## 5. 한 줄 암기 카드

- **DeiT-III, OpenCLIP → values / DINOv2 → keys**
- **gram matrix + manual bias** = "$\ge 0$" 판정 기준을 feature 공간에 맞게 옮기는 재보정
- OpenCLIP의 예외 = value projection의 **null space**가 outlier를 이미 삼켜버림
