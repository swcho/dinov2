# 비디오 행동 인식 평가에서 사용한 특징 추출 방식

> **Q.** 비디오 행동 인식 평가에서 사용한 특징 추출 방식은?
> **A.** 비디오에서 균등 간격 8프레임을 뽑아, UCF-101과 Kinetics-400은 특징 평균 위에 linear classifier를 학습한다. SSv2는 시간 정보를 더 유지하기 위해 평균 대신 concatenation을 쓴다.

출처: DINOv2 논문(arXiv 2304.07193v2) §7.2 "Additional Image and Video classification Benchmarks", Table 7.
(이 카드 내용을 다루는 논문 그림은 없다. Table 7은 수치 표라서 아래에 그대로 옮겼다.)

---

## 1. 한 문장 요약

DINOv2는 **이미지로만** 학습한 모델이다. 그런데도 비디오 행동 인식에서 얼마나 쓸 만한지 보기 위해, 비디오를 "프레임 몇 장의 묶음"으로 취급해 **프레임별 이미지 특징을 뽑고 → 하나의 벡터로 합친 뒤 → linear classifier만** 학습했다. 합치는 방식이 데이터셋에 따라 다르다.

| 단계 | UCF-101 / Kinetics-400 | Something-Something v2 (SSv2) |
|---|---|---|
| ① 프레임 샘플링 | 비디오 전체에서 **균등 간격(evenly spaced) 8프레임** | 동일 |
| ② 프레임별 특징 | frozen DINOv2 backbone으로 프레임마다 D차원 벡터 | 동일 |
| ③ 8개 벡터 합치기 | **평균(average)** → D차원 | **연결(concatenation)** → 8×D차원 |
| ④ 분류기 | linear classifier | linear classifier |

백본은 얼려 두고(frozen), 시간축을 다루는 모듈(3D conv, temporal attention 등)은 **전혀 없다**. 순수하게 "이미지 특징이 얼마나 좋은가"를 재는 평가다.

---

## 2. 왜 프레임을 "균등 간격 8장"만 뽑나

- 비디오는 초당 25~30프레임이라 10초짜리 클립도 250~300장이다. 전부 넣으면 비용이 크고, 인접 프레임은 거의 같은 그림이라 정보 중복이 크다.
- **균등 간격(uniform / evenly spaced) 샘플링**은 클립 길이를 8등분해서 각 구간에서 한 장씩 뽑는 방식이다. 시작~끝을 고르게 덮어서 행동의 전개(시작 자세 → 중간 → 끝 자세)를 놓치지 않는다.
- 8이라는 수는 TSN(Temporal Segment Networks) 계열 비디오 연구에서 관행적으로 쓰는 값이다. DINOv2는 자체 실험 없이 이 관행을 그대로 따랐다.

---

## 3. 세 데이터셋의 성격 차이 — 이것이 합치는 방식을 결정한다

### 3-1. UCF-101, Kinetics-400: "외형(appearance)만 봐도 맞힐 수 있는" 태스크

| | UCF-101 | Kinetics-400 |
|---|---|---|
| 출처 | YouTube (Soomro et al., 2012) | YouTube (Kay et al., 2017) |
| 규모 | 13k 클립, 101 클래스 | ~300k 클립, 400 클래스 |
| 클래스 예 | Basketball, Playing Guitar, Horse Riding, Skiing | playing violin, surfing water, making pizza, riding elephant |

이 클래스들은 **한 프레임의 장면·객체·배경**으로 상당 부분 판별된다. 기타가 보이면 "기타 연주", 눈 덮인 경사면이면 "스키", 말이 보이면 "승마". 프레임 순서를 뒤섞거나 한 장만 봐도 정답이 크게 바뀌지 않는다. 이런 태스크를 **appearance-biased** 혹은 **scene/object-biased** 라고 부른다.

→ 따라서 8장의 특징을 **평균**해도 "이 비디오에 무엇이 나오는가"라는 핵심 정보는 살아남는다. 평균은 오히려 노이즈(흔들린 프레임, 컷 전환)를 완화하는 장점도 있다.

### 3-2. SSv2 (Something-Something v2): "시간 순서가 정답을 결정하는" 태스크

- 출처: Goyal et al., 2017. ~220k 클립, 174 클래스. 크라우드워커가 지시문(템플릿)에 맞춰 직접 손으로 물건을 조작한 짧은 영상.
- 클래스가 **"Something"이라는 자리 표시자가 들어간 동사 템플릿**이다:
  - *Putting [something] on a surface* vs *Taking [something] from a surface*
  - *Moving [something] from left to right* vs *Moving [something] from right to left*
  - *Pushing [something] so that it falls off the table* vs *Pushing [something] so that it almost falls off*
  - *Pretending to pick [something] up* vs *Picking [something] up*
- **물체가 무엇인지는 의도적으로 무의미하다.** 컵이든 펜이든 상관없고, 오직 "손과 물체가 시간에 따라 어떻게 움직였나"가 정답이다.
- "왼→오"와 "오→왼"은 **같은 프레임 집합을 순서만 뒤집은 것**이다. 한 장의 스틸컷, 혹은 순서를 잊은 프레임 묶음으로는 원리적으로 구분이 불가능하다.

→ 이런 태스크를 **motion-biased / temporal reasoning** 벤치마크라 부르며, "이미지 모델이 비디오를 진짜 이해하는가"를 가려내는 시험지로 자주 쓰인다. 논문도 "SSv2 requires a much richer understanding of the video frames"라고 적었다.

---

## 4. 왜 평균은 순서를 잃고, concatenation은 유지하는가

프레임별 특징을 $f_1, f_2, \dots, f_8 \in \mathbb{R}^D$ 라 하자.

### 평균 (average pooling)

$$\bar f = \frac{1}{8}\sum_{t=1}^{8} f_t \in \mathbb{R}^D$$

- 덧셈은 **교환법칙**이 성립한다. $f_1+f_2+\dots+f_8 = f_8+f_7+\dots+f_1$.
- 따라서 프레임 순서를 어떻게 섞어도(**임의의 순열 permutation**) 결과가 완전히 같다 → **순열 불변(permutation-invariant)**.
- "왼→오"와 "오→왼" 비디오는 (이상적으로는) 같은 프레임 집합이므로 **평균 벡터가 동일**하다. 그 위에 어떤 분류기를 얹어도 두 클래스를 구분할 수 없다.
- 즉 평균은 "무엇이 나오는가"의 **집합(bag-of-frames)** 정보만 남기고, "언제 나오는가"를 버린다.

### 연결 (concatenation)

$$f_{\text{cat}} = [\,f_1 \;\|\; f_2 \;\|\; \dots \;\|\; f_8\,] \in \mathbb{R}^{8D}$$

- 슬롯 1에는 항상 첫 프레임, 슬롯 8에는 항상 마지막 프레임이 들어간다. **위치가 시간을 인코딩**한다.
- 순서를 뒤집으면 벡터 자체가 달라진다 → **순열에 민감(permutation-sensitive)**.
- linear classifier $W \in \mathbb{R}^{C \times 8D}$ 는 슬롯마다 서로 다른 가중치 $W = [W_1 \| \dots \| W_8]$ 를 배울 수 있다. 그러면 로짓은 $\sum_t W_t f_t$ 가 되어 **"시간 t에 무엇이 있었는가"에 따라 다른 점수**를 줄 수 있다. 예컨대 $W_1$은 "물체가 왼쪽에 있는 특징"에, $W_8$은 "물체가 오른쪽에 있는 특징"에 반응하도록 학습되면 "왼→오"가 구분된다.
- 물론 이것은 아주 원시적인 시간 모델링이다(프레임 간 상호작용은 선형 결합뿐, 속도·가속 같은 2차 정보 없음). 그래도 평균보다는 훨씬 많은 시간 정보가 살아 있어 논문은 "to retain more temporal information than with feature averaging"이라고 표현했다.

### 왜 UCF/K400에도 concatenation을 안 썼나

- 필요가 없다: 외형 정보로 충분히 풀린다.
- 파라미터가 8배로 늘어 linear probe가 과적합·최적화 난이도가 올라간다. 시간 정보가 필요 없는 태스크에서는 손해일 수 있다.
- 평균은 프레임 간 노이즈를 상쇄해 더 안정적인 표현을 준다.

---

## 5. 특징 차원 계산: D vs 8×D

DINOv2 각 아키텍처의 토큰 차원 D와, 두 방식으로 합친 뒤 linear classifier 입력 차원.

| 백본 | D (per frame) | 평균 → D | concat → 8×D |
|---|---|---|---|
| ViT-S/14 | 384 | 384 | 3,072 |
| ViT-B/14 | 768 | 768 | 6,144 |
| ViT-L/14 | 1,024 | 1,024 | 8,192 |
| ViT-g/14 | 1,536 | 1,536 | 12,288 |

- 400클래스 K400 기준 linear head 파라미터: 평균이면 $1536 \times 400 \approx 0.6\text{M}$, concat이면 $12288 \times 400 \approx 4.9\text{M}$.
- 참고: 논문 부록 B.3의 linear probing 그리드에는 "마지막 1개 vs 4개 층 사용", "CLS 토큰 + 평균 풀링된 패치 토큰 concat 여부"가 있어 실제 per-frame D는 최대 $4 \times 2 \times D$ 까지 커질 수 있다. 비디오 평가에서 어떤 조합을 썼는지는 논문에 명시되지 않았으므로, 카드에서는 "프레임당 D → 평균 D / 연결 8D"라는 구조만 기억하면 된다.

---

## 6. 결과 표와의 연결 (논문 Table 7)

> 참고: 이 힌트의 스펙에는 "Table 6"이라고 적혀 있지만, 논문에서 **Table 6은 domain generalization(ImageNet-A/R/Sketch/C)** 이고, 비디오 결과는 **Table 7**에 있다. 아래는 Table 7의 비디오 열이다.

| Feature | Arch | K400 | UCF-101 | SSv2 |
|---|---|---|---|---|
| OpenCLIP | ViT-G/14 | 78.3 | 90.7 | 35.8 |
| MAE | ViT-H/14 | 54.2 | 70.6 | 29.2 |
| DINO | ViT-B/8 | 64.5 | 85.0 | 32.6 |
| iBOT | ViT-L/16 | 72.6 | 88.6 | **38.7** |
| DINOv2 | ViT-S/14 | 67.8 | 87.0 | 33.1 |
| DINOv2 | ViT-B/14 | 73.2 | 89.1 | 34.4 |
| DINOv2 | ViT-L/14 | 76.3 | 90.5 | 35.6 |
| DINOv2 | ViT-g/14 | **78.4** | **91.2** | 38.3 |

모두 frozen feature + linear probe. 단위는 top-1 정확도(%).

읽는 법:

1. **UCF·K400은 90%대·70%대, SSv2는 30%대.** 같은 특징, 같은 linear probe인데 격차가 이렇게 큰 것 자체가 "SSv2는 외형만으로는 안 풀린다"는 증거다. 이미지 특징을 8개 이어 붙인 정도로는 시간적 추론에 한계가 있음을 보여준다. (전용 비디오 모델은 SSv2에서 70%대를 낸다.)
2. **DINOv2 vs OpenCLIP**: UCF +0.1, K400 +0.5로 거의 동률이지만 **SSv2에서 +2.5**로 뚜렷하게 앞선다. 논문은 이 점을 강조한다 — 텍스트 감독(CLIP)은 "무엇이 있는가"(명사)를 잘 잡지만, DINOv2의 자기지도 특징은 손·물체의 미세한 위치·자세 같은 **국소·기하 정보**를 더 잘 보존해서, concatenation을 통해 시간적 변화를 읽어내는 데 더 유리하다는 해석이 가능하다.
3. **SSv2 최고는 iBOT ViT-L/16 (38.7)** 이고 DINOv2 ViT-g/14는 38.3으로 근소하게 뒤진다. 논문이 "amongst self-supervised approaches, our model clearly sets a new state of the art"라고 쓴 것은 UCF·K400 기준이며, SSv2에서는 iBOT과 거의 동률 수준이라는 점을 정직하게 읽어야 한다. (iBOT의 patch 16 vs DINOv2의 patch 14, 백본 크기 차이 등 변수도 다르다.)
4. **모델 크기 스케일링**: S→B→L→g로 갈수록 세 데이터셋 모두 단조 증가. 특히 SSv2도 33.1 → 38.3으로 오르므로, 더 좋은 이미지 특징이 concatenation을 통해 시간 정보 활용에도 도움이 된다.

---

## 7. 핵심 포인트 / 암기 훅

- **"8장 뽑고, 평균 or 이어붙이기, 그 위에 선형 분류기"** — 백본은 얼림, 시간 모듈 없음.
- **평균 = 순서 잊음 (bag-of-frames), 연결 = 순서 기억 (슬롯 = 시간).** 덧셈의 교환법칙이 핵심 이유.
- **UCF/K400 = "무엇이 보이나" (appearance-biased) → 평균 OK.**
  **SSv2 = "어떻게 움직였나" (temporal) → 연결 필요.** 클래스 이름에 "Something"이 들어가는 이유: 물체는 상관없고 동작만 중요.
- 차원: 평균 **D**, 연결 **8×D** (ViT-g: 1536 → 12288).
- 결과(Table 7): UCF/K400은 OpenCLIP과 동률, **SSv2에서 +2.5**로 앞섬 — "더 풍부한 이해가 필요한 태스크"에서 DINOv2가 강함을 시사.
- 이미지 전용 사전학습 모델이 비디오에 **학습 없이(frozen)** 이 정도 나온다는 것이 이 절의 메시지.
