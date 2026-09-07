# OpenCLIP에서 LOST가 register로 개선되지 않은 이유

> 출처: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024), 본문 §3.3 및 **부록 C — Analysis of LOST Performance**

---

## 1. 문제 상황: 표 3의 이상한 한 줄

논문은 register 토큰을 붙인 모델과 붙이지 않은 모델에 대해 unsupervised object discovery 알고리즘 **LOST**를 돌려 corloc을 비교한다 (Table 3).

| 모델 | VOC 2007 | VOC 2012 | COCO 20k |
|---|---|---|---|
| DeiT-III | 11.7 | 13.1 | 10.7 |
| DeiT-III+reg | **27.1** | **32.7** | **25.1** |
| OpenCLIP | **38.8** | **44.3** | **31.0** |
| OpenCLIP+reg | 37.1 | 42.0 | 27.9 |
| DINOv2 | 35.3 | 40.2 | 26.9 |
| DINOv2+reg | **55.4** | **60.0** | **42.0** |

- DINOv2: +20.1 corloc (35.3 → 55.4). 극적인 개선.
- DeiT-III: 11.7 → 27.1. 역시 큰 개선.
- **OpenCLIP: 38.8 → 37.1. 오히려 아주 살짝 하락.**

register가 artifact(high-norm outlier patch)를 제거한다는 것은 OpenCLIP에서도 분명히 확인된다 (Fig. 7의 norm 분포, Fig. 20의 feature map). 그런데 **artifact를 지웠는데 LOST 점수는 안 오른다.** 부록 C가 이 모순을 해명한다.

---

## 2. 결정적 실험 설정: "무엇을 feature로 쓰는가"

LOST는 backbone의 patch 표현들 사이의 유사도(gram matrix)로 seed를 고르고 확장한다. 그런데 "patch 표현"으로 무엇을 넣느냐는 구현 선택 사항이다. 논문 §3.3은 이렇게 밝힌다:

> *"We use **values** for DeiT and OpenCLIP, and for DINOv2, we use **keys**."*

즉 OpenCLIP 실험에서는 **transformer 블록의 최종 출력 feature를 쓰는 게 아니라, attention 계산 중간 산물인 value 벡터 $v = W_v x$ 를 쓴다.** 이 한 줄이 카드의 핵심 전제다.

부록 C:

> *"In the LOST experiment with OpenCLIP models, we do not use the features directly, but the **values** from the computation of attention maps."*

---

## 3. 그림으로 본 증거 (Fig. 14)

부록 C는 OpenCLIP에 대해 **key / query / value 각각을 LOST에 넣었을 때의 seed expansion score**를 register 유무로 나눠 시각화한다.

![Fig. 14 — OpenCLIP의 key/query/value별 seed expansion score (위: register 없음, 아래: register 있음)](fig-1.jpeg)

그림에서 실제로 관찰되는 바:

| | w/o REG (위 줄) | w/ REG (아래 줄) |
|---|---|---|
| **keys** | 배경 전체가 노란/보라 얼룩으로 뒤덮임 — 전형적인 outlier artifact. 새의 형체가 거의 안 보임 | 배경이 훨씬 잔잔해지고 새의 실루엣이 드러남 (여전히 잔점은 남음) |
| **queries** | 배경에 밝은 점들이 흩뿌려짐. 물체가 아닌 곳이 높은 점수를 받음 | 점들이 사라지고 새 몸통에 점수가 집중됨 |
| **values** | **이미 깨끗하다.** 배경에 점이 없고 새 몸통만 밝게 뜬다 | 위와 거의 구분되지 않는다 |

논문 Figure 14 캡션:

> *"Interestingly, the seed expansion map computed using **values** does not exhibit artifacts **with nor without** registers."*

핵심은 마지막 줄이다. key/query 맵은 register가 있고 없고의 차이가 눈에 띄게 크지만, **value 맵은 register 유무에 따른 차이가 거의 없다.** 애초에 register 없이도 artifact가 없기 때문이다.

같은 현상이 Fig. 13(LOST 중간 단계 전체 비교)에서도 나타난다. DeiT-III와 DINOv2는 w/o REG 열이 점 투성이인데, OpenCLIP은 w/o REG 열부터 이미 매끈하다.

![Fig. 13 — 세 모델의 LOST 중간 계산 (LOST score / dot prod. w/ seed / seed expansion)](fig-2.jpeg)

> *"The difference is less striking for the OpenCLIP model."*

---

## 4. 논문의 해석: value projection이 outlier를 걸러낸다

부록 C의 결론 문장:

> *"We qualitatively observe that for the OpenCLIP model, the **value projection filters out the outliers** even without registers. This means that the outliers **appear to live in the null space of the value projection layer**; the investigation for this phenomenon is left for future work."*

논리 흐름을 정리하면:

1. OpenCLIP의 patch token $x$ 중 일부는 high-norm outlier다 (Fig. 7, Fig. 21이 증명).
2. 그 $x$를 그대로 쓰거나 $W_k x$, $W_q x$로 투영하면 outlier 성분이 살아남아 맵에 점으로 보인다.
3. 그런데 $W_v x$로 투영하면 그 성분이 사라진다.
4. → outlier가 담고 있는 방향 성분이 대체로 $W_v$의 **null space**(즉 $W_v u = 0$ 이 되는 $u$들의 집합) 안에 놓여 있다고 보면 설명이 된다.
5. → LOST가 OpenCLIP에서 쓰는 신호(value)는 **처음부터 artifact가 없는 신호**였다. register는 $x$ 쪽의 outlier를 없애 주지만, LOST가 보는 $W_v x$에는 그 outlier가 애초에 들어오지 않았으므로 **얻을 게 없다.**

### 주의: 이것은 추정이지 증명이 아니다

논문의 표현은 `seems`/`appear to live`이며, 저자들이 명시적으로 *"the investigation for this phenomenon is left for future work"* 라고 못박았다. 즉 다음은 **하지 않았다**:

- $W_v$의 SVD를 구해 outlier 방향의 singular value가 실제로 0에 가까운지 확인
- $\|W_v x_{\text{outlier}}\| / \|x_{\text{outlier}}\|$ 를 정량 측정
- 왜 학습이 그런 배치를 만드는지에 대한 인과적 설명

근거는 **정성적(qualitative) 관찰 하나** — Fig. 14의 value 열이 register 유무에 무관하게 깨끗하다는 것 — 뿐이다. 카드 답을 말할 때도 "null space에 있는 것으로 **보인다**"로 서술하는 편이 정확하다.

---

## 5. 왜 그런 일이 일어날 법한가 (논문 밖 직관)

논문이 남긴 빈칸이지만, 이 논문 자체의 outlier 해석과 이어 붙이면 자연스러운 그림이 나온다.

논문 §2의 해석: 모델은 **정보량이 적은(redundant) 배경 patch를 재활용해 전역 정보를 담는 내부 스크래치패드로 쓴다.** 그 토큰의 local 정보는 버려지고(Fig. 6: patch 재구성/위치 예측 성능이 낮음), 대신 global 정보가 실린다(Fig. 6: 이미지 분류 성능이 높음).

attention에서 각 역할은 이렇다:

- $W_q x$, $W_k x$: "누가 누구를 볼지" 정하는 **주소** 역할. outlier 토큰이 global 정보를 뿌리는 broadcaster로 쓰이려면 다른 토큰들이 그것을 **찾아야** 하므로, outlier 신호가 key/query에 강하게 남아 있어야 한다.
- $W_v x$: 실제로 **전달되는 내용물**.

만약 학습 과정에서 attention이 이 outlier 토큰을 "읽어야 할 실질 내용"으로 쓰지 않는 방향으로 수렴한다면, $W_v$가 그 성분을 통과시킬 이유가 없다. $W_v$는 $d$차원을 head 차원 $d/h$로 **줄이는** 납작한 행렬이라 반드시 무언가를 버려야 하는데, 버릴 우선순위 1번이 바로 "내용으로는 쓰이지 않는 outlier 방향"이 되는 것이다.

(이 문단은 논문이 검증한 내용이 아니라 정황상 그럴듯한 해석이다.)

---

## 6. 그래서 무엇을 기억할 것인가

- **표 3에서 OpenCLIP만 register 효과가 없는 이유는 "register가 OpenCLIP에서 안 통해서"가 아니다.** register는 OpenCLIP에서도 artifact를 완전히 제거한다 (Fig. 7, 20).
- 이유는 **평가 파이프라인 쪽**에 있다. LOST가 OpenCLIP에서 사용하는 입력이 feature가 아니라 **value**이고, value는 register 없이도 이미 깨끗했다.
- 메커니즘 가설: outlier 성분이 **value projection $W_v$의 null space**에 놓여 있어 $W_v$가 자동으로 걸러낸다 (저자들도 추정으로만 제시).
- 일반화된 교훈: **"artifact가 있다"와 "그 artifact가 특정 downstream 지표를 망친다"는 별개다.** 어떤 텐서를 뽑아 쓰느냐(feature / key / query / value)에 따라 artifact가 보이기도, 안 보이기도 한다. 실제로 논문도 DINOv2에는 key를, DeiT/OpenCLIP에는 value를 썼다.

---

## 7. 관련 그림/표 색인

| 참조 | 내용 |
|---|---|
| Table 3 (§3.3) | LOST corloc — OpenCLIP만 register로 소폭 하락 |
| Fig. 7 (§3.1) | register가 OpenCLIP의 norm outlier를 실제로 제거함 |
| Fig. 13 (부록 C) | 세 모델 LOST 중간 단계. OpenCLIP은 w/o REG부터 이미 매끈 |
| **Fig. 14 (부록 C)** | **핵심 근거** — OpenCLIP의 key/query/value별 seed expansion |
| Fig. 20 (부록 F) | register 유무에 따른 feature map (PCA) 비교 |
| Fig. 21 (부록 F) | patch token norm — artifact patch = norm outlier 확인 |
