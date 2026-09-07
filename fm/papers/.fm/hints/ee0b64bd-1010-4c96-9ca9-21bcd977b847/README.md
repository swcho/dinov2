# memory token 선행 연구 대비 이 논문의 새로운 통찰

**Q.** memory token 관련 선행 연구 대비 이 논문이 기여한 새로운 통찰은?

**A.** memory token이 구현하는 메커니즘이 Vision Transformer에서 **이미 자연적으로 발생하고 있다**는 점이다. 즉 register는 이 동작을 **'만드는(create)' 것이 아니라 '분리(isolate)'**해 부작용을 피하게 해준다.

---

## 0. 논문의 원문 진술

Related Work의 "Additional tokens in transformers" 문단 마지막이 이 카드의 출처다.

> The Memory Transformer (Burtsev et al., 2020), closer to our work, presents a simple approach to improve transformer models using memory tokens added to the token sequence, improving translation performance. (…) **our study contributes the following new insight in Sec. 2: the mechanism implemented through memory tokens already appears naturally in Vision Transformers; our study shows that _such tokens allow us not to create but to isolate this existing behavior_, and thus avoid collateral side-effects.**

여기서 명시적으로 대비되는 선행 연구는 다음과 같다.

| 선행 연구 | 추가 토큰의 역할 | 이 논문과의 차이 |
|---|---|---|
| Memory Transformer (Burtsev et al., 2020) | 시퀀스에 memory token을 붙여 번역 성능 향상 | "새 메커니즘을 준다"는 프레이밍. 왜 도움이 되는지에 대한 자연 발생 근거는 없음 |
| Recurrent Memory Transformer (Bulatov et al., 2022) | copy-repeat-reverse 등 복잡한 태스크 처리 | 태스크 확장이 목적 |
| Learnable Memory ViT (Sandler et al., 2022) | vision 도메인 **fine-tuning**용 memory token | 태스크 간 전이가 잘 안 됨. 이 논문은 fine-tuning이 아니라 **pretraining** 단계에 넣어 모든 downstream에 이득 |
| BERT [SEP]/[CLS], DETR object query, Perceiver latent, AdaTape tape token | 정보를 **주입**하거나, 출력값을 **사용**하거나, 연산량을 **추가** | register는 정보를 주지도 않고 출력값을 쓰지도 않음. 순수히 "저장/검색용 빈 자리" |

포인트: 다른 특수 토큰들은 전부 **없던 기능을 부여**한다. register만 유일하게 **이미 있는 기능을 옮겨 담는 그릇**이다.

---

## 1. 논증 구조 (1) — "메커니즘은 이미 존재한다"는 증거

논문 2절 전반부는 겉보기에 "artifact 분석"이지만, 논증 상으로는 **"memory token이 하려던 일을 ViT가 이미 스스로 하고 있다"는 증명**이다. 세 종류의 증거가 순서대로 쌓인다.

### (a) 어디서 발생하는가 — 중복(redundant) patch에서 발생

- artifact는 출력 token norm이 비정상적으로 큰 **high-norm outlier token** (DINOv2 ViT-g 기준 norm > 150, 전체의 약 2.37%).
- 이 token들이 앉는 위치를 patch embedding 직후에서 조사하면, **이웃 4개 patch와의 cosine similarity가 1.0 근처에 몰려 있다** (Fig. 5a). 즉 하늘·벽 같은 균일한 배경, **버려도 되는 자리**다.
- 게다가 모델이 충분히 크고(ViT-L 이상) 충분히 오래 학습된(1/3 지점 이후) 뒤 **중간 레이어(40층 중 15층 근처)부터** 나타난다 (Fig. 4). 우연한 노이즈가 아니라 **학습을 통해 획득된 전략**이라는 뜻이다.

### (b) 무엇을 버리는가 — local 정보를 버린다

- outlier token에 linear probe를 달아 (i) 자기 patch의 원래 위치 예측, (ii) 원본 픽셀 복원을 시키면 normal token보다 **현저히 낮은 점수**가 나온다 (Fig. 5b).
- 위치 정보는 첫 레이어 앞에서 absolute position embedding으로 **분명히 주입되었는데도** 사라져 있다. → 모델이 능동적으로 **덮어썼다(overwrite)**.

![Fig. 5 — (좌) artifact patch는 이웃과 cosine similarity가 1.0에 몰림 = 중복 영역. (우) outlier token은 위치 예측·픽셀 복원 점수가 낮음 = local 정보를 버림](fig-1.jpeg)

### (c) 무엇을 담는가 — global 정보를 담는다

- patch token 하나를 무작위로 뽑아 그것만으로 이미지 분류기를 학습시키면 (Table 1):

| | IN1k | Airc. | CF100 | CUB | Cars | Flow. | Pets |
|---|---|---|---|---|---|---|---|
| [CLS] | 86.0 | 87.3 | 94.5 | 91.3 | 91.5 | 99.7 | 96.9 |
| normal patch | 65.8 | **17.1** | 81.3 | 18.6 | 10.8 | 59.5 | 47.8 |
| outlier patch | 69.0 | **79.1** | 93.7 | 84.9 | 85.2 | 99.6 | 94.1 |

- Aircraft에서 17.1 → 79.1. outlier token 하나가 **[CLS]에 필적하는 이미지 전역 정보**를 들고 있다.

### (a)+(b)+(c)를 합치면

> 크고 충분히 학습된 ViT는 **중복 token을 스스로 골라내(redundant), 그 안의 local 정보를 지우고(discard), 그 자리를 global 정보의 저장·처리·검색 공간으로 재활용(recycle)한다.**

이것이 정확히 memory token이 손으로 만들어주려던 그 메커니즘이다. **선행 연구는 이 능력을 "부여했다"고 믿었지만, 실제로는 모델이 이미 갖고 있었다.** 다만 놓을 자리가 없어서 patch token을 **강탈**하고 있었을 뿐이다.

---

## 2. 논증 구조 (2) — 그래서 register는 "능력 부여"가 아니라 "자리 마련"

문제의 재정의가 결정적이다. 논문 2.2절의 가설:

> we posit that **while this behavior is not bad in itself**, the fact that it happens **inside the patch tokens** is undesirable.

- 나쁜 것은 **동작**이 아니라 **위치**다. 전역 정보 집약은 유용하다. 문제는 그 계산이 dense prediction에 써야 할 patch token 위에서 벌어져 local 정보를 파괴한다는 점(= collateral side-effect).
- 따라서 처방은 "없던 능력을 심어라"가 아니라 **"이미 있는 능력에게 전용 주차 공간을 내줘라"** 가 된다.
- 구현이 극단적으로 단순한 이유도 여기 있다: patch embedding 뒤에 학습 가능한 토큰 N개를 붙이고, **출력에서는 그냥 버린다**. 손실 함수도, 정규화도, 지도 신호도 추가하지 않는다. 능력을 만들어야 한다면 이 정도로 아무것도 안 해도 될 리가 없다.

![Fig. 15 — register 없이는 patch token에 norm 150+ outlier가 흩어져 있지만(좌), register 4개를 두면 high-norm 동작이 전부 register로 흡수되고 patch는 깨끗해진다(우)](fig-2.jpeg)

Fig. 15가 "isolate"의 시각적 정의다. 고노름 동작의 **총량이 사라진 게 아니라, 위치가 옮겨졌다**. 논문 표현으로 "the behavior leading to high-norm outliers in the model is **effectively absorbed in the registers**".

---

## 3. 논증 구조 (3) — "성능은 유지·향상, 부작용만 소멸"이 이 프레이밍의 예측이다

"create"라면 성능이 **올라야** 한다(새 능력이 생겼으므로). "isolate"라면 성능은 **거의 그대로이고 부작용만 사라져야** 한다. 관측은 후자에 정확히 부합한다.

### Table 2a — 성능은 회귀하지 않는다 (그리고 폭증하지도 않는다)

| | ImageNet Top-1 | ADE20k mIoU | NYUd rmse ↓ |
|---|---|---|---|
| DeiT-III | 84.7 | 38.9 | 0.511 |
| DeiT-III+reg | 84.7 | 39.1 | 0.512 |
| OpenCLIP | 78.2 | 26.6 | 0.702 |
| OpenCLIP+reg | 78.1 | 26.7 | 0.661 |
| DINOv2 | 84.3 | 46.6 | 0.378 |
| DINOv2+reg | 84.8 | 47.9 | 0.366 |

- 분류(global) 성능은 **거의 불변**. 실험 논조도 "성능이 좋아졌다"가 아니라 애초에 **"performance regression"** 체크, 즉 *망가지지 않았음*의 확인이다.
- 개선은 dense task(ADE20k, NYUd)와 object discovery에 몰린다 — 정확히 **local 정보가 강탈당하던 곳**이다. LOST corloc은 DINOv2 VOC07 35.3 → 55.4, DeiT-III 11.7 → 27.1 (Table 3). 즉 **부작용이 사라진 만큼만 좋아진다.**
- Fig. 8 ablation도 같은 이야기: register **1개면 artifact가 이미 사라진다**. 새 능력을 학습시키는 중이라면 이렇게 즉각 포화될 이유가 없다.

### 부록 Table 4 (Aircraft) — 이 프레이밍의 가장 직접적인 증거

| #registers | [CLS] | normal patch | outlier patch | register |
|---|---|---|---|---|
| 0 | 84.6 | 15.5 | **73.3** | N/A |
| 1 | 85.2 | 14.5 | N/A (존재하지 않음) | **71.1** |

읽는 법:
1. **[CLS] 84.6 → 85.2, normal patch 15.5 → 14.5** — register를 넣어도 기존 token들의 probing 점수는 **거의 그대로**다. 새 표현력이 주입된 게 아니다.
2. **outlier patch 73.3이 사라지고, 그 자리에 register 71.1이 나타난다** — 전역 정보를 나르던 그 능력이 **소멸하지 않고 register로 이주**했다. 값까지 비슷하다.
3. 논문 문장: "the behavior of the outlier tokens, aggregating global information, is **absorbed into the register**."

### 부록 Table 5 — 남은 patch는 건드리지 않았다

| #registers | patches | position pred. top-1 | reconstruction L2 ↓ |
|---|---|---|---|
| 0 | non-outliers | 66.3 | 15.9 |
| 4 | non-outliers (= 전체) | 65.8 | 16.0 |

- 원래도 멀쩡했던 normal patch의 local 정보량은 **변하지 않는다**. register가 한 일은 "정상 patch를 더 좋게 만들기"가 아니라 **"희생당하던 patch를 희생시키지 않기"** 다.
- 주목: register 4개일 때 non-outlier가 곧 **전체 patch**다. 강탈당하는 patch 자체가 없어졌다는 뜻이며, 이것이 dense task 개선의 실제 출처다.

### 부록 D.3 / Fig. 9, 16 — 요구한 적 없는 행동까지 그대로 따라온다

- register의 평균 attention map은 [CLS]처럼 **넓은 support**를 갖고(patch는 국소적), 서로 다른 객체/영역에 slot attention처럼 **자발적으로 분화**한다. reg3는 테두리, reg2는 상단 쪽으로.
- 논문 표현: "This behaviour was **never required** from the model, and **emerged naturally** from training."
- 아무 지도 신호 없이 붙여둔 빈 토큰이 곧바로 예전 outlier의 역할·attention 패턴·전역 정보량을 물려받는다는 것은, 그 역할이 **register가 만든 것이 아니라 원래부터 모델 안에 있던 것**이라는 가장 강한 증거다.

---

## 4. 왜 "create vs isolate" 구분이 중요한가

1. **처방의 성격이 달라진다.** create라면 register 수·용량·손실항을 튜닝해야 할 하이퍼파라미터로 본다. isolate라면 "부작용이 사라질 최소 개수"면 충분하다 — 실제로 1개로 artifact가 없어지고, 저자들은 4개를 쓴다(FLOPs 증가 2% 미만).
2. **일반성을 예측한다.** 자연 발생하는 메커니즘이라면 학습 방식과 무관해야 한다. 실제로 DINOv2(자기지도), DeiT-III(레이블), OpenCLIP(텍스트) 모두에서 artifact가 나타나고 모두 register로 해소된다. 반대로 **DINO(작음)와 MAE(전역 집약 목적함수 없음)에는 artifact가 없다** — "전역 정보를 모아야 하는 압력 + 충분한 규모"가 있을 때만 발생한다는 예측과 일치.
3. **artifact의 도덕적 지위가 바뀐다.** artifact는 버그가 아니라 **유용한 내부 계산이 잘못된 장소에서 새어나온 흔적**이다. 그래서 "억제"가 아니라 "배출구 제공"이 옳은 해법이 된다.
4. **DINOv2가 퇴보한 게 아니다.** 논문 서두의 반전 — DINO에 artifact가 없는 것이 **예외**이고, DINOv2가 ViT의 기본 동작(baseline behavior)에 부합한다.

한 문장 요약: **register는 새 기능을 추가하는 부품이 아니라, 이미 돌고 있던 계산에 붙여준 전용 스크래치패드다.**

---

## 5. 이후 연구로의 연결 (참고)

이 "이미 존재하는 메커니즘" 관점은 NLP 쪽에서 독립적으로 발견된 현상들과 곧바로 맞물렸다.

- **Attention sink (Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023 / ICLR 2024).** LLM이 의미와 무관하게 첫 토큰에 attention을 몰아주는 현상. softmax가 합 1을 강제하니 **남는 attention을 버릴 곳**이 필요하다는 설명이다. 이들도 처방으로 사전학습 때 전용 **sink token(placeholder)** 하나를 앞에 붙일 것을 제안했는데, 이는 register와 사실상 같은 "isolate" 처방이다.
- **Massive activations (Sun et al., 2024).** 소수 차원에서 발생하는 거대 활성값이 실질적으로 **고정된 attention bias**로 작동함을 보였고, 이를 명시적 bias 항으로 빼내면 massive activation이 사라진다고 보고했다. 같은 현상이 **ViT에서도** 관찰되며 register와 직접 연결된다. "값을 눌러 없애는 게 아니라 전용 통로를 만들어준다"는 구조가 동일하다.
- **Vision Transformers Don't Need Trained Registers (Jiang, Dravid, Efros, Gandelsman, 2025).** high-norm token을 만들어내는 **특정 뉴런 활성**을 찾아, 재학습 없이 테스트 타임에 그 활성을 별도 토큰으로 옮겨 register 효과를 재현했다. 재학습 없이 "옮기기"만으로 된다는 사실은 **isolate 프레이밍의 사후 확증**에 가깝다 — 메커니즘이 원래 거기 있었기에 옮길 수 있었던 것이다.
- 이후 attention sink 완화·재분배(예: LVLM에서 무의미한 visual token이 attention 예산을 빨아들이는 visual attention sink 문제), sink-aware training, KV-cache pruning 등으로 계열이 확장되었다.

**출처**
- [Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588)
- [Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)
- [Massive Activations in Large Language Models (arXiv:2402.17762)](https://arxiv.org/pdf/2402.17762)
- [Vision Transformers Don't Need Trained Registers (arXiv:2506.08010)](https://arxiv.org/pdf/2506.08010)

---

## 6. 30초 복습

- **선행 연구(Memory Transformer 등):** 추가 토큰이 memory 메커니즘을 **부여한다**고 봄.
- **이 논문의 새 통찰:** 그 메커니즘은 큰 ViT에 **이미 자연 발생**한다. 다만 patch token을 강탈해서 벌어진다.
- **증거 3종:** 중복 patch에서 발생(Fig. 5a) → local 정보 소실(Fig. 5b) → global 정보 보유(Table 1).
- **따라서 register의 역할:** 능력 창조가 아니라 **자리 제공 = 격리**. 결과적으로 부작용(artifact)만 제거.
- **결정적 증거:** Table 4에서 [CLS]/normal patch 점수는 불변, outlier 73.3의 역할이 register 71.1로 **이동**. Table 5에서 정상 patch의 local 정보도 불변. Table 2a에서 성능은 유지되거나 dense task에서만 상승.
