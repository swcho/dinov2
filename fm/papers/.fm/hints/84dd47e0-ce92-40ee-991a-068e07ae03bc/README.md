# register 도입 후 high-norm 토큰은 어디로 갔는가?

> **한 줄 답**: 사라진 게 아니라 **옮겨 갔다**. patch token 쪽 outlier는 완전히 없어지고, high-norm 토큰은 전부 register 집합 안에 들어앉는다. 즉 high-norm outlier를 만들어내던 모델의 *동작 자체*가 register에 흡수(absorbed)된다.

출처: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024), 본문 §2 + **부록 D "Behavior of models trained with registers"**.
부록 D는 서두에서 질문을 이렇게 못박는다 — "register token이 high-norm token을 어느 정도까지 **'replace'** 했고 **같은 역할**을 떠안았는가."

---

## 1. 결정적 증거: 토큰 유형별 norm 분포 (Figure 15)

![DINOv2의 토큰 유형별 output norm 분포 — 왼쪽 register 없음, 오른쪽 register 4개](fig-1.jpeg)

본문 Figure 7이 "register를 쓰면 norm outlier가 없어진다"만 보여줬다면, 부록 Figure 15는 **같은 그림을 토큰 유형별로 쪼갠** 더 세밀한 버전이다. 여기서 x축이 `CLS / patch`가 아니라 `CLS / reg_0 / reg_1 / reg_2 / reg_3 / patch`로 나뉘는 것이 핵심이다.

**(a) register 없음 (왼쪽)**
- `CLS`: norm 20 부근에 얇게 뭉쳐 있음.
- `patch`: 대다수는 20 부근이지만, 위로 50 ~ 200까지 **길게 흩뿌려진 꼬리**가 보인다. 이게 바로 artifact, 즉 high-norm outlier patch다 (본문 Fig. 3 기준 norm > 150인 토큰이 약 2.37%).

**(b) register 4개 (오른쪽)**
- `patch`: 꼬리가 **완전히 잘려나가고** 20 부근의 좁은 띠 하나만 남는다 → 분포가 깨끗한 **unimodal**로 바뀐다.
- 대신 `reg_1` ≈ 77, `reg_2` ≈ 130, `reg_3` ≈ 66 위치에 **높은 norm이 몰려 있다**. (`reg_0`만 ≈ 15로 CLS와 비슷한 낮은 값.)

논문의 결론 문장 그대로:
> "with registers, the norms of patch tokens do not contain outliers anymore, and **the high-norm tokens are entirely contained in the set of registers**. As a result, we conclude that the behavior leading to high-norm outliers in the model is **effectively absorbed in the registers**."

### 왜 이 그림이 논문 전체 논지의 결정적 증거인가

이 지점이 이 카드의 진짜 요점이다. **두 가지 시나리오를 구별해 주기 때문**이다.

| | 관찰되어야 할 것 | 함의 |
|---|---|---|
| 가설 A: register가 artifact를 **없앴다** (suppress) | patch도 낮고, register도 전부 낮음 — 어디에도 high-norm이 없음 | high-norm 동작은 학습의 *버그*였고 register는 그걸 껐다 |
| 가설 B: register가 artifact를 **격리했다** (isolate) | patch는 낮아지고, **그 high-norm이 register로 통째로 이동** | high-norm 동작은 모델에 *필요한 기능*이고, register는 그 저장소를 patch 바깥으로 옮겨준 것뿐 |

Figure 15가 보여주는 건 명백히 **B**다. 만약 A였다면 §2.2의 가설("크고 충분히 학습된 모델은 redundant token을 알아보고 그것을 global 정보를 **저장·처리·회수(store, process, retrieve)** 하는 자리로 쓴다")은 근거를 잃는다. high-norm 동작이 그냥 없어져도 모델이 멀쩡하다면, 애초에 그 동작이 필요했다는 주장이 성립하지 않으니까.

실제로는 동작이 **살아남은 채 자리만 옮겼다**. 그래서 논문의 처방은 "artifact를 없애는 패치"가 아니라 **"모델이 이미 하고 있던 일에 전용 공간을 내주는 것"** 으로 정확히 재해석된다. register는 새로운 능력을 **create**하지 않는다 — 이미 존재하던 능력을 patch 격자 밖으로 **isolate**할 뿐이다. §2.2의 표현대로 "이 동작 자체가 나쁜 건 아니고, 그것이 **patch token 안에서** 일어나는 것이 바람직하지 않을 뿐"이기 때문이다.

부수적으로, 이건 왜 register가 성능을 깎지 않는지도 설명한다 (Table 2a에서 오히려 소폭 개선). 모델의 정보 집약 메커니즘을 제거한 게 아니라 이사시킨 것이므로, 잃을 게 없고 대신 patch token은 local 정보를 온전히 지킬 수 있게 된다.

---

## 2. 같은 그림을 뒷받침하는 부록의 다른 관찰들

Figure 15만으로도 강하지만, 부록 D는 "옮겨 갔다"는 주장을 **norm / 정보 내용 / attention 패턴** 세 축에서 교차 검증한다.

### (a) norm이 양자화된 것처럼 보인다 — D.1

Figure 15(b)를 자세히 보면 각 register의 norm이 **띠(band)처럼 아주 좁게** 찍힌다. reg_1은 77 근처, reg_2는 130 근처로 이미지가 바뀌어도 값이 거의 흔들리지 않는다. 원래 outlier patch의 norm이 50~200에 걸쳐 연속적으로 퍼져 있던 것과 대조적이다.

> "the norms of the registers appear to be **quantized**, compared to the previous outliers; we leave the investigation of this phenomenon for future work."

논문은 원인을 미해결로 남겨두지만, 그림의 함의는 분명하다. register마다 **고유한 역할과 고유한 스케일**로 수렴했다는 뜻이다. 즉 4개의 register가 "여분의 슬롯 4개"로 뭉뚱그려진 게 아니라 서로 다른 기능을 나눠 맡았고 — 이는 아래 (c)의 specialization 관찰과 정확히 맞물린다. (참고로 이 현상은 이후 후속 연구들에서 LayerNorm/softmax와 결합된 "attention sink" 논의로 이어진다.)

### (b) global-probing 성능이 outlier에서 register로 이전된다 — Table 4 (D.2)

norm은 껍데기일 뿐이고, 정말 중요한 건 **그 안에 담긴 정보가 따라 옮겨 갔는가**다. Table 4는 Aircraft 데이터셋에서 각 토큰을 이미지 표현으로 써서 linear probing한 결과다 (본문 Table 1의 register 버전).

| #registers | [CLS] | normal patch | outlier patch | register |
|---|---|---|---|---|
| 0 | 84.6 | 15.5 | **73.3** | — |
| 1 | 85.2 | 14.5 | (존재하지 않음) | **71.1** |

읽는 법:
- **normal patch는 15% 수준으로 무능하다.** 하나의 patch token만으로 비행기 기종을 맞추는 건 원래 불가능한 일이다.
- **register 없을 때 outlier patch는 73.3%.** 한 개의 patch token 주제에 [CLS](84.6)에 육박한다 — 이게 본문 §2.1의 "artifacts hold global information" 주장이다. outlier는 자기 위치의 local 정보를 버리고(Fig. 5b: position prediction·pixel reconstruction 점수가 낮음) 대신 이미지 전역 정보를 담고 있다.
- **register를 넣으면 outlier 행이 통째로 사라지고, 그 자리에 register가 71.1로 들어온다.** 73.3 → 71.1, 거의 그대로다.
- 동시에 **[CLS](84.6→85.2)와 normal patch(15.5→14.5)는 사실상 불변**이다.

즉 "high-norm이라는 껍데기"만 옮겨간 게 아니라 **"global 정보를 aggregate하는 기능" 자체가 이전**됐다. 표 캡션이 쓰는 단어가 그대로 *absorbed into the register*다.

Table 5는 반대편을 잠근다: register 유무와 무관하게 **non-outlier patch의 local 정보량은 그대로**다 (position prediction 66.3 → 65.8, reconstruction L2 error 15.9 → 16.0). 다시 말해 register는 정상 patch를 건드리지 않았고, 오직 outlier 동작만 걷어내 갔다.

### (c) register가 [CLS]처럼 넓은 attention support를 갖는다 — Figure 16 (D.3)

![[CLS]·register 4개·일반 patch의 평균 attention map](fig-2.jpeg)

ImageNet-22k 부분집합에서 마지막 layer의 attention map을 토큰별로 평균낸 그림이다.

- **(f) patch**: 자기 위치 주변에만 뾰족하게 몰린, 극도로 **국소적인** map.
- **(a) [CLS]**: 이미지 전체에 걸친 넓고 부드러운 blob — global 정보를 모으는 토큰의 전형적 서명.
- **(b)~(e) register**: 세부는 다르지만 하나같이 **[CLS]에 가까운 넓은 support**를 갖는다. patch와는 확연히 다르다.

여기에 두 겹의 결론이 붙는다.

1. **register는 global 정보를 읽고 있다.** [CLS]는 linear probing 성능으로 global 정보 운반이 이미 입증된 토큰이고, register의 attention 패턴이 그와 닮았다는 것은 register가 같은 종류의 일 — 이미지 전역에서 정보를 긁어모으는 일 — 을 하고 있다는 방증이다. Table 4의 71.1이라는 숫자에 **메커니즘 수준의 설명**을 붙여주는 셈이다.
2. **register끼리는 서로 다르다.** reg_3은 테두리 영역에, reg_2는 위쪽에 더 반응하는 등 편향이 갈린다. 본문 Figure 9에서 register들이 이미지의 서로 다른 큰 영역에 붙는 (slot attention 비슷한) 모습과 일치한다. 논문의 강조점은 이것이 **아무도 시키지 않았는데 학습에서 자연히 창발했다**는 것이다 — register에는 어떤 loss도, 어떤 구조적 제약도 걸려 있지 않다. 그리고 이 specialization이 (a)의 "norm이 register마다 다른 값으로 양자화된다"와 정확히 같은 이야기의 두 얼굴이다.

---

## 3. 하나의 그림으로 합치기

```
[register 없음]                              [register 있음]
  patch 격자 안 일부 토큰이                     patch 격자는 깨끗
  강제 징발됨 (redundant한 배경 패치)             (norm 분포 unimodal, local 정보 온전)
        │                                            │
        ├─ norm이 폭발 (50~200)   ──── 이동 ────▶  register가 높고 양자화된 norm을 가짐
        ├─ local 정보 상실 (Fig.5b)                  (원래 그 자리의 patch는 local 정보 회복)
        ├─ global 정보 축적 (73.3)  ─── 이동 ────▶  register가 global 정보 축적 (71.1)
        └─ 전역적 attention          ─── 이동 ────▶  register가 [CLS]급 넓은 support
        │
        └─▶ 대가: attention map 오염, dense task 손해, LOST 같은 object discovery 붕괴
```

**모델은 register가 있든 없든 "어딘가에 global 정보를 모아두는 일"을 계속한다.** 차이는 오직 *어디에* 모으느냐다. register 없이는 배경의 redundant patch를 징발해서 그 자리의 local 정보를 파괴하며 쓴다. register가 있으면 그 일을 전용 슬롯에서 처리하고, 출력 시점에 그 슬롯을 그냥 버린다 (Fig. 6) — 그래서 patch feature map과 attention map이 깨끗해진다 (Fig. 1, 19, 20, 21).

**암기용 한 문장**: register는 high-norm 동작을 **없앤 게 아니라 격리했다**. Figure 15에서 high-norm 봉우리가 patch 칸에서 reg 칸으로 자리를 옮긴 것, Table 4에서 73.3이라는 성능이 outlier 행에서 register 행으로 옮겨간 것, Figure 16에서 register가 [CLS]를 닮은 넓은 attention을 갖는 것 — 셋 다 같은 사건의 서로 다른 측정이다.

---

## 4. 자주 헷갈리는 지점

- **"register가 [CLS]를 대체하나?"** — 아니다. [CLS]는 그대로 있고 성능도 그대로다 (84.6 → 85.2). register는 [CLS]와 **별개의** 여분 슬롯이며, 출력에서는 버려진다. 대체된 것은 [CLS]가 아니라 *outlier patch*다.
- **"register 1개면 끝인가?"** — artifact 제거만 놓고 보면 1개로 충분하다 (Fig. 8). 다만 downstream 성능은 4개 정도까지 조금씩 좋아져서 논문은 대부분 실험에서 4개를 쓴다. 비용은 4개 기준 FLOPs +2% 미만, 파라미터 증가는 무시 가능 (Fig. 12).
- **"register 안에 뭐가 들었는지 우리가 아는가?"** — 정확히는 모른다. 논문이 보인 것은 (i) norm이 높고 양자화돼 있으며 (ii) global 정보를 담고 (iii) 넓게 attend한다는 관찰뿐이다. norm 양자화의 원인은 명시적으로 future work로 남겨져 있다.
- **"모든 ViT가 이런가?"** — 아니다. artifact는 **크고(≥ViT-L) 충분히 오래 학습된** 모델에서, 학습 중반쯤 나타난다 (Fig. 4). DINO(v1)에는 없고, patch 단위 local loss만 쓰는 MAE에도 없다 (부록 E) — global aggregation을 요구하는 목적함수가 방아쇠라는 정황이다.
