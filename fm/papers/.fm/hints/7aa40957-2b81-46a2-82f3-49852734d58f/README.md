# pixel reconstruction probing이 시사하는 것

> **Q.** pixel reconstruction probing 결과가 시사하는 바는?
> **A.** high-norm token은 다른 토큰보다 픽셀 복원 정확도가 훨씬 낮다. 즉 이미지를 재구성할 국소 정보를 덜 담고 있다.

출처: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024), §2.1 "High-norm tokens hold little local information", Fig. 5b.

---

## 0. 먼저: "high-norm token(= outlier, artifact)"이 뭔가

![Fig. 3 — DINO vs DINOv2의 patch token norm 분포](fig-2.jpeg)

Fig. 3에서 DINOv2 ViT-g/14의 출력 patch token norm 분포는 **bimodal**이다. 대부분은 norm 0~100 구간에 있지만, 소수의 토큰이 극단적으로 큰 norm을 가진다. 논문은 여기서 **norm > 150** 이라는 단순한 컷오프를 잡고, 그 위쪽 **2.37%** 의 토큰을 "high-norm / outlier / artifact" 토큰이라 부른다(DINO ViT-B/16에는 이런 봉우리가 없다). 이후의 모든 probing 실험은 "normal vs outlier" 이 두 집단을 나눠서 비교하는 구조다.

즉 이 카드의 실험은 **"이 이상한 2.37%의 토큰 안에는 대체 무슨 정보가 들어 있는가"** 를 알아내려는 진단 절차의 일부다.

---

## 1. probing 설계: frozen feature 위의 linear model

핵심 방법론은 아주 단순하다.

1. DINOv2-g를 **freeze** 하고 이미지를 통과시켜 patch embedding을 얻는다.
2. 그 embedding 위에 **linear model 하나만** 학습시킨다.
3. 그 linear model의 성능 = "이 embedding 안에 해당 정보가 선형적으로 읽어낼 수 있는 형태로 얼마나 남아 있는가"의 대리 지표.
4. 같은 probe를 **normal 토큰 집합**과 **outlier 토큰 집합**에 각각 돌려 점수를 비교한다.

linear probe를 쓰는 이유가 중요하다. 강력한 non-linear head를 쓰면 "정보가 있다/없다"가 아니라 head의 표현력을 측정하게 된다. 선형 probe는 *정보가 얼마나 쉽게 접근 가능한 형태로 보존돼 있는가* 를 잰다. 여기서 점수가 낮다는 건 "정보가 지워졌거나, 최소한 선형적으로 복구 불가능할 만큼 뒤섞였다"는 뜻이다.

논문은 **local(국소) 정보**를 겨냥한 두 가지 task를 고른다.

| task | 무엇을 묻는가 | local 정보인 이유 |
|---|---|---|
| **position prediction** | 이 패치가 이미지의 어느 위치에서 왔는가 | 위치 정보는 첫 ViT layer 이전에 absolute position embedding으로 **주입된** 값이다. 원래 토큰 안에 확실히 들어 있었다. |
| **pixel reconstruction** | 이 패치의 원본 픽셀 값은 무엇인가 | 자기 패치의 raw appearance. 그 토큰이 자기 자리를 설명하는 정보. |

두 task 모두 "그 토큰이 **자기 자신의 자리**에 대해 무엇을 알고 있는가"를 묻는다는 점에서 짝을 이룬다.

---

## 2. 결과 수치 (Fig. 5b)

![Fig. 5 — (a) 이웃 패치와의 cosine similarity, (b) local information linear probing](fig-1.jpeg)

오른쪽 표(Fig. 5b)가 이 카드의 근거다.

| patches | position prediction<br>top-1 acc ↑ | position prediction<br>avg. distance ↓ | reconstruction<br>**L2 error ↓** |
|---|---|---|---|
| **normal** | **41.7** | **0.79** | **18.38** |
| **outlier** | 22.8 | 5.09 | **25.23** |

읽는 법:

- **pixel reconstruction (L2 error, 낮을수록 좋음): normal 18.38 → outlier 25.23.** outlier 쪽 복원 오차가 약 **37% 더 크다**. 같은 선형 decoder로도 outlier 토큰에서는 원본 패치 픽셀을 훨씬 못 맞춘다.
- **position prediction (top-1 acc, 높을수록 좋음): 41.7% → 22.8%.** 정확도가 거의 반토막이다.
- **position avg. distance (낮을수록 좋음): 0.79 → 5.09.** 이게 특히 극적이다. normal 토큰은 틀려도 **이웃 패치 정도로만** 빗나가지만(평균 0.79 패치), outlier 토큰은 **평균 5 패치 이상** 떨어진 곳을 찍는다. 즉 "약간 헷갈리는" 수준이 아니라 위치 감각 자체가 거의 사라진 것에 가깝다.

### 주의할 점 (해석의 함정)

- outlier 점수가 **0이 아니다.** 정보가 완전히 소멸한 게 아니라 **상대적으로 크게 줄었다**는 주장이다. 카드 답의 "훨씬 낮다 / 덜 담고 있다"라는 표현이 정확한 강도다.
- 이 열의 화살표 방향이 서로 다르다. reconstruction은 **error**라 낮은 쪽(normal 18.38)이 좋고, position은 **acc**라 높은 쪽(normal 41.7)이 좋다. 표를 급히 읽으면 "outlier가 25로 더 크니 더 잘한다"고 뒤집어 읽기 쉽다.
- normal의 절대 성능(41.7% top-1)도 대단히 높진 않다. 비교는 **outlier 대비 상대값**으로만 의미가 있다.

### 보조 근거로 붙는 Fig. 5a

같은 그림 왼쪽 패널은 **patch embedding 직후**(ViT 첫 layer 이전) 입력 패치와 그 4개 이웃 사이의 cosine similarity 분포다. artifact patch(주황)는 1.0 근처에 거대한 봉우리를 만든다 — 즉 high-norm 토큰은 **이웃과 거의 똑같은, 균일한 배경 영역**에서 생긴다. 이 패치들의 국소 정보는 원래부터 **중복(redundant)** 이라 버려도 이미지 표현 품질에 손해가 없다. Fig. 2의 정성적 관찰(artifact가 하늘·배경 같은 uniform 영역에 뜬다)과도 맞는다.

---

## 3. 논증 구조: 두 probing → 하나의 결론

논문의 추론 사슬을 정리하면 이렇다.

```
[전제 A] high-norm 토큰은 이웃과 매우 유사한 패치에서 생긴다 (Fig. 5a)
         → 그 패치의 국소 정보는 중복이며, 버려도 손해가 적다

[증거 1] position prediction: 41.7 → 22.8 (acc), 0.79 → 5.09 (distance)
         → 첫 layer에서 "주입해 준" 위치 정보조차 출력에 남아 있지 않다

[증거 2] pixel reconstruction: L2 error 18.38 → 25.23
         → 자기 패치의 원본 픽셀을 복원할 appearance 정보도 남아 있지 않다

────────────────────────────────────────────────
[결론 1] high-norm 토큰은 local 정보를 상실했다.
         모델이 추론 과정에서 이 패치의 국소 정보를 "폐기(discard)"했다.
```

증거 1과 2가 **왜 둘 다 필요한가**가 이 논증의 핵심이다. local 정보에는 두 축이 있다 — **"어디"(spatial/position)** 와 **"무엇"(appearance/pixel)**. 하나만 떨어졌다면 "특정 종류의 정보만 압축됐다"는 좁은 해석이 가능하다. 그런데 **두 축이 동시에 무너졌으므로**, 개별 정보 유형의 손실이 아니라 **"국소성 자체가 버려졌다"** 는 일반적 결론으로 올라갈 수 있다. 특히 position은 모델이 명시적으로 입력받은 정보라서, 그게 사라졌다는 건 "원래 없었다"가 아니라 **"중간에 덮어써졌다"** 를 강하게 시사한다(Fig. 4a: outlier는 40-layer 중 layer 15 근처에서 분화한다 — 즉 네트워크 중간에서 일어나는 일이다).

---

## 4. 이 결론이 다음 절로 이어지는 방식

여기서 자연스러운 질문이 생긴다: **정보를 버렸다면, 그 토큰의 자리에는 대신 무엇이 들어갔나?** norm이 오히려 *커졌다*는 사실이 "빈 토큰"이라는 설명을 배제한다. 무언가가 채워졌다.

다음 절 **"Artifacts hold global information"** 이 그 답이다. DINOv2-g의 patch embedding 중 토큰을 **하나만** 무작위로 골라(high-norm 또는 normal) 그것을 이미지 전체의 표현으로 삼고 logistic regression 분류기를 학습한다.

| token | IN1k | Airc. | CUB | Cars | Flow. | Pets | SUN |
|---|---|---|---|---|---|---|---|
| `[CLS]` | 86.0 | 87.3 | 91.3 | 91.5 | 99.7 | 96.9 | 78.6 |
| normal | 65.8 | 17.1 | 18.6 | 10.8 | 59.5 | 47.8 | 37.7 |
| **outlier** | **69.0** | **79.1** | **84.9** | **85.2** | **99.6** | **94.1** | **78.5** |

(Table 1 발췌) outlier 토큰 하나가 fine-grained 데이터셋에서 normal 토큰을 압도하고, 거의 `[CLS]` 급이다 (Aircraft 17.1 → 79.1, Cars 10.8 → 85.2). patch 토큰 하나로 비행기 기종을 맞히려면 그 토큰이 **이미지 전역** 정보를 들고 있어야만 한다.

두 결과를 합치면 논문의 해석이 완성된다:

> 모델은 **쓸모없는(중복된) 정보를 담은 패치를 알아보고, 그 토큰을 재활용(recycle)하여 global 이미지 정보를 집계하며, 그 과정에서 spatial 정보를 폐기한다.**

즉 local 정보 상실은 **부작용**이 아니라 **재활용의 대가**다. 이 "일부 토큰을 전역 연산용 스크래치패드로 징발한다"는 가설이 곧바로 처방으로 이어진다: 모델이 패치 토큰을 훔쳐 쓰지 않도록 **전용 스크래치패드를 따로 주자** → 입력 시퀀스에 이미지와 무관한 학습 가능 토큰 **register** 를 append (Fig. 6). 결과적으로 patch 토큰의 high-norm outlier는 사라지고(그 행동이 register로 흡수되고), feature map이 매끄러워지며 dense prediction 성능이 오른다.

논증의 전체 형태:

```
관찰(artifact 존재) → 진단(local 정보 상실: 이 카드) → 진단(global 정보 보유)
   → 가설(토큰 재활용) → 처방(register) → 검증(outlier 소멸 + 성능 유지/향상)
```

---

## 5. 사후 검증: register가 normal 패치를 건드리지 않는다 (Table 5)

같은 local probing을 register 학습 모델에 다시 적용한 결과(부록 D.2):

| #registers | patches considered | position prediction top-1 acc | reconstruction L2 error ↓ |
|---|---|---|---|
| 0 | non-outliers | 66.3 | 15.9 |
| 4 | non-outliers (= 전부) | 65.8 | 16.0 |

register를 넣어도 normal 패치의 local 정보는 그대로다(15.9 vs 16.0). 즉 register는 **outlier 행동만 걷어내고** 나머지 패치의 정보 내용은 바꾸지 않는다. 이 카드의 probing이 처방의 사후 검증 도구로도 재사용되는 셈이다.

> 참고: Table 5의 절대 수치(66.3 / 15.9)는 Fig. 5b(41.7 / 18.38)와 다르다. 부록의 실험은 설정·모델이 다르므로, **표 안에서의 상대 비교**만 의미가 있고 두 표의 숫자를 가로질러 비교하면 안 된다.

---

## 한 줄 요약

Frozen DINOv2 patch embedding 위에 선형 decoder를 올려 원본 픽셀을 복원시키면 high-norm 토큰의 L2 error가 25.23으로 normal 토큰의 18.38보다 훨씬 크다. 같은 방향의 position prediction 결과(41.7% → 22.8%, 평균 오차 0.79 → 5.09 패치)와 묶이면, high-norm 토큰은 "무엇"과 "어디"를 모두 잃은 **local 정보 폐기 토큰**이라는 결론이 서고, 이는 곧 "그 자리에 global 정보가 대신 들어찼다"(Table 1)는 다음 절과 register라는 처방으로 이어진다.
