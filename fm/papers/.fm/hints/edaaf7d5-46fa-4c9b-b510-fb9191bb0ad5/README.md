# position prediction probing 결과가 시사하는 바는?

**한 줄 답**: high-norm(outlier) 토큰은 patch 위치를 맞히는 linear probe 정확도가 정상 토큰보다 훨씬 낮다(41.7% → 22.8%). 즉 첫 ViT 층 이전에 **절대 위치 임베딩으로 주입되었던 위치 정보를 이 토큰들은 출력 시점에 상당 부분 잃어버린(덜 보유한)** 상태다.

출처: *Vision Transformers Need Registers* (Darcet et al., ICLR 2024, arXiv 2309.16588) §2.1 "High-norm tokens hold little local information", Fig. 5b.

---

## 1. 먼저, 무엇을 "high-norm 토큰"이라 부르는가

논문은 DINOv2 ViT-g의 **출력 patch 토큰 임베딩의 L2 norm** 분포가 뚜렷하게 bimodal 임을 관찰하고, norm > 150 인 토큰을 "high-norm" = "artifact" = "outlier" 토큰으로 정의한다(모델마다 cutoff는 달라짐). 전체 토큰의 약 **2.37%** 에 불과하며, 주변 패치와 매우 유사한(=중복된) 배경/균일 영역에서 주로 나타난다.

![Fig. 3: DINO vs DINOv2 local feature norm — DINOv2에만 norm 150 이상의 소수 outlier가 존재](fig-2.jpeg)

- 왼쪽: 같은 이미지에 대한 DINO(ViT-B/16)와 DINOv2(ViT-g/14)의 patch별 feature norm 맵. DINOv2 맵에만 유독 밝게 튀는 몇 개의 픽셀(=artifact patch)이 보인다.
- 오른쪽: norm 히스토그램이 0~100 대의 큰 봉우리와 수백 대의 작은 봉우리로 갈라져 있어, 150이라는 손쉬운 임계값을 고를 수 있다.

이 소수의 토큰들이 attention map에서 튀는 점(Fig. 2의 peaky 값)의 정체이고, LOST 같은 object discovery가 DINOv2에서 망가지는 원인이었다.

## 2. 절대 위치 임베딩(absolute position embedding)이란

ViT는 이미지를 패치로 잘라 각 패치를 선형 사영해 토큰으로 만든다. 이 자체로는 순서 정보가 전혀 없다(self-attention은 permutation-equivariant). 그래서 **패치 격자의 각 위치마다 하나씩 학습 가능한 벡터** `pos_embed[i] ∈ R^D` 를 두고, **첫 번째 Transformer 블록에 들어가기 전에** 패치 임베딩에 더해 준다.

```
z_0 = [x_cls; x_p^1 E; ...; x_p^N E] + E_pos      # E_pos: (N+1) x D 학습 파라미터
```

DINOv2 구현에서도 동일하다(`dinov2/models/vision_transformer.py`): `self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim))` 를 `trunc_normal_` 로 초기화해 두고, 입력 해상도가 바뀌면 bicubic `interpolate` 로 격자를 리사이즈해 더한다. (참고: 논문 Appendix A는 학습 시 16×16 → 7×7 보간을 antialiasing 없이 수행한 것이 outlier 위치의 세로 줄무늬 패턴을 만들었다고 지적한다. 즉 위치 임베딩 처리 방식이 artifact 분포에도 흔적을 남긴다.)

핵심은 **위치 정보가 "입력 단계에서 한 번" 더해진 뒤 40개 층을 통과하는 동안 유지되어야 한다**는 점이다. 출력 토큰에서 위치를 못 맞힌다면, 그것은 위치 정보가 애초에 없어서가 아니라 **중간 층들이 그 정보를 덮어썼기 때문**이다.

## 3. 실험 설계: position prediction probing

- 동결된 DINOv2-g의 **출력 patch 임베딩** 위에 **linear model** 하나만 학습한다(백본은 건드리지 않는다 → 표현 안에 그 정보가 선형적으로 남아 있는지를 측정).
- 과제 1 **position prediction**: 그 토큰이 이미지의 어느 패치 좌표에서 왔는지 분류. 지표는 top-1 accuracy와 평균 거리(avg. distance).
- 과제 2 **pixel reconstruction**: 해당 패치의 원본 픽셀 값 복원. 지표는 L2 error.
- normal 토큰 집합과 outlier(norm > 150) 토큰 집합에 대해 각각 학습·평가해 비교한다.

## 4. 결과 (Fig. 5)

![Fig. 5: (a) 이웃 패치와의 cosine similarity 분포, (b) local information probing 결과](fig-1.jpeg)

**(b) 표의 수치 — 이 카드의 근거:**

| patches | position prediction top-1 acc | position avg. distance ↓ | reconstruction L2 error ↓ |
|---|---|---|---|
| **normal** | **41.7** | **0.79** | **18.38** |
| outlier | 22.8 | 5.09 | 25.23 |

- 위치 top-1 정확도가 41.7% → **22.8%** 로 거의 절반. 게다가 틀릴 때의 평균 거리도 0.79 패치 → **5.09 패치**로 6배 이상 벌어진다. 즉 outlier 토큰은 "한 칸 옆으로 헷갈리는" 수준이 아니라 **이미지 어디에서 왔는지 감을 잃은** 상태다.
- pixel reconstruction L2도 18.38 → 25.23으로 악화. 위치뿐 아니라 **원본 픽셀이라는 국소 정보 전반**이 사라졌다.
- **(a) 그림**은 그 전제를 보여준다: artifact patch(주황)의 이웃 4개와의 cosine similarity가 1.0 근처에 극단적으로 몰려 있다. outlier가 생기는 자리는 **주변과 거의 동일해 버려도 손해가 없는 중복 패치**라는 뜻이다.

부록 Table 5는 대조군으로, 레지스터 유무와 무관하게 **non-outlier 패치들의 위치 예측 성능은 66.3 vs 65.8 로 사실상 동일**하다고 보고한다(Fig. 5b와 학습 설정이 달라 절대값은 다르지만, 비교의 방향은 같다). 즉 레지스터는 정상 패치의 정보를 건드리지 않고 outlier 현상만 걷어낸다.

## 5. 왜 위치 정보가 사라지는가 — 논문의 가설

Fig. 5의 "잃어버린 것"과 짝을 이루는 것이 Table 1의 "얻은 것"이다. patch 토큰 하나만 이미지 표현으로 써서 분류기를 학습하면:

| token | IN1k | Aircraft | CUB | Cars |
|---|---|---|---|---|
| [CLS] | 86.0 | 87.3 | 91.3 | 91.5 |
| normal patch | 65.8 | 17.1 | 18.6 | 10.8 |
| **outlier patch** | **69.0** | **79.1** | **84.9** | **85.2** |

outlier 토큰 하나가 [CLS] 토큰에 육박하는 **전역(global) 분류 성능**을 낸다. 두 관찰을 합치면 논문의 가설이 나온다:

> **충분히 크고 충분히 오래 학습된 ViT는 "중복(redundant)"한 패치를 알아보고, 그 토큰을 전역 이미지 정보를 저장·처리·조회하는 스크래치패드(레지스터)로 재활용한다.**

메커니즘적으로 읽으면:

1. 배경처럼 이웃과 똑같은 패치는 그 자리에 담긴 국소 정보의 **한계 가치가 0에 가깝다**(Fig. 5a).
2. 반면 모델은 전역 정보를 모아 둘 "여유 슬롯"이 필요하다(DINOv2의 이미지 수준 목적함수가 이런 aggregation을 유도한다). Transformer에는 그런 전용 슬롯이 없으므로 **기존 patch 토큰을 징발**한다.
3. 그 결과 해당 토큰의 residual stream에는 전역 정보가 대량으로 **덮어써지고**, 원래 실려 있던 patch content와 **입력 시점에 더해진 절대 위치 임베딩은 밀려난다**. 15층 부근(40층 중)에서 norm이 튀기 시작하는 것이 이 재작성의 신호다(Fig. 4a).
4. 그래서 출력에서 linear probe로 위치를 읽어낼 수 없다 → **22.8%**.

즉 position probing 결과는 **"모델이 이 토큰의 국소·공간 정보를 의도적으로 버렸다"는 직접 증거**이며, 동시에 "그 자리에 무엇이 들어왔는가"에 대한 답(Table 1의 global 정보)과 맞물려 register 가설을 완성한다.

## 6. 그래서 무엇을 고쳤나

문제는 전역 정보를 모으는 행동 자체가 아니라, 그것이 **patch 토큰 안에서** 일어난다는 점이다. patch 토큰은 dense prediction(세그멘테이션, 깊이 추정)에서 위치·국소 정보를 그대로 써야 하는 출력이기 때문이다. 해결책은 단순하다: patch embedding 뒤에 [CLS]처럼 학습 가능한 **register 토큰 N개**를 추가로 붙이고 출력에서는 버린다. 그러면 high-norm 행동이 전부 레지스터로 흡수되어 patch 토큰의 norm outlier가 사라지고(Fig. 7, 15), feature map이 매끄러워지며 dense task 성능과 object discovery(LOST)가 개선된다.

---

### 암기 포인트

- 수치: position top-1 **41.7 (normal) vs 22.8 (outlier)**, avg. distance 0.79 vs 5.09, reconstruction L2 18.38 vs 25.23.
- 해석: 위치 정보는 **첫 층 전에 absolute position embedding으로 주입**된다 → 출력에서 못 읽는다는 건 **중간 층이 덮어썼다**는 뜻.
- 짝이 되는 증거: Table 1(outlier 토큰의 높은 이미지 분류 정확도) → 잃은 것은 local/위치, 얻은 것은 global.
- 결론: 중복 토큰을 global 정보 저장소로 **재활용** → 그래서 register 토큰이 필요하다.
