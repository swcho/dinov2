# outlier는 왜 feature map 경계 쪽에 몰리는가

출처: *Vision Transformers Need Registers* (Darcet et al., arXiv:2309.16588) — 부록 A "Interpolation artifacts and outlier position distribution", 부록 D.3 "Positional focus".

---

## 1. 관찰: 위치별 outlier 발생 확률 히트맵

논문은 DINOv2 ViT-g의 patch token 중 norm이 컷오프(150)를 넘는 "high-norm = outlier" 토큰이 **feature map의 어느 위치에서** 얼마나 자주 나오는지를, 여러 이미지에 걸쳐 위치별 비율로 집계했다.

![Fig. 10 — 위치별 high-norm 토큰 발생 비율. 왼쪽: 원본 DINOv2(안티에일리어싱 없음), 오른쪽: 안티에일리어싱 적용](fig-1.jpeg)

그림에서 실제로 보이는 것:

- **왼쪽(원본 DINOv2)**: 밝은 픽셀이 **세로 줄무늬(vertical stripe)** 형태로 배열돼 있다. 특정 열(column)이 통째로 밝다. 어떤 위치는 20% 이상의 이미지에서 high-norm이 뜬다.
- **오른쪽(안티에일리어싱 적용)**: 줄무늬가 사라지고, 남은 밝은 픽셀이 **네 모서리와 가장자리 한 줄**에 집중된다. 정중앙 영역은 거의 전부 검다(=거의 0).

즉 "경계에 몰린다"는 주장의 근거는 **오른쪽 히트맵**이다. 왼쪽의 줄무늬는 별개 원인(아래 4절)이며, 그것을 제거하고 나서야 경계 편향이 깨끗하게 드러난다.

---

## 2. 논문이 제시한 해석: 논증 사슬

부록 A의 원문은 짧다.

> "the outliers tend to appear in areas closer to the border of the feature map rather than in the center. Our interpretation is that the base model tends to recycle tokens in low-informative areas to use as registers: pictures produced by people tend to be object-centric, and in this case the border areas often correspond to background, which contains less information than the center."

이걸 단계로 풀면:

| # | 단계 | 근거 |
|---|---|---|
| 1 | **데이터 편향**: 사람이 찍은/웹에서 수집된 사진은 대체로 object-centric이다. 피사체를 프레임 중앙에 놓고 셔터를 누른다. | 사진 촬영 관행. 논문 D.3에서도 "ImageNet-22k contains mostly object-centric images rather than scenes"라고 명시 |
| 2 | **경계 patch = 배경**: 그 결과 feature map 가장자리의 patch는 하늘·벽·바닥·잔디 같은 배경일 확률이 높다. | 1의 직접 귀결 |
| 3 | **배경 patch = 저정보·이웃과 중복**: 배경은 균일(uniform)해서, 그 patch는 상하좌우 이웃 patch와 거의 같다. 즉 그 토큰이 담을 수 있는 *고유한* 로컬 정보가 거의 없다. | 본문 "High-norm tokens appear where patch information is redundant" |
| 4 | **모델이 그 토큰을 재활용**: 충분히 크고 충분히 오래 학습된 ViT는 "버려도 되는 토큰"을 알아보고, 거기에 로컬 정보 대신 **전역(global) 정보를 저장·처리·회수**하는 용도로 덮어쓴다 — 사실상 즉석 register. | 본문 가설: "recognize redundant tokens, and use them as places to store, process and retrieve global information" |
| 5 | **결과: 위치 분포가 경계에 몰림** | Fig. 10 (right) |

핵심은 3→4 구간이다. 모델은 "여긴 가장자리니까 지워야지"라고 위치를 보고 판단하는 게 아니라, **"이 patch는 이웃과 중복이니 지워도 손해가 없다"**고 내용을 보고 판단한다. 위치 편향은 그 판단 기준이 object-centric 데이터셋 위에서 집계됐을 때 나타나는 **부산물**이다.

---

## 3. 본문 발견과의 연결 — 같은 이야기의 두 가지 표현

이 경계 편향은 새로운 현상이 아니라, 본문 §2의 두 발견을 **공간적으로** 다시 그린 것이다.

![Fig. 5 — (좌) 입력 patch와 4-이웃의 코사인 유사도 분포, (우) 로컬 정보 프로빙](fig-2.jpeg)

- **Fig. 5a (왼쪽 그래프)**: patch embedding 직후, 각 patch와 상하좌우 4-이웃의 코사인 유사도 분포. `artifact patches`(주황) 곡선은 **1.0 바로 앞에서 density 20 가까이 치솟는 뾰족한 봉우리**를 만든다. `normal patches`(파랑)는 0.0~1.0에 넓게 퍼져 있다. → outlier는 **이웃과 거의 똑같은 patch**에서 나온다.
- **Fig. 5b (오른쪽 표)**: outlier 토큰은 position prediction top-1이 41.7 → 22.8로, 평균 거리 오차가 0.79 → 5.09로, pixel reconstruction L2가 18.38 → 25.23으로 악화. → outlier는 **자기 자리와 자기 픽셀 정보를 버렸다**.
- 반면 Table 1의 linear probing에서 outlier 토큰은 normal 토큰보다 이미지 분류 정확도가 훨씬 높다. → 대신 **전역 정보**를 들고 있다.

정리하면:

- **내용 축(본문)**: "high-norm 토큰은 이웃과 코사인 유사도가 높은(=중복) patch에서 나온다."
- **위치 축(부록 A)**: "그런 중복 patch가 object-centric 데이터에서는 주로 가장자리에 있다."

두 번째는 첫 번째에 "웹 사진은 object-centric이다"라는 데이터 사실 하나를 곱한 결과일 뿐이다. 새로운 메커니즘이 아니다.

---

## 4. 반대 방향의 증거: register가 실제로 경계를 본다

register를 명시적으로 추가한 DINOv2+reg 모델에서, 마지막 층의 attention map을 ImageNet-22k 랜덤 서브셋에 대해 평균 낸 그림이다.

![Fig. 16 — [CLS]와 4개 register, 그리고 일반 patch의 평균 attention map](fig-3.jpeg)

그림에서 실제로 보이는 것:

- **(a) [CLS], (b) reg₀**: 정중앙이 밝은 커다란 **중앙 blob**. 가장자리는 어둡다. 논문의 설명대로, ImageNet-22k가 scene이 아니라 object-centric 이미지 위주라서 평균 attention이 중앙에 뭉친다 — 즉 **"관심 대상은 중앙에 있다"는 데이터 편향이 attention map 자체로 시각화된 것**이다. 1절의 outlier 히트맵과 정확히 상보적(밝은 곳이 정반대)이다.
- **(e) reg₃**: 예외적으로 **바깥 테두리가 밝고 중앙이 어둡다**. register마다 담당 영역이 분화(specialization)되며, 그중 하나가 경계 영역을 맡는다.
- **(f) patch**: 아주 좁은 한 점만 밝다. register/[CLS]의 넓은 support와 대비 — register가 patch보다 [CLS]에 가깝게, 즉 전역 정보를 다룬다는 근거.

카드 답의 "모델은 저정보 영역의 토큰을 register로 재활용한다"에서 **register라는 이름의 출처**가 여기다. 명시적 register를 주면 outlier가 patch token에서 완전히 사라지고 그 high-norm 거동이 register로 옮겨간다(Fig. 15, Table 4). 즉 원래 outlier가 하던 일이 register의 일과 같은 일이었다는 뜻이다.

---

## 5. 이 해석의 한계 — 어디까지 검증됐나

시험에서 흔히 흐려지는 지점이므로 분리해서 기억해 둘 것.

**(a) 이것은 상관 관찰에 대한 사후(post-hoc) 해석이다.**
논문이 실제로 측정한 것은 "outlier의 위치 분포가 경계 쪽으로 치우쳐 있다"는 **상관**뿐이다. "웹 사진이 object-centric이라서 그렇다"는 인과 주장은 저자들이 `Our interpretation is that…`이라고 명시적으로 hedge를 걸어 붙인 설명이며, 부록 A에는 이를 직접 검증하는 실험(예: 중앙이 배경인 데이터셋으로 학습해 outlier가 중앙으로 이동하는지 확인)이 **없다**. 검증 가능한 예측이긴 하다 — scene-centric 코퍼스(예: 항공/위성 영상, 파노라마, 균일 텍스처)로 학습하면 경계 편향이 약해지거나 사라져야 한다. 논문은 그 실험을 하지 않았다.

간접 지지 증거는 있다: Fig. 5a의 코사인 유사도 결과가 "위치가 아니라 중복성이 진짜 기준"임을 보이고, Fig. 16의 중앙 blob이 데이터가 실제로 object-centric임을 보여준다. 사슬의 양 끝은 측정됐지만 중간 연결(데이터 편향 → 위치 편향)은 추론이다.

**(b) 보간(interpolation) aliasing과는 완전히 별개 효과다.**
Fig. 10 왼쪽의 **세로 줄무늬**는 object-centricity와 아무 상관이 없다. 원인은 순수한 구현 아티팩트다: 원본 DINOv2 구현이 학습 중 position embedding을 16×16 → 7×7로 **안티에일리어싱 없이 bicubic 보간**했고, 이 연산에 단위 gradient를 통과시키면 Fig. 11처럼 줄무늬 gradient 패턴이 생긴다. 안티에일리어싱을 켜면 줄무늬는 사라진다(Fig. 10 오른쪽).

그러므로 두 효과를 이렇게 구분하라:

| | 줄무늬 패턴 | 경계 편향 |
|---|---|---|
| 원인 | position embedding 보간 aliasing (구현 버그성) | 데이터 편향 + 중복 토큰 재활용 |
| 제거 방법 | 보간에 antialias 적용 | 제거되지 않음 (register 추가로 outlier 자체를 없애야 함) |
| 검증 상태 | gradient 계산으로 **직접 설명됨** | 사후 해석, 미검증 |

Fig. 10 왼쪽 하나만 보고 "outlier가 경계에 몰린다"고 말하면 두 효과가 섞인 그림을 근거로 삼는 셈이다. 저자들이 안티에일리어싱 적용 버전(오른쪽)을 따로 제시한 이유가 이것이다.

**(c) 어차피 "경계"는 원인이 아니라 증상이다.**
근본 원인은 위치가 아니라 정보 중복성이고, 근본 해법도 위치와 무관하다 — 이미지와 독립적인 register 토큰 몇 개를 시퀀스에 붙이면 모델이 patch를 훔쳐 쓸 필요가 없어지고 outlier가 전부 사라진다. 위치 통계는 어디까지나 현상을 이해하는 단서였다.

---

## 한 줄 요약

웹 사진은 object-centric → 경계 patch는 배경이라 이웃과 중복(코사인 유사도 ≈ 1) → 모델이 "버려도 되는 토큰"으로 판단해 전역 정보 저장소(register)로 재활용 → outlier 위치 분포가 경계에 몰림. 이는 본문의 "중복 patch에서 high-norm이 난다"와 같은 이야기를 공간적으로 본 것이며, 상관에 대한 사후 해석이고, 별개 원인인 보간 aliasing 줄무늬와는 구분해야 한다.
