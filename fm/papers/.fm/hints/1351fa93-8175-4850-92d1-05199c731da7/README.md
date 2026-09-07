# 이 동작 자체는 나쁘지 않은데도 왜 문제인가

> 출처: *Vision Transformers Need Registers* (Darcet et al., 2023), 특히 §2.2 "Hypothesis and remediation"

## 0. 문제의 "동작"이 무엇인지부터

논문이 관찰한 현상은 이렇다. 충분히 크고 충분히 오래 학습된 ViT는 출력 patch token 중 약 2%가
나머지보다 **norm이 10배쯤 큰 outlier**가 된다. 이 high-norm token들을 조사해 보면:

- **어디에 생기나**: 이웃 patch와 cosine similarity가 매우 높은 자리, 즉 하늘·벽·잔디 같은
  **중복되고 정보가 적은 배경 영역**에 생긴다.
- **무엇을 잃었나**: 그 token으로 position 예측 / pixel 재구성을 linear probing하면 정확도가
  일반 token보다 크게 낮다. 즉 **자기 자리의 국소 정보를 거의 갖고 있지 않다**.
- **무엇을 얻었나**: 반대로 그 token 하나만 가지고 이미지 분류를 linear probing하면 정확도가
  일반 patch token보다 훨씬 높다 (예: Aircraft 17.1 → 79.1, Cars 10.8 → 85.2, [CLS]에 거의 근접).
  즉 **이미지 전체에 대한 global 정보**를 담고 있다.

![Fig. 5b — outlier token은 position/pixel 국소 정보를 훨씬 덜 담고 있다](fig-1.jpeg)

논문의 가설은 여기서 나온다: 모델은 "쓸모없는 patch"를 알아보고, 그 자리를 **global 정보를
저장(store)·처리(process)·회수(retrieve)하는 스크래치 공간**으로 재활용(recycle)한다.

## 1. 그래서 왜 "동작 자체는 나쁘지 않다"인가

Transformer가 몇 개의 token을 골라 전역 정보를 모아두고 나중에 꺼내 쓰는 것은
**정상적이고 유용한 연산**이다. [CLS] token이 하는 일이 바로 그것이고, NLP의 Memory Transformer도
같은 메커니즘을 명시적으로 도입해 번역 성능을 올렸다. 이미지 전체를 요약하는 계산은 어딘가에서
반드시 일어나야 하고, 그 계산 자체를 없애면 오히려 표현력이 줄어든다.

실제로 논문이 register(별도 토큰)를 붙여서 이 동작을 **다른 곳으로 옮겨주었을 때**, 그 동작은
사라지지 않고 register token으로 그대로 이전됐다 (Appendix: register가 high-norm이 되고,
register만으로 분류 probing을 해도 outlier와 비슷한 성능이 나옴). 즉 모델은 이 연산을 계속 필요로 한다.

논문 원문:

> "we posit that while this behavior is not bad in itself, **the fact that it happens inside the
> patch tokens is undesirable**. Indeed, it leads the model to discard local patch information,
> possibly incurring decreased performance on dense prediction tasks."

## 2. 문제는 "저장소의 위치"다 — patch token은 출력이다

핵심 구조는 이렇게 정리된다.

| | 하는 일 | 문제 여부 |
|---|---|---|
| global 정보를 어딘가에 모은다 | 필요하고 유용한 연산 | 문제 아님 |
| 그 저장소로 **patch token을 쓴다** | 저장하려면 원래 내용을 덮어써야 함 | **문제** |

patch token은 모델 내부의 임시 변수가 아니라 **밖으로 나가는 출력**이다. ViT의 patch token은
입력 이미지의 특정 격자 좌표 하나에 대응하고, downstream은 그 대응 관계를 신뢰한다.
그런데 모델이 그 자리를 저장소로 징발하면 저장할 공간을 만들기 위해 **그 patch의 국소 정보를
버린다**. 결과적으로 feature map 위에 "이 자리의 값은 이 자리를 설명하지 않는" 구멍이 뚫린다.
그것이 attention map과 norm map에서 배경에 튀는 점(artifact)으로 보이는 것이다.

정리하면: 계산은 옳지만 **주소가 틀렸다**. 임시 계산 결과를 하필 API 응답 필드에 써버린 셈이다.

## 3. 왜 하필 dense prediction이 무너지나

dense prediction은 **patch token 하나하나가 자기 위치를 설명한다**는 전제 위에 서 있다.

- **Semantic segmentation (ADE20k)**: 각 patch token에 linear classifier를 붙여 픽셀 단위 라벨을
  뽑고 업샘플링한다. token이 자기 자리 대신 이미지 전역 요약을 담고 있으면, 그 자리의 예측은
  "이미지 전체가 무엇인가"에 끌려가 국소적으로 틀린 라벨이 찍힌다.
- **Monocular depth estimation (NYUd)**: 깊이는 정의상 좌표별로 다른 값이다. position 정보를 잃은
  token은 자기 좌표에 해당하는 깊이를 낼 근거가 없다. Fig. 5b의 position prediction 정확도 하락이
  곧 이 능력의 손실이다.
- **Unsupervised object discovery (LOST)**: patch끼리의 유사도 그래프에서 "주변과 덜 유사한 patch"를
  seed로 잡아 객체를 확장한다. 배경에 global 정보를 담은 튀는 token이 섞이면 그 token이
  잘못된 seed가 되거나 유사도 구조를 왜곡해 알고리즘 자체가 어긋난다. 그래서 DINOv2가
  성능은 좋은데 LOST와는 유독 궁합이 나빴다.

세 과제 모두 **국소성(locality)** 에 의존하는데, 오염은 정확히 그 국소성을 지운다.
반면 이미지 분류처럼 [CLS] 하나만 쓰는 global 과제는 거의 타격을 받지 않는다.
문제가 벤치마크 평균 점수에 잘 안 잡히면서도 실재했던 이유가 이것이다.

## 4. 그래서 처방이 register다 — 역할 분리

진단이 "연산은 옳고 위치가 틀렸다"이므로, 처방은 **연산을 없애는 것이 아니라 옮기는 것**이다.
patch embedding 뒤에 [CLS]처럼 학습 가능한 토큰 N개(register)를 입력 시퀀스에 추가하고,
출력에서는 **버린다**. 모델은 전역 정보를 모을 자유로운 공간을 얻고, patch token은 더 이상
징발당하지 않는다.

![Fig. 6 — register token을 추가하고 출력에서는 버린다](fig-2.jpeg)

결과:

- 출력 patch token의 high-norm outlier가 **완전히 사라진다** (Fig. 7, Fig. 15).
  그 고-norm 행동은 register token으로 그대로 옮겨간다.
- register가 없을 때의 "정상" patch token과 register가 있을 때의 patch token은
  국소 정보량이 비슷하다 — 즉 register는 outlier 행동만 걷어낼 뿐 다른 patch를 바꾸지 않는다.
- dense 과제가 개선된다: DINOv2 ADE20k mIoU 46.6 → 47.9, NYUd rmse 0.378 → 0.366,
  LOST corloc VOC07 35.3 → 55.4 / COCO20k 26.9 → 42.0. ImageNet 분류는 84.3 → 84.8로 손해가 없다.
- attention map이 눈에 띄게 깨끗해진다.

![Fig. 1 — register 유무에 따른 attention map](fig-3.jpeg)

## 5. 한 줄 요약

전역 정보를 모으는 연산 자체는 유용하다. 문제는 그 **저장소가 출력으로 쓰이는 patch token을
덮어쓴다**는 것이고, 그 대가로 버려지는 국소 patch 정보는 semantic segmentation, depth estimation,
object discovery처럼 patch 단위 국소성에 의존하는 dense prediction에서 그대로 성능 저하로 나타난다.
register는 그 저장소를 출력 밖의 전용 토큰으로 분리해 역할 충돌을 해소한다.
