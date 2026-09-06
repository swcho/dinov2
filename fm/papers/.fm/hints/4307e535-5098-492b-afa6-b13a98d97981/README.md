# DINO head와 iBOT head의 파라미터 공유(tying) vs 분리(untying)

## 한 줄 요약

iBOT 원 논문(Zhou et al., 2022a)의 ablation은 "[CLS] 토큰용 DINO head와 패치 토큰용 iBOT head의 **파라미터를 공유**하는 쪽이 더 좋다"고 보고했다. 그러나 DINOv2가 대규모(ViT-L/g, ImageNet-22k → LVD-142M, 배치 3k, 128k 프로토타입)로 학습해 보니 **반대**였다. 그래서 DINOv2는 **모든 실험에서 두 head를 분리(untying)** 해 사용한다.

---

## 1. 배경: 두 개의 head가 왜 존재하는가

DINOv2의 손실은 DINO 손실 + iBOT 손실의 조합이다(논문 §4). 각 손실은 backbone(ViT) 출력 토큰 위에 **학습 가능한 MLP projection head**를 얹고, 그 출력(prototype score)에 softmax → centering → cross-entropy를 계산한다.

| 손실 | 어떤 토큰에 적용 | head 이름 | 무엇을 비교하나 |
|---|---|---|---|
| $\mathcal{L}_{DINO} = -\sum p_t \log p_s$ | **class 토큰** (이미지 수준) | DINO head | 같은 이미지의 서로 다른 crop에서 나온 student/teacher class 토큰 |
| $\mathcal{L}_{iBOT} = -\sum_i p_{ti} \log p_{si}$ | **마스킹된 패치 토큰** (패치 수준) | iBOT head | student의 mask 토큰 vs teacher의 (보이는) 해당 패치 토큰 |

두 head는 구조가 같다(MLP → bottleneck → prototype 벡터). 그래서 "하나의 MLP를 두 손실이 같이 쓸까(공유, tying), 각자 따로 둘까(분리, untying)"라는 설계 선택이 생긴다.

### iBOT 원 논문의 입장
iBOT(Zhou et al., 2022a)은 [CLS] 토큰과 패치 토큰이 **같은 프로토타입 공간**을 공유하면 이미지 수준 의미와 패치 수준 의미가 서로 정렬된다고 보고, ablation에서 **head 공유가 더 낫다**는 결과를 얻어 기본값으로 삼았다. 파라미터 수도 줄어드는 부수 효과가 있다.

### DINOv2의 관찰
> "In Zhou et al. (2022a), an ablation study shows that sharing parameters between the DINO and iBOT heads leads to better performance. **At scale, we observed that the opposite is true**, and we therefore use two separate heads in all our experiments." (§4)

즉 iBOT의 결론은 iBOT이 실험한 규모(ViT-S/B, ImageNet-1k 수준)에서는 맞았지만, 모델·데이터·프로토타입 수를 모두 키운 DINOv2 환경에서는 **분리하는 쪽이 이겼다**. 논문은 이유를 길게 분석하지 않지만, 직관적으로는 다음처럼 설명할 수 있다.

- class 토큰(전역 의미)과 패치 토큰(국소 의미)이 요구하는 프로토타입 분포가 다르다. 작은 규모에서는 정보를 합치는 것이 정규화 효과를 주지만, 용량이 충분한 대규모에서는 하나의 head가 두 역할을 동시에 맡는 것이 오히려 **병목**이 된다.
- 프로토타입이 128k(131072)개로 매우 많아진 상황에서는 각 목적 함수가 자기만의 프로토타입 집합을 갖는 것이 더 자연스럽다.

---

## 2. Ablation 표에서의 위치 (Table 1, §6.1)

DINOv2는 iBOT 베이스라인에서 시작해 구성 요소를 하나씩 쌓아 올리는 ablation을 ViT-L / ImageNet-22k로 수행했다(k-NN 성능을 우선 최적화). **"+Untying heads"가 마지막 단계이며, 그 결과가 곧 DINOv2 최종 레시피**다.

| 단계 | INet-1k k-NN | INet-1k linear |
|---|---|---|
| iBOT (논문 수치) | 72.9 | 82.3 |
| + our reproduction | 74.5 | 83.2 |
| + LayerScale, Stochastic Depth | 75.4 | 82.0 |
| + 128k prototypes | 76.6 | 81.9 |
| + KoLeo | 78.9 | 82.5 |
| + SwiGLU FFN | 78.7 | 83.1 |
| + Patch size 14 | 78.9 | 83.5 |
| + Teacher momentum 0.994 | 79.4 | 83.6 |
| + Tweak warmup schedules | 80.5 | 83.8 |
| + Batch size 3k | 81.7 | 84.7 |
| + Sinkhorn-Knopp | 81.7 (=) | 84.7 (=) |
| **+ Untying heads = DINOv2** | **82.0 (↑0.3)** | **84.5 (↓0.2)** |

읽는 법:
- head 분리는 **k-NN +0.3**, **linear −0.2**로 효과가 크지는 않다. 하지만 DINOv2는 "linear probe 성능은 k-NN 성능에 의해 하한이 정해진다"는 경험 때문에 k-NN을 기준으로 선택했다고 명시한다.
- 이 표의 다른 항목들(LayerScale/Stochastic Depth 등)도 linear는 떨어지지만 학습 안정성을 위해 채택한 것처럼, DINOv2의 선택은 "하나의 지표에서 무조건 최고"가 아니라 대규모 학습에서의 종합적 판단이다.
- 주의: 이 표는 ViT-L, ImageNet-22k에서의 결과다. iBOT 원 논문의 ablation(더 작은 모델·데이터)에서는 반대 부호가 나왔다는 것이 카드의 핵심 포인트다.

---

## 3. 코드에서 확인하기 (이 저장소)

논문의 "untying"은 코드에서 `ibot.separate_head` 옵션으로 구현되어 있다.

- `dinov2/configs/ssl_default_config.yaml` — 기본값은 **`separate_head: false`** (iBOT 원 설정 = 공유). `dino.head_n_prototypes`와 `ibot.head_n_prototypes`가 각각 65536.
- `dinov2/configs/train/vitl14.yaml`, `vitg14.yaml` — 실제 DINOv2 학습 설정은 **`separate_head: true`**, 두 head 모두 `head_n_prototypes: 131072`(=128k), `dino.head_bottleneck_dim: 384`.
- `dinov2/train/ssl_meta_arch.py` `SSLMetaArch.__init__`:
  - `dino_head`는 항상 student/teacher에 생성된다(`DINOHead` MLP).
  - `separate_head=True`이면 같은 `DINOHead` 클래스로 **별도의 `ibot_head`** 를 student/teacher에 추가하고, iBOT 출력 차원은 `cfg.ibot.head_n_prototypes`를 쓴다.
  - `separate_head=False`이면 `"IBOT -- head shared with DINO"` 로그를 남기고 iBOT 손실도 `dino_head` 출력을 그대로 사용한다(출력 차원 = `cfg.dino.head_n_prototypes`).

즉 "공유"와 "분리"는 같은 MLP 클래스 인스턴스를 하나 만들지 둘 만들지의 차이이며, teacher 쪽 head도 각각 EMA로 따라간다.

---

## 4. 기억용 정리

- **누가 뭐라고 했나**: iBOT → "공유가 낫다". DINOv2 → "대규모에서는 반대, 분리가 낫다".
- **DINOv2의 결정**: 모든 실험에서 head 분리(untying). Table 1의 마지막 줄 "+Untying heads = DINOv2" (k-NN 82.0, linear 84.5).
- **코드 키워드**: `ibot.separate_head: true`, `student["ibot_head"]`, `teacher["ibot_head"]`.
- **교훈**: 소규모 ablation의 결론이 스케일을 키우면 뒤집힐 수 있다. DINOv2는 이런 항목들을 다시 검증해 자신의 레시피를 만들었다.
