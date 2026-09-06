# 모델 크기 × 데이터 크기의 상호작용 (DINOv2 Fig. 4, §6.3)

**Q.** 모델 크기와 데이터 크기의 상호작용(Fig. 4)에 대한 결론은?

**A.** 모델이 커질수록 ImageNet-22k(14M)보다 LVD-142M(142M) 학습의 이점이 커진다. LVD-142M의 ViT-g는 ImageNet-1k 성능은 동등하게 유지하면서 다른 벤치마크에서 크게 앞선다.

---

## 1. 실험 설정 — 무엇을 비교했나

DINOv2 논문 §6.3 "Model Size and Data"는 아주 단순한 2×3 격자 실험이다.

| 축 | 값 |
|---|---|
| **사전학습 데이터** | ImageNet-22k (약 14M 장) vs **LVD-142M** (142M 장, 논문이 제안한 큐레이션 데이터) |
| **모델 크기** | ViT-**L** (0.3B) → ViT-**H** (0.6B) → ViT-**g** (1.1B) |
| **평가** | frozen feature + 선형 분류/검색 (ImageNet-1k, ImageNet-V2, ImageNet-Sketch, Food101, Cars, AmsterTime, Oxford-H) |

즉 "데이터를 10배로 늘리는 것이 언제 도움이 되는가? 모델이 작을 때도 되는가, 클 때만 되는가?"를 묻는 실험이다.

## 2. 그림 읽기

![Fig. 4: 모델 크기(L/H/g)에 따른 성능. 주황=INet-22k, 파랑=LVD-142M](fig-1.jpeg)

각 패널의 x축은 모델 크기(L, H, g), y축은 해당 벤치마크 성능이다. **주황(▲)=ImageNet-22k, 파랑(●)=LVD-142M.** 그림에서 실제로 관찰되는 것을 벤치마크 유형별로 나누면 다음과 같다.

### (a) ImageNet 계열 — 작을 땐 INet-22k가 유리, 커지면 따라잡거나 역전
- **ImageNet-1k**: ViT-L에서는 주황이 위(약 85.1 vs 84.7), ViT-H에서도 주황이 약간 위. **ViT-g에서 두 선이 거의 한 점(≈85.9)으로 만난다.** → 답에 나오는 "ImageNet-1k 성능은 동등하게 유지"가 바로 이 수렴점이다.
- **ImageNet-V2**: L, H에서는 주황이 위지만 파랑의 기울기가 훨씬 가파르고, **g에서 파랑이 역전**(≈77.6 vs 77.1).
- **Food101**: 같은 패턴. L에서 2점 이상 뒤지던 파랑이 g에서 동등 이상으로 올라온다.

왜 작은 모델에서는 INet-22k가 유리한가? ImageNet-22k는 ImageNet-1k의 상위집합이라 평가 분포와 거의 같다. 용량이 작은 모델은 "평가 분포에 가까운 데이터"에 집중 학습하는 편이 이득이고, 142M 장의 다양한 이미지를 소화할 여력이 없다.

### (b) 분포 이동·검색 벤치마크 — LVD-142M이 항상 위, 격차가 벌어짐
- **ImageNet-Sketch**: 파랑이 처음부터 위이고 격차가 L(≈3점) → g(≈7점)으로 벌어진다.
- **AmsterTime**(장소 검색): 파랑이 항상 위, 격차가 커진다. 주황은 거의 평평.
- **Oxford-H**(랜드마크 검색, hard split): 가장 극적. 주황은 ViT-L→g로 키워도 ≈20 근처에서 **전혀 오르지 않는데**, 파랑은 ≈33→40으로 계속 상승한다.

→ ImageNet-22k만으로는 모델을 아무리 키워도 검색·도메인 이동 성능이 늘지 않는다. 데이터의 **다양성**이 병목이며, 이 병목은 모델 크기로 해결되지 않는다.

### (c) Cars — 가장 교과서적인 "교차(crossover)" 패턴
- ViT-L: 주황 ≈82 vs 파랑 ≈72로 **INet-22k가 10점 앞선다.**
- ViT-H: 두 선이 ≈80에서 교차.
- ViT-g: 파랑 ≈90 vs 주황 ≈85로 **LVD-142M이 역전.**

한 패널 안에서 "작은 모델엔 작은 데이터, 큰 모델엔 큰 데이터"라는 상호작용이 그대로 보인다.

## 3. 결론 정리 — 왜 "상호작용"인가

핵심은 두 축이 독립적이지 않다는 것이다.

1. **기울기 차이**: 거의 모든 패널에서 파랑(LVD-142M) 선의 기울기가 주황보다 가파르다. 즉 **모델 크기를 키울 때 얻는 이득이 데이터가 클 때 더 크다.** 큰 데이터는 큰 모델이 있어야 활용된다.
2. **ViT-g 시점의 결과**: LVD-142M ViT-g는 ImageNet-1k에서 INet-22k ViT-g와 **동등**하고(수렴), 나머지 벤치마크에서는 **모두 앞선다.** 논문 원문: *"As the size of models grow, training on LVD-142M becomes more beneficial than training on ImageNet-22k. For instance, a ViT-g trained on LVD-142M matches the performance on ImageNet-1k of a model trained on ImageNet-22k while significantly outperforming it on the other benchmarks."*
3. **작은 모델에서의 손해는 일시적**: ViT-L 수준에서 LVD-142M이 ImageNet 계열·Cars에서 뒤지는 것은 용량 부족 때문이고, 모델을 키우면 사라진다.

## 4. 주변 맥락과의 연결

- **Table 2 (§6.2)** 는 ViT-g 하나로 데이터 소스만 바꾼 ablation이다. 거기서도 LVD-142M은 INet-1k에서는 INet-22k와 동등(85.8 vs 85.9)하지만 Im-A, ADE-20k, Oxford-M, iNat, Places에서 모두 앞섰다. Fig. 4는 이 결과를 **모델 크기 축으로 확장**해서 "이 이득은 큰 모델일수록 커진다"를 보여준다.
- **결론 절(§7)** 에서 DINOv2 성능의 요인으로 (ii) 모델 스케일 확대, (iii) 데이터셋 확대 둘 다 Fig. 4를 근거로 인용한다. 이 그림이 "스케일링" 주장의 핵심 증거다.
- **왜 ViT-g를 학습한 뒤 distillation하는가?** 작은 모델은 큰 데이터를 직접 소화하지 못하므로(Fig. 4의 ViT-L 결과), 큰 데이터의 이득을 ViT-g에 먼저 담고 작은 모델에는 지식 증류(Fig. 5, §6.5)로 전달하는 설계가 자연스럽다.

## 5. 한 줄 암기

> **데이터 10배의 이득은 모델이 커질수록 커진다.** ViT-g@LVD-142M = ImageNet-1k는 INet-22k와 동률, 나머지는 압도. 작은 모델(ViT-L)에서는 오히려 INet-22k가 ImageNet 계열·Cars에서 앞설 수 있다 — 하지만 검색(Oxford-H, AmsterTime)·분포이동(Sketch)은 크기와 무관하게 LVD-142M이 항상 우세.
