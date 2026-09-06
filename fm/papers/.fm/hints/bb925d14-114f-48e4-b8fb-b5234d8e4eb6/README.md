# Discriminative 자기지도학습의 두 갈래

**Q. Discriminative 자기지도학습 계열의 두 갈래는?**

**A.** instance-level objective(MoCo, SimCLR, BYOL, DINO 등)와 clustering 기반(DeepCluster, SwAV 등)으로 나뉜다. 표준 벤치마크에서 좋은 frozen feature를 주지만 큰 모델로 확장하기 어려웠다.

---

## 1. 출처: DINOv2 논문 2장 Related Work

DINOv2 논문(Oquab et al., 2023, arXiv 2304.07193)은 Related Work에서 자기지도학습(SSL)을 크게 두 가족으로 정리한다.

| 가족 | 학습 신호 | 대표 방법 | 특징 |
|---|---|---|---|
| **Intra-image SSL** | 이미지 *내부*에서 만든 pretext task (한 부분을 가리고 나머지로 예측) | context prediction, colorization, rotation 예측, inpainting, jigsaw, **MAE / BEiT** | 좋은 finetuning 성능. 단, frozen feature는 약해 supervised finetuning이 필요 |
| **Discriminative SSL** | 이미지 *사이*(또는 이미지 그룹 사이)의 구별 신호 | instance classification 계열, clustering 계열 | **frozen feature가 바로 쓸 만함** → DINOv2가 속하는 계열 |

이 카드는 두 번째 가족인 **discriminative SSL**이 다시 어떻게 두 갈래로 나뉘는지를 묻는다.

## 2. Discriminative SSL의 두 갈래

논문 원문(2장):

> Several improvements were made based either on **instance-level objectives** (Hénaff et al., 2019; He et al., 2020; Chen & He, 2021; Chen et al., 2020; Grill et al., 2020; Caron et al., 2021) or **clustering** (Caron et al., 2018; Asano et al., 2020; Caron et al., 2020). These methods provide performant frozen features on standard benchmarks like ImageNet, but they are hard to scale to larger model sizes (Chen et al., 2021).

### 2-1. Instance-level objective (인스턴스 판별)

- **핵심 아이디어**: "각 이미지 하나가 곧 하나의 클래스". 같은 이미지의 서로 다른 augmentation(view)은 같은 표현으로, 다른 이미지는 다른 표현으로 밀어낸다.
- **뿌리**: Hadsell et al. (2006)의 contrastive loss → Dosovitskiy et al. (2016, Exemplar CNN), Bojanowski & Joulin (2017, Noise-as-Targets), Wu et al. (2018, non-parametric instance discrimination, memory bank)으로 "instance classification"이 대중화.
- **대표 방법**(논문이 인용한 순서대로):
  - CPC v2 (Hénaff et al., 2019) — contrastive predictive coding
  - **MoCo** (He et al., 2020) — momentum encoder + negative queue
  - **SimSiam** (Chen & He, 2021) — negative 없이 stop-gradient만으로 collapse 방지
  - **SimCLR** (Chen et al., 2020) — 대배치 in-batch negative + NT-Xent loss
  - **BYOL** (Grill et al., 2020) — negative 없이 online/target 네트워크(EMA) 예측
  - **DINO** (Caron et al., 2021) — self-distillation, teacher를 EMA로 만들고 centering+sharpening으로 collapse 방지
- **분류 기준으로 보면** negative pair를 쓰는 contrastive 방식(MoCo, SimCLR)과 negative 없이 self-distillation/예측으로 가는 방식(BYOL, SimSiam, DINO)이 섞여 있지만, 모두 *"개별 이미지 인스턴스의 불변성"*을 목표로 한다는 점에서 한 갈래로 묶인다.

### 2-2. Clustering 기반

- **핵심 아이디어**: 개별 이미지가 아니라 **이미지 그룹(클러스터)**을 판별 신호로 쓴다. 특징을 클러스터링해 pseudo-label을 만들고 그것을 예측하도록 학습하는 것을 반복한다.
- **대표 방법**:
  - **DeepCluster** (Caron et al., 2018) — k-means로 pseudo-label을 만들고 분류기를 학습, 이를 교대 반복
  - **SeLa** (Asano et al., 2020) — self-labelling. 클러스터 할당을 최적수송(Sinkhorn-Knopp) 문제로 풀어 균등 분배를 강제하여 퇴화(모든 샘플이 한 클러스터)를 방지
  - **SwAV** (Caron et al., 2020) — 온라인 클러스터링. 한 view의 cluster assignment(code)를 다른 view에서 예측(“swapped prediction”), Sinkhorn-Knopp으로 code를 균등화, multi-crop 도입
- **왜 별도 갈래인가**: instance-level에서는 비교 대상이 "다른 이미지 전부"(negatives)이지만, clustering에서는 "prototype/클러스터 중심"이라 negative 비교가 배치 크기에 묶이지 않고, 의미적으로 비슷한 이미지를 같은 그룹으로 모으는 유도 편향이 있다.

### 2-3. 두 갈래는 실제로는 수렴한다

DINO는 instance-level 계열로 인용되지만 teacher 출력이 K개의 prototype 위의 softmax 분포라는 점에서 SwAV의 "prototype 예측"과 매우 가깝다. 실제로 DINOv2는 4장에서 자기 방법을 이렇게 요약한다.

> a combination of DINO and iBOT losses **with the centering of SwAV**

즉 **DINO(instance-level self-distillation) + iBOT(patch-level masked prediction) + SwAV의 Sinkhorn-Knopp centering(clustering 계열 기법)** — 두 갈래의 기법을 한 레시피로 합친 것이 DINOv2다. 이 카드의 두 갈래 구분은 DINOv2 방법론의 구성요소를 이해하는 배경이 된다.

## 3. "좋은 frozen feature를 주지만 큰 모델로 확장하기 어려웠다"

- **좋은 frozen feature**: MAE 같은 intra-image 계열은 finetuning해야 성능이 나오지만, discriminative 계열은 backbone을 고정하고 linear probe/k-NN만 해도 ImageNet에서 높은 성능이 나온다. 이것이 DINOv2가 이 계열을 택한 이유다("our features perform well out of the box").
- **큰 모델로 확장 어려움**: 논문은 Chen et al. (2021) — MoCo v3 논문 *"An Empirical Study of Training Self-Supervised Vision Transformers"* — 를 인용한다. 이 연구는 ViT를 SSL로 학습할 때 훈련 불안정(loss spike, 성능 급락)이 흔하고, 모델을 키울수록 이득이 잘 늘지 않음을 보였다. 또한 Related Work의 다음 단락에서, 지금까지의 대규모 SSL 시도(SEER 등)는 uncurated 데이터를 썼기 때문에 결과가 대개 finetuning 기준이었다고 지적한다.
- **DINOv2의 대응**: 그래서 DINOv2 기여의 대부분은 "discriminative SSL을 모델·데이터 크기에서 스케일할 때 안정화·가속"하는 기술(untied head, Sinkhorn-Knopp centering, KoLeo regularizer, FlashAttention·FSDP 등 효율화, 그리고 curated 142M 데이터셋 LVD-142M)에 집중된다. iBOT(Zhou et al., 2022a)을 기반으로 삼은 이유도 "스케일링에 특히 적합"하다고 판단했기 때문이다.

![Fig. 2: 파라미터(flops) 증가에 따른 성능 변화. 주황 삼각형=기존 SSL, 분홍=weakly-supervised, 파란 선=DINOv2](fig-1.jpeg)

위 그림(논문 Fig. 2)이 카드 마지막 문장을 시각적으로 보여준다. 8개 과제 패널 모두에서 **기존 SSL 방법(주황 삼각형)은 flops(모델 크기)가 10^10에서 10^12로 커져도 점들이 위로 올라가지 않고 흩어져 있다** — 즉 큰 모델로 확장해도 frozen feature 품질이 잘 늘지 않는다. 반면 DINOv2(파란 선)는 ViT-S → B → L → g로 갈수록 단조 증가하며, 특히 Segmentation·Monocular Depth·Instance Retrieval에서는 weakly-supervised 최고치(분홍 점선)까지 크게 넘어선다. "스케일이 어렵다"는 한계를 해결한 것이 DINOv2의 핵심 주장이다.

## 4. 암기 포인트

- **두 갈래**: (1) **instance-level** — 이미지 하나 = 클래스 하나, 뷰 간 불변성 (MoCo, SimCLR, BYOL, SimSiam, DINO) / (2) **clustering** — 그룹 단위 pseudo-label·prototype (DeepCluster, SeLa, SwAV)
- **공통 장점**: frozen feature가 바로 쓸 만함 (↔ MAE 계열은 finetuning 필요)
- **공통 한계**: 큰 모델로 스케일 어려움 (Chen et al., 2021 / MoCo v3) → DINOv2가 풀고자 한 문제
- **DINOv2 = DINO + iBOT + SwAV centering**: 두 갈래의 기법이 결합된 레시피
