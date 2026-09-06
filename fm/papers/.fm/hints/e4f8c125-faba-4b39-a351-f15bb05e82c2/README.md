# MAE(Masked Auto-Encoder)와 DINOv2의 핵심 차이

**한 줄 요약**: MAE가 배운 특징은 *지도 파인튜닝(supervised finetuning)* 을 거쳐야 진가가 드러나지만, DINOv2의 특징은 백본을 **frozen** 한 채로 선형 분류기 하나만 얹어도 곧바로(out of the box) 강력하게 작동한다.

DINOv2 논문(Oquab et al., 2023) 2절 Related Work의 표현을 그대로 옮기면:

> He et al. (2022) show that a masked auto-encoder (MAE) learns features that provide substantial improvements **when finetuned** on downstream tasks. ... However, their features **require supervised finetuning**, while our features **perform well out of the box**.

---

## 1. 두 방법은 무엇을 학습하는가

| | MAE (He et al., 2022) | DINOv2 (Oquab et al., 2023) |
|---|---|---|
| SSL 계열 | **Intra-image / 생성형(reconstruction)** — 이미지 안에서 신호를 뽑아 나머지를 예측하는 pretext task (inpainting 계열) | **Discriminative** — 이미지(또는 crop) 사이의 판별 신호로 학습. DINO(image-level) + iBOT(patch-level) 손실 결합 |
| 학습 목표 | 입력 패치의 약 75%를 가리고, 디코더가 가려진 **픽셀을 복원** | student/teacher 네트워크의 출력 분포를 맞추는 cross-entropy. teacher는 student의 EMA. 여기에 SwAV의 Sinkhorn-Knopp centering, KoLeo 정규화, 고해상도 마무리 학습 등 추가 |
| 사전학습 데이터 | ImageNet-1k | 자동 큐레이션한 LVD-142M (142M 이미지) |
| 특징의 성격 | 저수준 복원에 유리한 표현. 선형 분리성이 낮아 **선형 프로빙 성능이 낮고**, 전체 파인튜닝 후에 성능이 크게 뛴다 | 의미 단위로 잘 정렬된 표현. **frozen + linear probe** 만으로도 최고 수준. 파인튜닝은 "선택 사항(finetuning is optional)" |

핵심 대비는 "얼마나 잘 배우는가"보다 **"배운 특징을 어떤 방식으로 써야 하는가"**에 있다. MAE는 좋은 *초기화(initialization)* 를 제공하는 데 강하고, DINOv2는 좋은 *고정 특징 추출기(frozen feature extractor)* 를 제공하는 데 강하다.

---

## 2. 논문의 실험 수치로 보는 차이

DINOv2 논문은 모든 baseline을 **백본을 얼린 상태(frozen)** 에서 동일한 코드로 재평가했다. 이 조건에서 MAE는 전반적으로 크게 뒤처진다.

### ImageNet-1k 선형 평가 (Table 4, frozen features)

| Method | Arch. | Data | kNN | linear val | ReaL | V2 |
|---|---|---|---|---|---|---|
| MAE | ViT-H/14 | INet-1k | **49.4** | **76.6** | 83.3 | 64.8 |
| iBOT | ViT-L/16 | INet-22k | 72.9 | 82.3 | 87.5 | 72.4 |
| DINOv2 | ViT-g/14 | LVD-142M | **83.5** | **86.5** | 89.6 | 78.4 |

- MAE의 kNN 49.4%는 표에 있는 SSL 모델 중 가장 낮다. 특징 공간에서 같은 클래스가 가까이 모여 있지 않다는 뜻이며, 이것이 "MAE 특징은 그대로 쓰기 어렵다"의 정량적 표현이다.
- 참고로 MAE 원논문(He et al., 2022)에서 ViT-H를 ImageNet-1k에 **end-to-end 파인튜닝**하면 86.9%(448 해상도에서 87.8%)에 도달한다. 즉 MAE는 파인튜닝하면 강하지만 frozen에서는 약하다 — 카드가 말하는 그 성격이다.

### DINOv2는 파인튜닝해도 이득이 작다 (Table 5)

| Arch. | Res. | Linear | Finetuned | Δ |
|---|---|---|---|---|
| ViT-g/14 | 224 | 86.5 | 88.5 | +2.0 |
| ViT-g/14 | 448 | 86.7 | 88.9 | +2.2 |

논문의 해석: "observe only modest improvements with fine-tuning: this suggests that DINOv2 features **already perform well out-of-the-box**." MAE가 linear→finetune 사이에 10%p 이상 뛰는 것과 대조적으로, DINOv2는 +2%p 남짓이다. 파인튜닝 없이도 이미 대부분의 성능이 나온다는 증거다.

### 밀집(dense) 태스크에서도 같은 양상

- **Semantic segmentation (Table 10, ADE20k mIoU, frozen)**: MAE ViT-H/14 linear 33.3 / +ms 30.7 vs DINOv2 ViT-g/14 linear 49.0 / +ms 53.0.
  논문은 DINOv2의 frozen `+ms` 결과(53.0)가 **MAE를 UperNet 디코더로 완전 파인튜닝한 결과(53.6)** 와 거의 같다고 지적한다 — "훨씬 단순한 예측기"만 썼는데도 말이다.
- **Depth estimation (Table 11, NYUd RMSE, lin.1)**: MAE 0.517 vs DINOv2 ViT-g/14 0.344 (낮을수록 좋음).
- **Domain generalization (Table 6, ImageNet-A)**: MAE 10.2 vs DINOv2 ViT-g/14 75.9.
- **Fine-grained / video (Table 7, iNat2018)**: MAE 31.0 vs DINOv2 ViT-g/14 81.6.

![Fig. 2 — 모델 크기(flops)에 따른 8종 태스크 성능. 주황 삼각형이 SSL(MAE 포함), 파란 선이 DINOv2](fig-1.jpeg)

위 그림(논문 Fig. 2)에서 주황색 삼각형(SSL baseline, MAE 포함)은 모든 패널에서 파란 DINOv2 곡선 아래에 흩어져 있다. 특히 *Segmentation*, *Monocular Depth*, *ImageNet-{A,R,Sketch}*, *Instance Retrieval* 패널에서 격차가 크다 — 모두 **frozen 특징을 그대로 쓴** 평가이므로, 이 격차가 곧 "out of the box로 잘 되는가"의 차이다. 분홍 삼각형(약지도 CLIP 계열)과 점선(최고 WSL)에는 DINOv2가 근접하거나 넘는다.

---

## 3. 왜 이런 차이가 생기는가 (직관)

1. **목표 함수의 차이**. MAE는 픽셀 복원을 목표로 하므로 인코더가 색·질감·저수준 구조 정보를 많이 보존해야 한다. 이런 표현은 클래스 경계와 선형으로 정렬되어 있지 않아 linear probe/kNN이 약하다. 반면 DINO/iBOT류 discriminative 목표는 "같은 이미지의 다른 crop은 같은 prototype 분포로" 라는 불변성(invariance)을 직접 학습하므로 의미 단위로 뭉치는 특징이 나온다.
2. **패치 단위 목표의 성격**. DINOv2의 iBOT 항도 마스킹을 쓰지만, 복원 대상이 픽셀이 아니라 **teacher의 패치 토큰 분포(feature space)** 다. 그래서 patch feature가 의미적으로 정렬되어 분할·깊이 같은 dense 태스크에서도 frozen으로 바로 쓸 수 있다.
3. **데이터와 스케일**. DINOv2는 142M 큐레이션 데이터와 ViT-g(1B) 규모로 학습하고 이를 작은 모델에 distillation했다. Fig. 2에서 DINOv2 곡선이 모델 크기에 따라 계속 오르는 것이 이 효과다.

![Fig. 1 — DINOv2 패치 특징의 첫 3개 PCA 성분 시각화](fig-2.jpeg)

논문 Fig. 1은 frozen DINOv2의 patch feature에 PCA만 적용한 결과다. 학습이나 라벨 없이도 (a) 새·비행기의 날개/머리/꼬리, (b) 코끼리·가네샤 상의 머리/몸통/다리, (c) 말 사진·스케치·드로잉의 같은 부위, (d) 종이 자동차·버스·클립아트의 대응 부위가 **같은 색(=같은 PCA 성분)** 으로 맞춰진다. 자세·스타일·객체가 달라도 부위가 대응되는 이 정렬이 "out of the box"로 작동하는 특징의 실체이며, MAE의 픽셀 복원 특징에서는 이런 수준의 의미 정렬이 곧바로 나오지 않는다.

---

## 4. 기억 포인트

- **MAE = 파인튜닝 전제(good initialization)**. 마스크 75% → 픽셀 복원. frozen linear probe는 약함(IN-1k 76.6, kNN 49.4), 파인튜닝하면 강함(~87%).
- **DINOv2 = frozen 그대로(good frozen features)**. discriminative(DINO+iBOT) + 큐레이션 대규모 데이터. frozen linear 86.5%, 파인튜닝해도 +2%p 정도만 오름 → "finetuning is optional".
- 논문 자체의 문장: *"their features require supervised finetuning, while our features perform well out of the box."*

## 출처

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193 — §2 Related Work, §4, §7.1 (Table 4, 5, 6), §7.2 (Table 7), §7.4 (Table 10, 11), Fig. 1, Fig. 2.
- He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022 (MAE 파인튜닝 수치 참고).
