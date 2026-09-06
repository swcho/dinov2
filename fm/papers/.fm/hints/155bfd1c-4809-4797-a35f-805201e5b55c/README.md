# Depth estimation의 "DPT" 설정이란?

> **한 줄 답**: Ranftl et al.(2021)의 **DPT(Dense Prediction Transformer) 디코더**를 **동결(frozen)된 DINOv2 백본** 위에 올리고, 깊이를 **회귀(regression)** 로 직접 예측하도록 디코더만 학습하는 설정. 디코더(head)의 채널 폭은 백본의 특징 차원(ViT-S 384 / B 768 / L 1024 / g 1536)에 맞춰 키운다.

---

## 1. 어디에 나오는 설정인가 — 세 가지 depth 평가 프로토콜

DINOv2 논문 7.4절(Dense Recognition Tasks)의 **단안 깊이 추정(monocular depth estimation)** 실험은 NYUd, KITTI, 그리고 NYUd→SUN RGB-D 제로샷 전이의 3개 벤치마크에서 Li et al.(2022b, BinsFormer)의 평가 프로토콜을 따르며, 백본을 **항상 동결**한 상태로 세 가지 head를 비교한다.

| 설정 | 백본에서 꺼내는 것 | Head | 학습 목표 |
|---|---|---|---|
| **lin. 1** | 마지막 층 패치 토큰 + [CLS] 토큰을 각 패치에 concat → 4배 bilinear 업샘플 | 선형층 1개 | **분류**: 깊이 범위를 256개 균등 bin으로 나눠 분류 손실, Bhat et al.(2021, AdaBins) 식 linear normalization으로 기대값 계산 |
| **lin. 4** | 4개 층의 토큰을 concat (ViT-S/B: l={3,6,9,12}, ViT-L: {5,12,18,24}, ViT-g: {10,20,30,40}) | 선형층 1개 | lin. 1과 동일(분류) |
| **DPT** | 4개 층의 토큰(위와 같은 층) | **DPT 디코더** (Ranftl et al., 2021) | **회귀**: 깊이 값을 직접 예측 |

논문 원문(7.4절):

> **DPT:** we use the DPT decoder (Ranftl et al., 2021) on top of our frozen models and setup a regression task. We scale the size of the head following the dimension of the features for each architecture.

즉 "DPT 설정"의 핵심은 세 가지다 — (1) **어떤 head**: DPT 디코더, (2) **어떤 손실**: 회귀, (3) **head 크기**: 백본 차원에 비례해 조정.

---

## 2. DPT(Dense Prediction Transformer)의 구조

DPT는 Ranftl, Bochkovskiy, Koltun의 *"Vision Transformers for Dense Prediction"*(ICCV 2021)에서 제안된, **ViT 인코더를 U-Net/RefineNet 스타일의 conv 디코더와 결합**하는 밀집 예측(dense prediction) 아키텍처다. 원래는 MiDaS 계열 단안 깊이 추정과 ADE20K 세그멘테이션에 쓰였다. 구조는 크게 두 단계로 나뉜다.

### 2-1. Reassemble: 토큰 시퀀스 → 다중 해상도 특징맵

ViT는 모든 층에서 **같은 해상도(H/14 × W/14 격자)의 토큰**을 내보내므로, CNN처럼 "얕은 층은 고해상도, 깊은 층은 저해상도"라는 피라미드가 없다. DPT는 이를 **인위적으로** 만들어 준다. 백본의 서로 다른 4개 층(예: ViT-L에서 l={5,12,18,24})의 토큰을 각각 다음 3단계로 처리한다.

1. **Read** — [CLS] 토큰(readout token)을 어떻게 다룰지 정한다. 무시(`ignore`), 패치 토큰에 더함(`add`), 또는 각 패치 토큰에 concat한 뒤 선형 투영(`project`). DINOv2 공개 코드의 DPT head는 `readout_type="project"`를 쓴다 — 전역 문맥을 담은 [CLS]를 각 패치에 섞어 준다.
2. **Concatenate** — N개 토큰을 원래 위치대로 (H/p) × (W/p) × D 2-D 격자로 다시 배열한다.
3. **Resample** — 1×1 conv로 채널을 바꾸고, 층마다 다른 배율로 공간 해상도를 조정한다. 얕은 층은 **전치 conv로 4배/2배 업샘플**(고해상도), 가장 깊은 층은 **stride-2 conv로 절반 다운샘플**(저해상도). 그 결과 입력 해상도의 1/4, 1/8, 1/16, 1/32에 해당하는 4단계 특징 피라미드가 만들어진다.

### 2-2. Fusion: RefineNet 방식의 점진적 업샘플링

4단계 피라미드를 **가장 저해상도(가장 깊은 층)부터** 시작해 위쪽으로 합쳐 올라간다. 각 **Feature Fusion Block**은 RefineNet의 구성을 따른다.

- 현재까지의 출력과 다음(더 얕은/고해상도) 층 특징을 각각 **Residual Conv Unit**(pre-activation 잔차 conv 두 개)에 통과시킨 뒤 합산
- 2배 bilinear 업샘플
- 다음 블록으로 전달

이를 4번 반복하면 입력의 1/2 해상도 정도의 정밀한 특징맵이 얻어지고, 마지막에 **깊이 head**(3×3 conv → 업샘플 → conv → ReLU)가 픽셀별 깊이 값 1채널을 출력한다. DINOv2 코드에서 이 fusion 경로의 폭은 `channels=256`으로 고정된다.

```
ViT (frozen)  ──층 l1──▶ Reassemble(×4 업)  ──┐
              ──층 l2──▶ Reassemble(×2 업)  ──┤   Fusion ←─ Fusion ←─ Fusion ←─ Fusion
              ──층 l3──▶ Reassemble(×1)     ──┤   (1/4)      (1/8)     (1/16)     (1/32)
              ──층 l4──▶ Reassemble(×1/2 다운)─┘      │
                                                     ▼  depth head (conv + upsample + ReLU)
                                                  픽셀별 깊이(연속값)
```

**왜 이런 구조인가?** 선형 head(lin. 1/4)는 패치 격자(14픽셀 단위) 해상도에서 예측한 뒤 bilinear로 뻥튀기하므로 경계가 뭉개진다. DPT는 여러 층의 정보를 **학습된 conv로 합치며 해상도를 복원**하므로, 물체 경계처럼 세밀한 구조를 훨씬 잘 그린다. 동시에 백본 자체는 건드리지 않으므로 "백본 특징이 얼마나 좋은가"를 재는 목적은 유지된다.

---

## 3. 왜 lin. 1/4은 분류인데 DPT는 회귀인가

- **lin. 1 / lin. 4가 분류인 이유**: 선형층 하나로 깊이를 직접 회귀시키면 학습이 불안정하고 성능이 나쁘다. AdaBins(Bhat et al., 2021) 이후 표준이 된 방법은 깊이 범위를 **256개 bin으로 나눠 softmax 분류**하고, bin 중심값의 확률 가중 평균(linear normalization)으로 연속 깊이를 복원하는 것이다. 선형 프로브의 표현력 한계를 손실 설계로 보완하는 셈이다.
- **DPT가 회귀인 이유**: DPT는 원 논문(Ranftl et al., 2021)부터 **연속 깊이 값을 직접 출력하는 회귀 모델**로 설계되었고(마지막이 ReLU로 끝나는 1채널 출력), 충분히 큰 conv 디코더가 있어 bin 분류 같은 우회가 필요 없다. DINOv2는 DPT를 "원래 쓰이던 방식대로" 가져다 쓴 것이다. DINOv2 공개 코드의 depth 파이프라인에서는 SigLoss(scale-invariant log 손실, Eigen et al. 계열) + GradientLoss 조합이 회귀 손실로 쓰인다.
- 결과적으로 표의 세 열은 "**선형 분류 프로브 → 더 두터운 선형 분류 프로브 → 본격적 conv 회귀 디코더**"로 head의 용량이 커지는 순서이고, 백본은 셋 모두 동결이다.

---

## 4. "head 크기를 아키텍처별 특징 차원에 맞춰 조정"의 의미

DPT의 Reassemble 단계는 백본 토큰 차원 D를 입력으로 받는다. 원래 DPT 논문은 ViT-B(768)/ViT-L(1024) 하이브리드용으로 채널을 [96, 192, 384, 768] 등으로 고정했지만, DINOv2는 ViT-S(384)부터 ViT-g(1536)까지 **차원이 4배나 차이 나는** 백본을 모두 평가해야 한다. 작은 백본에 큰 head를 붙이면 head 자체의 학습 능력이 결과를 왜곡하고, 큰 백본에 작은 head를 붙이면 1536차원 정보를 96채널로 병목시켜 버린다. 그래서 head의 Reassemble 채널을 **D에 비례**하게 잡는다. DINOv2 공개 코드(`dinov2/hub/depthers.py`)에서는 다음처럼 구현된다.

```python
post_process_channels = [embed_dim // 2 ** (3 - i) for i in range(4)]   # D/8, D/4, D/2, D
```

| 백본 | 특징 차원 D | 4개 층 인덱스 | Reassemble 채널 (D/8, D/4, D/2, D) |
|---|---|---|---|
| ViT-S/14 | 384 | {3, 6, 9, 12} | 48, 96, 192, 384 |
| ViT-B/14 | 768 | {3, 6, 9, 12} | 96, 192, 384, 768 |
| ViT-L/14 | 1024 | {5, 12, 18, 24} | 128, 256, 512, 1024 |
| ViT-g/14 | 1536 | {10, 20, 30, 40} | 192, 384, 768, 1536 |

(ViT-B의 [96,192,384,768]이 바로 원 DPT의 기본값이며, 다른 아키텍처는 이를 비율대로 늘리거나 줄인 것이다. fusion 경로 폭 256은 공통.) "head 크기 조정"은 이렇게 **디코더 앞단의 채널 폭을 D에 맞춰 스케일링**한다는 뜻이며, 백본 층 수가 달라 4개 층을 뽑는 인덱스도 함께 바뀐다.

---

## 5. 왜 백본을 동결한 채 DPT를 올리는가 — 특징 품질의 검증

DINOv2의 전체 평가 철학은 "**동결된 특징(frozen features)만으로 얼마나 가는가**"다(7절 서두, Table 11 제목도 *Depth estimation with frozen features*). DPT 설정에서도 학습되는 것은 디코더뿐이다. 이것이 의미하는 바:

- 성능 차이는 전부 **백본이 뽑아 준 패치 특징의 품질**에서 나온다. 같은 DPT 디코더를 OpenCLIP, MAE, DINO, iBOT 위에도 똑같이 올렸으므로 공정한 비교가 된다.
- 백본을 fine-tune하면 (a) 백본이 태스크에 맞게 변형되어 "범용 특징"인지 알 수 없고, (b) 1.1B 파라미터를 다시 학습하는 비용이 든다. 동결하면 세그멘테이션 실험(ViT-Adapter + Mask2Former, 66% 파라미터 동결)처럼 훨씬 가벼운 학습으로 끝난다.
- 논문은 이 조건에서도 **"our model, with the DPT decoder and frozen backbone, matches or exceeds the performance of the recent work of Li et al. (2022b)"** — 즉 end-to-end로 학습한 당시 SOTA(BinsFormer)에 동결 백본으로 맞먹거나 넘는다고 강조한다.

---

## 6. Table 11의 DPT 열 수치 읽기 (RMSE, 낮을수록 좋음)

괄호 안은 Li et al.(2022b) SOTA 참고값. 각 데이터셋마다 lin. 1 / lin. 4 / DPT 세 열이 있다.

| Method | Arch. | NYUd (0.330) lin.1 | lin.4 | **DPT** | KITTI (2.10) lin.1 | lin.4 | **DPT** | NYUd→SUN RGB-D (0.421) lin.1 | lin.4 | **DPT** |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenCLIP | ViT-G/14 | 0.541 | 0.510 | 0.414 | 3.57 | 3.21 | 2.56 | 0.537 | 0.476 | 0.408 |
| MAE | ViT-H/14 | 0.517 | 0.483 | 0.415 | 3.66 | 3.26 | 2.59 | 0.545 | 0.523 | 0.506 |
| DINO | ViT-B/8 | 0.555 | 0.539 | 0.492 | 3.81 | 3.56 | 2.74 | 0.553 | 0.541 | 0.520 |
| iBOT | ViT-L/16 | 0.417 | 0.387 | 0.358 | 3.31 | 3.07 | 2.55 | 0.447 | 0.435 | 0.426 |
| DINOv2 | ViT-S/14 | 0.449 | 0.417 | 0.356 | 3.10 | 2.86 | 2.34 | 0.477 | 0.431 | 0.409 |
| DINOv2 | ViT-B/14 | 0.399 | 0.362 | 0.317 | 2.90 | 2.59 | 2.23 | 0.448 | 0.400 | 0.377 |
| DINOv2 | ViT-L/14 | 0.384 | 0.333 | 0.293 | 2.78 | 2.50 | 2.14 | 0.429 | 0.396 | 0.360 |
| DINOv2 | **ViT-g/14** | 0.344 | 0.298 | **0.279** | 2.62 | 2.35 | **2.11** | 0.402 | 0.362 | **0.338** |

관찰 포인트:

1. **head 용량 순서대로 좋아진다**: 모든 백본에서 lin. 1 > lin. 4 > DPT(RMSE 감소). DINOv2-g의 NYUd는 0.344 → 0.298 → 0.279. DPT 디코더가 해상도 복원과 다층 융합을 해 주는 효과다.
2. **동결 백본 + DPT로 SOTA 도달**: DINOv2-g DPT는 NYUd **0.279 < 0.330**(SOTA), SUN RGB-D 전이 **0.338 < 0.421**로 SOTA를 넘고, KITTI **2.11 ≈ 2.10**으로 사실상 동률. 백본을 한 번도 depth로 학습시키지 않았다는 점을 감안하면 매우 강한 결과다.
3. **작은 DINOv2도 큰 경쟁 모델을 이긴다**: DINOv2 ViT-S/14(21M) DPT 0.356은 OpenCLIP ViT-G/14(1.8B) DPT 0.414, MAE ViT-H 0.415보다 낫다. iBOT ViT-L(0.358)이 OpenCLIP ViT-G(0.414)보다 좋다는 것도 저자들이 짚는 부분 — 캡션 기반 약지도 학습은 깊이 같은 **미세한 기하 정보**를 잘 못 배운다는 직관을 뒷받침한다.
4. **도메인 전이**: NYUd(실내)로 학습한 DPT head를 SUN RGB-D에 그대로 적용해도 0.338로 SOTA(0.421)를 크게 앞선다. 특징이 도메인을 넘어 잘 옮겨 간다는 증거.

---

## 7. 정성적 결과로 보는 depth 벤치마크

![Figure 7: 동결 OpenCLIP-G vs DINOv2-g 위 선형 프로브의 세그멘테이션/깊이 결과](fig-1.jpeg)

논문 Figure 7. 주의: 이 그림은 **선형 프로브**(DPT가 아님) 결과지만, Table 11의 세 벤치마크가 어떤 장면인지와 "백본 특징 품질이 깊이 예측에 미치는 영향"을 직관적으로 보여 준다.

- 2~4번째 행이 각각 **NYUd(실내), SUN RGB-D(실내, 제로샷 전이), KITTI(도로 주행)**. 가운데 열이 OpenCLIP-G, 오른쪽이 DINOv2-g.
- OpenCLIP 열은 깊이맵에 노이즈와 얼룩이 많고, SUN RGB-D 오른쪽 예시의 **흔들의자**는 거의 사라져 있다. DINOv2 열은 같은 선형 head로도 의자·식탁 형태가 또렷하고 표면이 매끈하다 — 논문은 이를 "much smoother depth estimation, with less artifacts"라고 기술한다.
- 이 선형 head가 패치 격자 해상도의 한계로 경계가 뭉개지는 것을 보면, 왜 DPT 디코더(다층 reassemble + 점진적 업샘플)를 얹으면 RMSE가 0.344 → 0.279로 더 내려가는지 감이 온다.

---

## 8. 기억용 요약

- **DPT 설정 = 동결 DINOv2 + DPT 디코더(Ranftl et al., 2021) + 회귀 손실.**
- DPT 디코더: 4개 층 토큰 → **Reassemble**(read/concat/resample로 1/4~1/32 피라미드) → **RefineNet식 Fusion**(잔차 conv + 2배 업샘플 반복) → depth head.
- lin. 1/4은 256-bin **분류**, DPT는 원 설계대로 **회귀**.
- "head 크기 조정" = Reassemble 채널을 백본 차원 D(384/768/1024/1536)에 비례(D/8, D/4, D/2, D)해 스케일링.
- 결과: DINOv2-g DPT, NYUd RMSE **0.279**(SOTA 0.330), KITTI 2.11(2.10), SUN RGB-D 0.338(0.421) — 백본 동결만으로 SOTA 수준.

---

### 참고
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 — 7.4절 Depth estimation, Table 11, Figure 7.
- Ranftl, Bochkovskiy, Koltun, *Vision Transformers for Dense Prediction*, ICCV 2021 (DPT).
- Bhat et al., *AdaBins*, CVPR 2021 (bin 분류 + linear normalization).
- Li et al., *BinsFormer*, 2022 (평가 프로토콜 및 SOTA 참고값).
- DINOv2 공개 코드 `dinov2/hub/depthers.py`, `dinov2/eval/depth/models/decode_heads/dpt_head.py` (채널 스케일링, readout=project, SigLoss/GradientLoss).
