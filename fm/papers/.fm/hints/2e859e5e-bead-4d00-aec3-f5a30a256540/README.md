# iBOT(patch-level) objective의 동작 방식

> **Q.** iBOT(patch-level) objective의 동작 방식은?
> **A.** student에만 입력 패치 일부를 무작위 마스킹하고, student mask token은 student iBOT head에, 대응하는 (보이는) teacher patch token은 teacher iBOT head에 통과시킨다. 손실은 마스킹된 패치 인덱스 $i$에 대해 $\mathcal{L}_{iBOT} = -\sum_i p_{ti} \log p_{si}$이다.

출처: DINOv2 논문(Oquab et al., 2023, arXiv:2304.07193) §4 "Discriminative Self-supervised Pre-training" — *Patch-level objective (Zhou et al., 2022a)* 항목. 원래 iBOT(Zhou et al., 2022, "Image BERT Pre-training with Online Tokenizer")에서 제안된 목적함수를 DINOv2가 그대로 채택한 것이다.

---

## 1. 큰 그림: DINOv2 손실은 "DINO + iBOT + (SwAV centering)"

DINOv2는 **student / teacher** 두 개의 ViT를 사용하는 self-distillation 방식이다. 같은 이미지의 서로 다른 crop을 두 네트워크에 넣고, student가 teacher의 출력을 흉내내도록 학습한다. teacher는 학습되지 않고 student 파라미터의 **지수이동평균(EMA)** 으로 갱신된다.

손실은 두 수준에서 정의된다.

| 수준 | 이름 | 무엇을 비교하나 | 사용하는 토큰 | head |
|---|---|---|---|---|
| 이미지 수준 | **DINO loss** | crop 전체의 요약 표현 | `[CLS]` token | DINO head |
| **패치 수준** | **iBOT loss** | 개별 패치 위치의 표현 | patch token (마스킹된 위치) | iBOT head |

DINO loss는 "이 이미지가 전체적으로 무엇인지"를, iBOT loss는 "이 위치에 무엇이 있는지"를 학습시킨다. 카드가 묻는 것은 후자다.

## 2. iBOT loss의 다섯 단계

### (1) student 입력만 마스킹한다
global crop을 패치(DINOv2는 $14\times14$ 픽셀)로 쪼갠 뒤, **student에 들어가는 패치 중 일부를 무작위로 골라 학습 가능한 `mask_token` 벡터로 교체**한다. teacher는 **마스킹하지 않은 원본 crop**을 그대로 본다.

- 구현(`dinov2/models/vision_transformer.py`, `prepare_tokens_with_masks`): patch embedding 뒤에 `torch.where(masks, mask_token, x)`로 해당 위치의 임베딩을 `mask_token`으로 갈아 끼운다. 즉 픽셀을 지우는 것이 아니라 **토큰 자리를 "빈칸" 토큰으로 바꾸는** 것이다(BERT의 `[MASK]`와 같은 발상).
- 마스크 모양(`dinov2/data/masking.py`, `MaskingGenerator`): BEiT/iBOT식 **block-wise masking** — 개별 패치를 흩뿌려 지우는 게 아니라 종횡비 0.3~3.3 범위의 직사각형 블록을 여러 개 덮는다. 인접 패치 복사만으로 쉽게 맞출 수 없게 하기 위함이다.
- 마스킹 양(`ssl_default_config.yaml`): 배치 내 샘플의 `mask_sample_probability = 0.5`(절반)만 마스킹하고, 마스킹되는 샘플은 전체 패치의 `mask_ratio_min_max = [0.1, 0.5]` 사이 비율을 지운다.

### (2) 두 네트워크에 통과시킨다
- student: 마스킹된 crop → ViT → 각 위치의 patch token. 마스킹된 위치에서 나온 출력이 **student mask token**이다. 이 토큰은 자기 자리의 픽셀을 못 봤으므로, 주변 문맥(attention)으로 "여기에 무엇이 있어야 하나"를 추론한 결과다.
- teacher: 마스킹 안 된 crop → ViT → patch token. 이 중 **student에서 마스킹된 위치와 같은 인덱스**의 토큰만 뽑아 쓴다(코드에서 `mask_indices_list`로 `index_select`). 이 토큰은 실제 픽셀을 봤으므로 "정답에 가까운" 표현이다.

### (3) iBOT head로 prototype score를 낸다
두 patch token을 각각 **iBOT head**(3층 MLP, bottleneck 256 → 출력 $K$차원, 기본 $K=65{,}536$, ViT-g는 $131{,}072$)에 통과시켜 $K$개의 "prototype score"를 얻는다. prototype은 온라인으로 학습되는 시각 단어 사전(visual vocabulary)이라 볼 수 있고, iBOT 논문은 이 teacher+head를 **online tokenizer**라 부른다.

DINOv2의 차이점 — **head untying**: 원래 iBOT는 DINO head와 iBOT head의 파라미터를 공유하는 쪽이 좋다고 보고했지만, DINOv2는 대규모에서는 반대라고 관찰하여 **두 head를 분리**했다(§4 "Untying head weights", `ibot.separate_head: true` in `vitg14.yaml`).

### (4) softmax + centering으로 확률분포로 만든다
- student: $p_{si} = \mathrm{softmax}(z_{si} / \tau_s)$, $\tau_s = 0.1$.
- teacher: $p_{ti} = \mathrm{softmax}((z_{ti} - c) / \tau_t)$ — score에서 **center $c$** (배치 평균의 EMA)를 빼고 더 낮은 온도 $\tau_t$로 sharpening한다. 또는 DINOv2 최종 설정처럼 **Sinkhorn-Knopp** 3회 반복으로 배치 내 prototype 할당을 균등화한다(`centering: sinkhorn_knopp`).

centering/sharpening은 모든 토큰이 한 prototype으로 몰리는 **collapse**를 막기 위한 장치로, DINO loss에 쓰는 것과 같은 절차를 패치 토큰에도 적용한 것이다.

### (5) 마스킹된 위치에서만 cross-entropy를 취한다
$$
\mathcal{L}_{iBOT} = -\sum_{i \in \mathcal{M}} p_{ti} \log p_{si}, \qquad \mathcal{M} = \text{마스킹된 패치 인덱스 집합}
$$

teacher 분포 $p_{ti}$를 "soft label"로 두고 student 분포 $p_{si}$의 cross-entropy를 최소화한다. 그래디언트는 **student로만** 흐르고(teacher는 `@torch.no_grad()`), teacher는 EMA로 따라온다. 코드(`iBOTPatchLoss.forward_masked`)는 마스킹된 토큰만 모아 `−Σ t·log_softmax(s/τ_s)`를 계산하고, 이미지마다 마스킹 개수가 다르므로 `masks_weight = 1/(마스킹 개수)`로 이미지별 평균을 맞춘 뒤 배치 평균을 낸다.

## 3. 왜 이렇게 설계했나

- **MIM(masked image modeling)을 feature 공간에서**: MAE/BEiT처럼 "가려진 부분을 복원"하는 과제이지만, 픽셀이나 고정 VQ 토큰을 맞추는 대신 **EMA teacher가 만든 prototype 분포**를 맞춘다. 정답이 학습과 함께 개선되므로(online tokenizer) 고수준 의미를 가진 표적이 된다.
- **teacher는 마스킹하지 않는 이유**: teacher가 실제 내용을 봐야 "그 자리에 무엇이 있는지"에 대한 좋은 표적이 나온다. 마스킹된 student 토큰이 그것을 문맥만으로 예측하도록 강제하는 것이 학습 신호의 핵심이다.
- **패치 수준 신호의 효과**: DINOv2 §6.4 Table 3(b)에서 MIM(iBOT) 항을 제거하면 ADE-20k 분할 mIoU가 47.1 → 44.2로 약 3점 떨어진다. ImageNet 분류(85.8 → 85.3)보다 **dense prediction**에 훨씬 크게 기여한다. 반대로 KoLeo 항은 검색(Oxford-M)에 기여한다 — 두 항의 역할이 다르다.

| MIM(iBOT) 항 | INet-1k | Im-A | ADE-20k | Oxford-M |
|---|---|---|---|---|
| ✕ | 85.3 | 72.0 | 44.2 | 64.3 |
| ✓ | 85.8 | 72.8 | **47.1** | 63.9 |

## 4. DINO loss와 나란히 놓고 보기

| | DINO | iBOT |
|---|---|---|
| student 입력 | 여러 crop (global 2 + local 8) | global crop, 일부 패치 마스킹 |
| teacher 입력 | global crop (마스킹 없음) | global crop (마스킹 없음) |
| 비교 토큰 | `[CLS]` (crop 간 교차 비교) | 같은 위치의 patch token (마스킹된 위치만) |
| 손실 | $-\sum p_t \log p_s$ | $-\sum_i p_{ti}\log p_{si}$ |
| 학습하는 것 | 이미지 전체의 불변 표현 | 위치별(dense) 표현, 문맥 추론 |

두 손실은 구조가 같고(teacher soft label에 대한 student cross-entropy, EMA teacher, centering) **비교 단위가 이미지인지 패치인지**만 다르다. 이것이 카드 답의 "위와 같이 softmax와 centering을 적용" 부분이다.

## 5. 한 줄 요약
student에게만 패치를 가리고, 가린 자리의 student 출력(iBOT head 통과)이 **같은 자리의 안 가린 teacher 출력**(iBOT head 통과)의 prototype 분포를 cross-entropy로 맞추게 한다 — 마스킹된 인덱스 $i$에 대해서만 $-\sum_i p_{ti}\log p_{si}$.

## 참고
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 — §4, §6.4, Appendix B.1.
- Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer*, ICLR 2022.
- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers (DINO)*, ICCV 2021.
- 구현: `dinov2/loss/ibot_patch_loss.py`, `dinov2/train/ssl_meta_arch.py`, `dinov2/data/masking.py`, `dinov2/data/collate.py`.
