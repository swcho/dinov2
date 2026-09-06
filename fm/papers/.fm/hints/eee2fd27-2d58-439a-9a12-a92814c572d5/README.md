# DINOv2 depth estimation의 "lin. 1" 설정

> **Q.** depth estimation의 "lin. 1" 설정은 어떻게 구성되나?
> **A.** frozen transformer의 마지막 층을 뽑아 [CLS] token을 각 patch token에 concat하고 4배 bilinear 업샘플링한다. 깊이 범위를 256개 균등 bin으로 나눈 classification loss로 선형 층 하나를 학습한다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193v2), §7.4 "Dense Recognition Tasks" — **Depth estimation** 단락과 Table 11.

---

## 1. 실험의 목적 — "frozen feature가 깊이를 얼마나 담고 있는가"

DINOv2 논문 §7.4는 **backbone을 전혀 학습시키지 않은 채(frozen)** patch-level feature 위에 최소한의 예측기만 얹어 dense task(분할, 단안 깊이 추정)를 얼마나 잘 푸는지를 측정한다. 예측기가 단순할수록 성능은 예측기가 아니라 feature 자체의 품질을 반영한다. 깊이 추정에서는 세 가지 설정을 비교한다.

| 설정 | 사용 층 | 예측기 | 학습 방식 |
|---|---|---|---|
| **lin. 1** | 마지막 1개 층 | 선형 층 1개 | 256-bin 분류 |
| **lin. 4** | 4개 층 concat | 선형 층 1개 | 256-bin 분류 (lin. 1과 동일 프로토콜) |
| **DPT** | 여러 층 | DPT 디코더 (Ranftl et al., 2021) | 회귀(regression) |

벤치마크는 **NYUd**(실내), **KITTI**(실외 주행), 그리고 **NYUd → SUN RGB-D zero-shot 전이**(NYUd로 학습한 헤드를 SUN RGB-D에 그대로 적용) 세 가지이며, 평가 프로토콜은 Li et al. (2022b, BinsFormer)을 따른다. 지표는 RMSE(낮을수록 좋음).

---

## 2. lin. 1 파이프라인을 단계별로

논문 원문(§7.4):

> **lin. 1:** we extract the last layer of the frozen transformer and concatenate the [CLS] token to each patch token. Then we bi-linearly upsample the tokens by a factor of 4 to increase the resolution. Finally we train a simple linear layer using a classification loss by dividing the depth prediction range in 256 uniformly distributed bins and use a linear normalization following Bhat et al. (2021).

이를 텐서 모양과 함께 풀어 쓰면 다음과 같다. 입력 이미지 $H \times W$, 패치 크기 14, 임베딩 차원 $D$ (ViT-g는 1536).

### 2-1. Frozen transformer의 **마지막 층** 출력 추출
- backbone은 gradient를 받지 않는다. 출력은 patch token 격자 $\tfrac{H}{14} \times \tfrac{W}{14} \times D$ 와 [CLS] token $1 \times D$ 하나.
- "마지막 층 하나만" 쓰는 것이 lin. 1의 정의이며, 이것이 lin. 4와의 유일한 차이다.

### 2-2. [CLS] token을 **모든 patch token에 concat** → 채널 $2D$
- [CLS] token을 공간 방향으로 복제(broadcast)하여 각 위치의 patch token 뒤에 이어 붙인다: $\tfrac{H}{14} \times \tfrac{W}{14} \times 2D$.
- **왜?** patch token은 "이 패치가 무엇처럼 보이는가"라는 **국소** 정보가 강하다. 그러나 단안 깊이는 본질적으로 전역 문맥에 의존한다 — 같은 크기의 나무 패치라도 실내 방 사진인지 도로 사진인지, 카메라가 얼마나 넓은 장면을 보는지에 따라 절대 깊이(미터)가 완전히 달라진다. [CLS] token은 이미지 전체를 요약한 **전역 문맥/장면 스케일** 벡터이므로, 이를 붙여주면 선형 층 하나로도 "장면 종류에 따른 깊이 스케일 보정"을 흉내낼 수 있다. 선형 층은 위치 간 정보를 섞을 수 없기 때문에(1×1 연산), 전역 정보를 채널에 미리 넣어주는 것이 유일한 방법이다.
- 공개 구현(`dinov2/hub/depthers.py`)에서도 헤드 입력 채널이 `embed_dim * len(layers) * 2`로 잡혀 있어, `*2`가 바로 이 [CLS] concat에 해당한다.

### 2-3. **4배 bilinear 업샘플링**
- $\tfrac{H}{14} \times \tfrac{W}{14}$ 격자를 bilinear 보간으로 $\tfrac{4H}{14} \times \tfrac{4W}{14}$ 로 키운다(채널 $2D$ 유지).
- **왜?** 패치 14의 ViT는 공간 해상도가 원본의 1/14로 매우 거칠다(예: 480×640 → 약 34×45). 깊이 맵은 픽셀 단위 출력이므로, 물체 경계를 어느 정도 살리려면 해상도를 복원해야 한다. 학습 파라미터가 없는 bilinear 보간으로 4배 키운 뒤(약 137×182) 선형 층을 적용하고, 최종 평가 해상도까지는 다시 보간한다. 업샘플링을 선형 층 **앞에** 두는 이유는 선형 층이 위치별로 독립이라 앞뒤 순서가 수학적으로 거의 동치이지만, 보간된 중간 위치 feature에서 예측하면 경계가 좀 더 부드럽게 나오기 때문이다.
- 구현(`linear_head.py`, `input_transform="resize_concat"`)에서는 `resize(..., size=[s * upsample ...], mode="bilinear")` 후 `torch.cat`.

### 2-4. **선형 층 하나** + **256-bin 분류 loss**
- 선형 층: 각 위치의 $2D$ 벡터를 256차원 logit으로 보내는 $W \in \mathbb{R}^{256 \times 2D}$, $b \in \mathbb{R}^{256}$ (구현에서는 BatchNorm 뒤 1×1 conv). 학습되는 파라미터는 이것이 전부다.
- 깊이 범위 $[d_{\min}, d_{\max}]$ (NYUd: 0.001–10 m, KITTI: 0.001–80 m)를 **256개 균등 간격 bin**으로 나눈다(`bins_strategy="UD"`, uniform distribution; `torch.linspace`).
- **Linear normalization (Bhat et al., 2021 = AdaBins 방식)**: 256개 logit에 ReLU를 취하고 작은 $\epsilon$(=0.1)을 더한 뒤 합이 1이 되도록 나눠 확률 $p_k$를 만든다(softmax 대신). 최종 깊이는 bin 중심 $c_k$의 **기대값** $\hat d = \sum_{k=1}^{256} p_k c_k$ 로 복원한다(`einsum("ikmn,k->imn", logit, bins)`).
- **왜 회귀 대신 분류인가?**
  1. **관행/공정 비교**: 평가 프로토콜의 기준인 Li et al. (2022b, BinsFormer)과 그 전신 AdaBins(Bhat et al., 2021)가 모두 "bin 분류 + 기대값" 방식을 쓴다. 같은 출력 형식이면 헤드 설계 차이가 아니라 feature 차이를 본다.
  2. **다중 모드(multi-modal) 분포 표현**: 물체 경계 픽셀은 "2 m 아니면 5 m"처럼 두 후보가 공존한다. 단일 스칼라 회귀는 그 중간값 3.5 m로 뭉개버리지만, 분포를 출력하면 두 봉우리를 유지한 채 불확실성을 표현할 수 있다.
  3. **학습 안정성**: 깊이는 0.001–80 m처럼 범위가 넓고 오차의 스케일이 픽셀마다 다르다. 출력을 유계(bounded)인 확률로 만들고 bin 중심의 가중 평균으로 깊이를 얻으면, 회귀에서 흔한 큰 gradient 폭주나 범위 밖 예측이 구조적으로 사라진다(예측은 항상 $[d_{\min}, d_{\max}]$ 안).
  4. **선형 층에 유리**: 분류 출력은 256개의 "템플릿" 방향을 학습하는 것이므로, 표현력이 약한 선형 층이 비선형 깊이 함수를 조각별로 근사하는 효과가 있다.
- 참고: 공개 구현의 실제 loss는 기대값 $\hat d$에 대한 SigLoss(scale-invariant log 오차)이며 논문은 이를 통틀어 "classification loss(분류 형태의 헤드)"로 서술한다. 카드 답안 수준에서는 "256 균등 bin 분류 + 선형 정규화"로 기억하면 충분하다.

---

## 3. lin. 4, DPT와의 비교

- **lin. 4**: 프로토콜은 lin. 1과 완전히 같고, 단지 **4개 층의 token을 concat**한다 — ViT-S/B는 $l=\{3,6,9,12\}$, ViT-L은 $\{5,12,18,24\}$, ViT-g는 $\{10,20,30,40\}$. 각 층마다 [CLS]도 함께 concat되므로 채널은 $4 \times 2D = 8D$. 중간 층에는 마지막 층에서 희석된 저수준 기하 정보(에지, 텍스처 기울기)가 남아 있어 깊이에 도움이 된다.
- **DPT**: 학습 가능한 다층 컨볼루션 디코더(Ranftl et al., 2021)를 frozen backbone 위에 얹고 **회귀**로 학습한다. 헤드 크기는 backbone feature 차원에 맞춰 스케일한다. 예측기가 훨씬 강하므로 절대 성능은 가장 좋지만, feature 자체의 선형 분리 가능성을 보는 지표로는 lin. 1이 가장 "순수"하다.

### Table 11 — frozen feature 위 깊이 추정 RMSE (낮을수록 좋음)

괄호 안은 Li et al. (2022b)의 SOTA(전체 fine-tuning) 참고치: NYUd 0.330 / KITTI 2.10 / SUN RGB-D 0.421.

| Method | Arch. | NYUd lin.1 | lin.4 | DPT | KITTI lin.1 | lin.4 | DPT | NYUd→SUN lin.1 | lin.4 | DPT |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenCLIP | ViT-G/14 | 0.541 | 0.510 | 0.414 | 3.57 | 3.21 | 2.56 | 0.537 | 0.476 | 0.408 |
| MAE | ViT-H/14 | 0.517 | 0.483 | 0.415 | 3.66 | 3.26 | 2.59 | 0.545 | 0.523 | 0.506 |
| DINO | ViT-B/8 | 0.555 | 0.539 | 0.492 | 3.81 | 3.56 | 2.74 | 0.553 | 0.541 | 0.520 |
| iBOT | ViT-L/16 | 0.417 | 0.387 | 0.358 | 3.31 | 3.07 | 2.55 | 0.447 | 0.435 | 0.426 |
| DINOv2 | ViT-S/14 | 0.449 | 0.417 | 0.356 | 3.10 | 2.86 | 2.34 | 0.477 | 0.431 | 0.409 |
| DINOv2 | ViT-B/14 | 0.399 | 0.362 | 0.317 | 2.90 | 2.59 | 2.23 | 0.448 | 0.400 | 0.377 |
| DINOv2 | ViT-L/14 | 0.384 | 0.333 | 0.293 | 2.78 | 2.50 | 2.14 | 0.429 | 0.396 | 0.360 |
| DINOv2 | **ViT-g/14** | **0.344** | **0.298** | **0.279** | **2.62** | **2.35** | **2.11** | **0.402** | **0.362** | **0.338** |

읽는 법:
- **lin. 1만으로도** DINOv2 ViT-g는 NYUd 0.344로, 모든 baseline의 lin. 4·DPT(iBOT DPT 0.358, OpenCLIP DPT 0.414)보다 낮다. 즉 선형 층 하나로 뽑아낸 DINOv2 깊이가 다른 모델이 학습형 디코더를 써서 얻는 것보다 정확하다.
- lin. 1 → lin. 4 → DPT 순으로 일관되게 개선되지만(ViT-g: 0.344 → 0.298 → 0.279), 개선 폭보다 **모델 간 격차**가 더 크다. 이것이 "feature 품질이 예측기보다 중요하다"는 논문 메시지의 근거다.
- DINOv2 ViT-g의 lin. 4(NYUd 0.298)는 이미 SOTA 참고치 0.330을 넘고, DPT(0.279)는 KITTI(2.11 vs 2.10)에서도 SOTA에 근접한다 — backbone은 frozen인 채로.
- iBOT ViT-L이 OpenCLIP ViT-G(더 큰 모델)를 이긴다. 캡션 기반 학습이 깊이처럼 미세한 기하 패턴을 잘 못 배운다는 논문의 해석을 뒷받침한다.
- NYUd → SUN RGB-D zero-shot 열: 실내 NYUd로 학습한 lin. 1 헤드가 다른 데이터셋에서도 0.402로 잘 전이된다.

---

## 4. 정성 결과 — Figure 7

![Figure 7: frozen OpenCLIP-G vs DINOv2-g 위 선형 probe의 분할·깊이 결과](fig-1.jpeg)

논문 Figure 7(16쪽). 2~4행이 깊이 추정(NYUd, SUN-RGBd, KITTI)이며, 각 데이터셋에 대해 **Input / OpenCLIP-G / DINOv2-g** 세 열이 짝을 이룬다. 두 모델 모두 frozen backbone에 **선형 헤드(lin. 설정)**를 얹은 결과다. 그림에서 관찰되는 점:

- **OpenCLIP-G 열**은 벽·바닥 같은 평탄한 영역에 얼룩진 고주파 잡음이 가득하고, 의자·테이블 다리 같은 얇은 구조가 배경과 섞여 사라진다. 논문이 지적한 "SUN RGB-D 이미지의 의자가 OpenCLIP에서는 완전히 무시된다"는 부분이 3행 오른쪽 예시(흔들의자)에서 보인다.
- **DINOv2-g 열**은 같은 선형 헤드인데도 깊이 맵이 매끄럽고, 소파·식탁·의자의 윤곽이 뚜렷하며 원근에 따라 색이 연속적으로 변한다. 이것이 lin. 1/lin. 4가 "예측기 능력"이 아니라 "feature에 이미 담긴 깊이 정보"를 측정한다는 주장의 시각적 증거다.
- KITTI(4행)에서는 도로가 아래에서 위로 갈수록 멀어지는 점진적 깊이, 양쪽 나무·차량의 경계가 DINOv2에서 더 선명하다.
- 두 모델 모두 깊이 라벨로 학습된 적이 없는데도 선형 층만으로 깊이가 어느 정도 분리된다는 점(논문: "able to linearly separate complex information such as depth")과, 그 품질 차이가 Table 11의 수치 격차(NYUd lin. 1: 0.541 vs 0.344)와 일치한다는 점을 같이 기억하면 좋다.

---

## 5. 한 줄 요약 (암기용)

**마지막 층 → [CLS]를 각 patch에 concat(전역 문맥 주입, $2D$) → 4배 bilinear 업샘플(해상도 복원) → 선형 층 1개 → 256 균등 bin 분류 + 선형 정규화 → bin 중심 기대값으로 깊이.** lin. 4는 층 4개 concat, DPT는 학습형 디코더+회귀. DINOv2 ViT-g는 lin. 1만으로 NYUd RMSE 0.344로 모든 baseline의 DPT를 이긴다.
