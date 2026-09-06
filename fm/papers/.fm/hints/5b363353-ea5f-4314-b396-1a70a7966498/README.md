# DINOv2의 고해상도 적응(resolution adaptation) 전략

**Q.** DINOv2의 고해상도 적응(resolution adaptation) 전략은?

**A.** 분할·검출 같은 픽셀 수준 태스크에는 고해상도가 중요하지만, 처음부터 고해상도로 학습하면 시간·메모리 비용이 크다. 그래서 사전학습 **막바지에 짧게** $518\times518$ 해상도로 올려 학습한다.

---

## 1. 왜 고해상도가 필요한가

DINOv2 논문(Oquab et al., 2023) 4장 "Discriminative Self-supervised Pre-training"의 마지막 구성요소가 **Adapting the resolution**이다. 논문의 근거는 단순하다.

- 시맨틱 분할, 객체 검출처럼 **픽셀(패치) 수준**의 출력을 내야 하는 태스크에서는 이미지 해상도가 곧 특징의 공간 분해능이다.
- 저해상도($224\times224$)에서는 **작은 물체가 패치 하나에 뭉개져 사라져** 버린다. ViT-L/14 기준 $224\times224$ 입력은 $16\times16=256$개 패치에 불과하다.
- 반면 $518\times518$은 $37\times37=1369$개 패치로, 같은 장면을 약 5.3배 촘촘하게 본다.

하지만 ViT의 self-attention 비용은 토큰 수의 제곱에 비례하므로, 처음부터 고해상도로 학습하면 **시간과 메모리가 폭증**한다. 논문 6.6절 ablation에서는 416 해상도 학습이 224 대비 약 **3배**의 연산량이 든다고 보고한다.

## 2. 해법: "짧은 고해상도 마무리 단계"

DINOv2는 두 가지 극단(전부 저해상도 / 전부 고해상도)의 절충으로, **저해상도로 대부분을 학습한 뒤 마지막에 잠깐 고해상도로 이어서 학습**한다.

| 항목 | 값 (부록 B.2, Table 16) |
|---|---|
| 본 사전학습 | $224\times224$ 글로벌 크롭, 625k iteration |
| 고해상도 적응 | 사전학습 가중치로 초기화 → $518\times518$ 에서 **10k iteration** 추가 학습 |
| 절차 | 원래 사전학습과 **완전히 같은 손실·스케줄**을 10k 스텝 안에 압축(compressed) |
| 하이퍼파라미터 | 모두 동일, 단 **base learning rate만 낮춤** |

즉 전체 학습의 약 1.6%(10k/625k)만 고해상도로 돌리는 셈이라 추가 비용이 매우 작다. 이 발상은 Touvron et al. (2019)의 "Fixing the train-test resolution discrepancy"(FixRes)에서 왔고, UniViT(Likhomanenko et al., 2021)·FlexiViT(Beyer et al., 2023)의 가변 해상도 학습과도 맥이 같다고 논문은 밝힌다.

## 3. 근거 실험 — Fig. 6 "Role of resolution"

![Fig. 6: 학습 해상도 224 / 416 / 224→416 모델의 평가 해상도별 성능](fig-1.jpeg)

논문 6.6절은 고해상도 학습이 비싸기 때문에 작은 세팅(ImageNet-1k로 학습한 ViT-L/16)에서 세 모델을 비교한다.

- **224**(주황 점선): 처음부터 끝까지 $224$로 학습
- **416**(빨강 점선): 처음부터 끝까지 $416$으로 학습 — 약 3배 비용
- **224→416**(파랑 실선): $224$로 학습 후 $416$에서 10k iteration만 추가 학습 — 실제 DINOv2 전략의 축소판

그림에서 관찰되는 점:

1. **왼쪽(ImageNet-1k 선형 분류 정확도)**: 224 모델은 평가 해상도 336 근처(약 82.5%)에서 정점을 찍고 768에서는 80% 근처까지 **급락**한다. 학습 해상도와 다른 큰 이미지를 넣으면 위치 임베딩 보간(interpolation) 밖으로 벗어나 일반화가 깨지는 것이다. 반면 416과 224→416은 512~640에서 83%대를 유지하며 거의 겹친다.
2. **오른쪽(ADE-20K mIoU, 픽셀 수준 태스크)**: 격차가 더 크다. 224 모델은 336을 넘으면 계속 떨어져 768에서 41%대인데, 224→416은 640에서 약 46%로 416 모델(~46.5%)에 0.5 포인트 이내로 근접한다. 분할처럼 **패치 수준 특징을 쓰는 태스크에서 고해상도 적응의 효과가 가장 뚜렷**하다.
3. 파랑 실선이 빨강 점선의 형태(고해상도에서도 성능이 유지되는 곡선)를 그대로 따라간다는 것이 핵심 메시지다. "짧은 고해상도 마무리만으로 처음부터 고해상도로 학습한 것과 거의 같은 거동을 얻는다."

이 결과를 바탕으로 논문은 "고해상도로 처음부터 학습하는 대신 학습 끝에 이 단계를 넣는다"고 결론짓는다.

## 4. 왜 518인가

- DINOv2는 패치 크기 **14**를 쓴다. $518 = 14\times37$이라 정확히 $37\times37$ 패치 격자가 되어 남는 픽셀이 없다(ablation의 ViT-L/16은 $416=16\times26$).
- 공개 모델은 이 $518$에 맞춘 위치 임베딩을 가지고 있다. 이 저장소의 평가 설정 `dinov2/configs/eval/vit*_pretrain.yaml`에 `global_crops_size: 518  # this is to set up the position embeddings properly`라고 적혀 있는 이유가 바로 이 마무리 단계 때문이다.
- 다른 해상도 입력은 `dinov2/models/vision_transformer.py`의 `interpolate_pos_encoding`으로 위치 임베딩을 보간해 처리한다. 고해상도에서 학습을 마쳤기 때문에, Fig. 6의 224→416 곡선처럼 큰 이미지를 넣어도 성능이 무너지지 않는다.

## 5. 한 줄 요약

> 고해상도는 **밀집(dense) 태스크에 필수**지만 처음부터 쓰기엔 **비싸다** → DINOv2는 $224$로 625k 스텝 학습한 뒤, **마지막 10k 스텝만 $518\times518$**로 올려 같은 레시피(학습률만 낮춤)를 압축 실행한다. Fig. 6이 보여주듯 이 짧은 단계만으로 전 구간 고해상도 학습과 거의 같은 성능을 훨씬 적은 비용으로 얻는다.

## 참고

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 — 4장 "Adapting the resolution", 6.6절 "Impact of Resolution", 부록 B.2 "High-Resolution adaptation", Table 16.
- Touvron et al., *Fixing the train-test resolution discrepancy*, NeurIPS 2019 (FixRes).
- Likhomanenko et al., 2021 (UniViT); Beyer et al., 2023 (FlexiViT).
