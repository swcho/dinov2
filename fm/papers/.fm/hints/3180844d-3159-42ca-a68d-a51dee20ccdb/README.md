# Knowledge distillation 효과 검증 (DINOv2, Fig. 5) — 실험 설계 해설

> **Q.** knowledge distillation 효과 검증(Fig. 5) 실험 설계는?
> **A.** scratch로 학습한 ViT-L/14와 ViT-g/14에서 증류한 ViT-L/14를 12개 벤치마크에서 비교하고, teacher인 ViT-g/14를 topline으로 함께 보고한다. 증류 모델이 12개 전부에서 우수했고 때로는 teacher까지 능가했다.

출처: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv 2304.07193v2), Sec. 5 "Model distillation", Sec. 6.5 "Impact of Knowledge Distillation", Fig. 5, Appendix Table 16–17.

---

## 1. 배경: DINOv2에서 distillation이 필요한 이유

DINOv2의 기술적 개선(KoLeo, 시퀀스 패킹, FSDP, 고해상도 마무리 학습 등)은 대부분 **큰 모델(ViT-g, 1B 파라미터)을 대량 데이터로 안정적으로 학습**시키는 데 초점을 둔다. 그런데 배포용으로 필요한 것은 ViT-S/B/L 같은 작은 모델이다. 논문은 작은 모델을 처음부터(scratch) 학습하는 대신 **가장 큰 ViT-g/14를 frozen teacher로 두고 knowledge distillation**(Hinton et al., 2014)으로 얻는다.

distillation 절차(Sec. 5)는 DINOv2 학습 루프를 거의 그대로 재사용하되 다음만 바꾼다.

| 항목 | scratch 학습 | distillation |
|---|---|---|
| teacher | student의 EMA(self-distillation) | 미리 학습된 **더 큰 모델, frozen** |
| 최종 모델 | student의 EMA | student의 별도 EMA 사본 |
| masking(iBOT MIM) | 사용 | **제거**, iBOT loss는 두 global crop에 적용 |
| stochastic depth | 0.4 | **0** |
| LR / batch (Table 16) | 3.5e-4 / 3072 | 1e-3 / 2048 |
| FFN (Table 17) | SwiGLU | MLP |

즉 DINOv2의 목적함수 자체가 teacher→student 증류 형태이므로, "teacher를 자기 EMA에서 큰 frozen 모델로 갈아끼우는 것"이 곧 distillation이 된다.

---

## 2. 실험 설계 (Sec. 6.5, Fig. 5)

세 모델을 **동일한 12개 벤치마크**에서 비교한다.

1. **ViT-L/14 Scratch** — DINOv2 레시피로 LVD-142M에서 처음부터 학습한 baseline.
2. **ViT-L/14 Distill** — 같은 ViT-L/14 아키텍처(1024 dim, 16 heads, 24 blocks)를 ViT-g/14에서 증류.
3. **ViT-g/14 Scratch** — 증류에 쓰인 **teacher**. 성능의 상한선(topline)을 보여주는 참조용으로만 표시.

![Fig. 5 — (a) 12개 벤치마크 radar chart, (b) 8개 task 그룹 평균](fig-1.jpeg)

### 그림에서 실제로 관찰되는 것

**(a) 개별 벤치마크 radar chart** — 12개 축은 다음과 같다(시계 방향, 위에서부터). 파란 실선 = Scratch, 주황 실선(채움) = Distill, 빨간 점선 = ViT-g teacher.

| 축 | 과제 유형 | Scratch | Distill | Teacher(g) |
|---|---|---|---|---|
| iNat18 | fine-grained 분류 | 77.8 | 80.4 | 81.6 |
| Cars | fine-grained 분류 | 81.8 | **90.1** | 91.4 |
| Food | fine-grained 분류 | 92.8 | 94.3 | 94.7 |
| INet-1k | 분류 (linear probe) | 84.5 | 86.3 | 86.5 |
| NYUd ↓ | 단안 depth (RMSE, 낮을수록 좋음) | 0.345 | 0.333 | 0.298 |
| Kitti ↓ | 단안 depth (낮을수록 좋음) | 2.57 | 2.50 | 2.35 |
| INet-R | 도메인 robustness | 68.1 | 74.1 | 78.8 |
| INet-A | 자연 적대적 예제 robustness | 61.7 | **71.3** | 75.9 |
| Paris-H | instance retrieval (mAP, Hard) | 77.6 | **84.4** | 82.7 |
| Oxford-H | instance retrieval (mAP, Hard) | 47.7 | 52.1 | 52.6 |
| Places205 | 장면 분류 | 66.0 | 67.3 | 67.5 |
| iNat21 | fine-grained 분류 | 83.1 | 85.1 | 85.7 |

- 주황 영역이 파란 다각형을 **12개 축 모두에서 바깥으로 감싼다** → "distilled가 scratch보다 전부 우수".
- Cars(+8.3), INet-A(+9.6), Paris-H(+6.8), INet-R(+6.0)처럼 격차가 큰 축이 있고, INet-1k(+1.8), Places205(+1.3)처럼 작은 축도 있다. 즉 이득은 **robustness·fine-grained·retrieval**에서 특히 크다.
- **Paris-H에서는 주황선이 빨간 점선 밖으로 나간다**(84.4 vs 82.7) — 학생이 teacher를 능가한 사례.
- depth(NYUd, Kitti)와 INet-A/R에서는 teacher와의 격차가 여전히 눈에 띈다 — 증류가 만능은 아니다.

**(b) 8개 vision task 그룹 평균** — 개별 벤치마크를 8개 카테고리로 묶은 표.

| | INet-1k | Segm. | Depth↓ | Classif. | Finegr. | Retriev. | ARSketch | Video |
|---|---|---|---|---|---|---|---|---|
| ViT-g/14 Scratch (teacher) | 86.5 | 73.4 | 1.00 | 92.1 | 78.3 | 75.2 | 77.0 | 69.3 |
| ViT-L/14 Scratch | 84.5 | 72.2 | 1.10 | 90.2 | 75.8 | 71.3 | 69.5 | 67.3 |
| ViT-L/14 Distill | **86.3** | **73.3** | **1.08** | **91.2** | **77.6** | **76.3** | **74.5** | **67.5** |

- 8개 그룹 모두 Distill > Scratch(굵게 표시).
- **Retrieval 평균 76.3은 teacher(75.2)보다 높다** — 그룹 평균 수준에서도 teacher 초과가 확인된다.
- INet-1k(86.3 vs 86.5), Segm.(73.3 vs 73.4)은 teacher에 거의 붙어 있다. 파라미터 수가 약 1/3(300M vs 1.1B)인 모델이 이 정도면 비용 대비 매우 효율적이다.

---

## 3. 왜 이 설계가 distillation 효과를 "공정하게 분리"하는가

핵심은 **비교 대상 두 개가 같은 ViT-L/14 아키텍처**라는 점이다.

- **아키텍처·용량 고정**: Scratch와 Distill은 embed dim, head 수, block 수가 완전히 같다(Table 17). 따라서 성능 차이가 "모델이 더 커서"가 아니라 **학습 신호(teacher)의 차이**에서 나온 것임이 보장된다. 만약 ViT-B scratch vs ViT-L distilled처럼 비교했다면 크기 효과와 증류 효과가 섞여 해석이 불가능하다.
- **데이터·학습 길이 고정**: 두 모델 모두 LVD-142M, 625k iteration, AdamW, 같은 스케줄로 학습(Table 16). 데이터 이점으로 설명될 여지를 제거한다.
- **평가 프로토콜 고정**: 12개 벤치마크 모두 **frozen backbone + linear probe / kNN / 비파라메트릭 retrieval**로 평가한다. fine-tuning이 없으므로 downstream 학습이 격차를 가리거나 만들지 못한다.
- **topline의 역할**: ViT-g/14 teacher는 비교 대상이 아니라 **"이 증류로 얻을 수 있는 최대치가 어디인가"**를 보여주는 기준선이다. Distill이 Scratch보다 낫다는 것만으로는 "얼마나 나은지"를 해석하기 어렵지만, teacher 선이 함께 있으면 (i) 학생이 teacher에 얼마나 근접했는지, (ii) 어느 벤치마크에서 아직 격차가 남는지(depth, INet-A/R), (iii) 어디에서 teacher를 넘어섰는지(Paris-H, Retrieval 평균)를 한눈에 읽을 수 있다.
- **폭넓은 벤치마크**: 분류·fine-grained·robustness·retrieval·depth를 모두 포함해, 증류가 특정 과제(예: ImageNet)에만 유리한 것이 아니라 **범용 feature 전반**에 도움이 됨을 보인다. 12/12 전승이라는 결과가 그래서 설득력이 있다.

이 설계의 한계도 알아두자. 증류 모델은 stochastic depth 0, LR 1e-3, MLP FFN 등 **하이퍼파라미터가 scratch와 다르다**. 저자들은 각 설정에 맞는 최적 레시피를 썼다고 보는 것이 합리적이지만, 엄밀히 말하면 "teacher의 유무" 하나만 바뀐 통제실험은 아니다.

---

## 4. 12개 벤치마크가 무엇인가

Fig. 5(a)의 12개 축을 과제 유형별로 정리하면:

- **이미지 분류(linear probe)**: ImageNet-1k, Places205(장면), iNaturalist 2018, iNaturalist 2021, Stanford Cars, Food-101 (뒤의 4개는 fine-grained).
- **도메인 일반화 / robustness**: ImageNet-A(자연 적대적 예제), ImageNet-R(rendition — 그림·조각·만화 등 스타일 변형). ImageNet-1k에서 학습한 linear head를 그대로 적용해 평가한다.
- **Instance-level retrieval(비파라메트릭, cosine 유사도 랭킹)**: Revisited Oxford-Hard, Revisited Paris-Hard (mAP).
- **단안 depth estimation(patch-level feature, linear head)**: NYUd(NYU Depth V2), KITTI. RMSE라서 화살표 ↓, 값이 작을수록 바깥쪽에 그려져 있다.

Fig. 5(b)는 여기에 Segmentation(ADE20k 등), ARSketch(ImageNet-A/R/Sketch 평균), Video(K400/UCF-101/SSv2) 등을 더한 8개 그룹 평균이다.

---

## 5. 학생이 teacher를 "능가"하는 것이 어떻게 가능한가

직관적으로는 teacher 출력을 모사하는 학생이 teacher보다 나을 수 없어 보이지만, 실제로 여러 이유로 가능하다.

1. **증류 목표는 teacher와 같아지는 것이 아니라 teacher의 확률 분포(soft target)에서 학습하는 것**이다. soft target은 클래스(프로토타입) 간 유사도 구조라는 "dark knowledge"를 담고 있어, one-hot이나 self-EMA 신호보다 정보량이 많고 노이즈가 적다. 학생은 teacher의 개별 실수까지 따라가지 않고 그 **평균적·부드러운 구조**를 배우므로 일종의 정규화 효과를 얻는다.
2. **teacher는 frozen이고, 학생은 다른 아키텍처·정규화로 학습**된다. 학생은 masking과 stochastic depth를 빼고 MLP FFN을 써서 최적화 조건이 다르다. 특정 과제(예: retrieval처럼 feature 공간의 균일한 분포가 중요한 과제)에서는 이 조건이 teacher보다 유리하게 작동할 수 있다.
3. **최종 모델이 학생의 EMA**다. EMA 가중치 평균은 flat minimum 쪽으로 이동해 일반화가 좋아지는 경향이 있고, 이것이 teacher 자체가 갖지 못한 추가 이득이 된다.
4. **평가 과제가 학습 목표와 다르다.** teacher는 ImageNet 같은 분류에 최적인 feature를 만들지만, retrieval(Paris-H)은 인스턴스 단위의 지역 구조를 본다. 학생이 teacher의 표현을 "압축·재배열"하는 과정에서 어떤 과제에는 더 적합한 geometry가 생길 수 있다. 실제로 Fig. 5에서 teacher를 넘어선 곳은 **retrieval**뿐이고, 분류·depth·robustness에서는 teacher가 여전히 위다 — 이 패턴이 위 설명과 일치한다.
5. 일반적으로 알려진 "Born-Again Networks"(Furlanello et al., 2018) 현상 — 같은 크기의 teacher로 증류해도 학생이 teacher를 넘는다 — 도 같은 맥락이다. 증류가 단순 복제가 아니라 **더 좋은 학습 신호**임을 시사한다.

---

## 6. 요약 카드

- **무엇을 비교?** ViT-L/14 scratch vs ViT-L/14 distilled(from ViT-g/14), 여기에 ViT-g/14 teacher를 topline으로.
- **어디서?** 12개 벤치마크: INet-1k, Places205, iNat18/21, Cars, Food, INet-A/R, Oxford-H, Paris-H, NYUd, KITTI (+ 8개 그룹 평균).
- **결과?** 12/12 distilled 승, 8/8 그룹 평균 승. Paris-H(84.4>82.7)와 Retrieval 평균(76.3>75.2)에서는 teacher까지 능가.
- **왜 공정?** 아키텍처·데이터·iteration·평가 프로토콜을 고정해 학습 신호(teacher) 차이만 남김; topline이 도달 가능한 상한을 보여줌.
- **함의?** DINOv2 배포용 ViT-S/B/L은 모두 ViT-g에서 증류한 모델이며, 이 실험이 그 선택을 정당화한다. 결론(Sec. 8)에서도 성능 원인 4가지 중 하나로 distillation을 명시한다.
