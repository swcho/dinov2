# Sandler et al. (2022) learnable memory vs. Vision Transformers Need Registers

## 카드 요약

- **질문**: Sandler et al.(2022)의 learnable memory 연구와 이 논문의 차이는?
- **답**: 그들은 vision 도메인에서 **fine-tuning**에 memory token을 적용했고 **과제 간 전이가 잘 안 됨**을 관찰했다. 이 논문은 fine-tuning 없이 **사전학습 단계**에서 추가 토큰을 써서 **모든 downstream 과제의 feature를 개선**한다.

이 카드는 Darcet et al., *Vision Transformers Need Registers* (arXiv:2309.16588)의 Related Work "Additional tokens in transformers" 문단에서 나온 것이다. 원문은 이렇게 적는다.

> Sandler et al. (2022) extend this line to the vision domain for fine-tuning but observe that such tokens do not transfer well across tasks. In contrast, we do not perform fine-tuning and employ additional tokens during pretraining to improve the features obtained for all tasks downstream.

즉 "토큰을 추가한다"는 형태만 보면 같은 계보(Memory Transformer, Burtsev et al. 2020)지만, **언제·왜 넣는가**가 완전히 다르다.

---

## 1. Sandler et al. (2022)는 정확히 무엇을 했나

논문: Mark Sandler, Andrey Zhmoginov, Max Vladymyrov, Andrew Jackson, *Fine-tuning Image Transformers using Learnable Memory*, CVPR 2022 (arXiv:2203.15243).

### 1.1 메커니즘: layer마다 붙였다 버리는 memory token

사전학습된 ViT를 두고, **각 레이어 입력에** 학습 가능한 임베딩 `E_mem ∈ R^(m×D)`를 concat한다.

```
z_mem^l = [ y^(l-1) ; E_mem^l ]        # 레이어 l의 입력
```

여기서 중요한 세 가지 설계 결정이 있다.

1. **레이어마다 별도의 memory**를 새로 concat한다 (입력층에 한 번만 넣는 Memory Transformer와 다름).
2. memory token 자신은 **다른 토큰에 attend하지 않는다**. 오직 "attend 당하는 쪽(source of attention)"으로만 쓰인다 — 즉 key/value 역할만 한다.
3. self-attention 출력은 앞의 `N+1`개(패치 + [CLS])로 **truncate**된다. 즉 memory는 다음 레이어로 **전파되지 않고 그 레이어에서 버려진다**.

> 저자들은 memory가 실제로 attend하고 다음 레이어로 전파되는 변형도 실험했지만 "일반적으로 성능이 나빠졌다(hurt performance)"고 보고한다.

토큰 수는 layer당 한 줌 수준이며, 논문은 **5개 정도가 데이터셋 전반에 잘 통하는 범용 크기**라고 결론짓는다(ablation은 1/5/10/20 등을 비교).

### 1.2 목적: parameter-efficient fine-tuning

backbone 가중치는 그대로 두고 **memory token + classifier head + [CLS] token만** gradient descent로 학습한다. 결과는

- head-only fine-tuning보다 **확실히 좋고**,
- 훨씬 비싼 full fine-tuning보다 **살짝 낮은** 수준.

덤으로 full fine-tuning은 learning rate에 매우 민감한 반면 memory 방식은 큰 lr에서도 안정적이다.

### 1.3 attention masking: 여러 task를 한 모델에 얹기 위한 장치

fine-tuning으로 [CLS]나 memory를 건드리면 hidden activation이 바뀌어 **원래 task 성능이 망가진다**. 이를 막기 위해 저자들은 attention mask를 도입한다.

- 새 task를 위해 `x_cls^new`와 새 memory를 추가하되,
- **기존 `x_cls`와 패치 토큰이 새로 추가된 토큰들에 attend하지 못하게 막는다.**

그러면 원래 task의 출력은 **비트 단위로 보존**되고, 하나의 forward pass에서 old task와 new task를 동시에 풀 수 있다(computation reuse). task마다 head를 하나씩 붙인다. 확장 방식은 두 가지 — 순차적으로 붙이는 *model extension*과 독립 학습된 모델들을 합치는 *model concatenation*.

### 1.4 "전이가 안 된다"의 실제 근거

Sec. 4.4가 이 카드의 핵심 근거다. 같은 사전학습 ViT-B/32에서 SUN-397, Places-365, i-Naturalist, CIFAR-100용 모델 4개를 **각각 독립적으로** fine-tuning한 뒤, **masking 없이** 그 class token과 memory를 순진하게 concat해 나가면:

| 누적 상태 | i-Naturalist 정확도 |
|---|---|
| i-Naturalist memory만 | **50.0** |
| + Places-365 memory | 28.1 |
| + CIFAR-100 memory | 27.8 |
| + SUN-397 memory | **17.3** |

한 task용으로 학습된 memory는 다른 task의 계산에 섞이는 순간 **간섭(interference)만 일으킨다**. 즉 memory는 철저히 **dataset-specific**이고, 그래서 attention masking으로 서로를 격리해 줘야만 공존이 가능하다. 이것이 Darcet et al.이 인용한 "do not transfer well across tasks"의 실체다.

---

## 2. 이 논문(Registers)은 무엇을 하나

![Figure 6: register token을 붙인 ViT 구조](fig-6-registers.jpeg)

- `N`개의 학습 가능한 토큰을 **patch embedding 직후 입력 시퀀스에** 붙인다([CLS]와 같은 방식).
- 이 토큰들은 **일반 토큰처럼 전 레이어를 통과하며 attend도 하고 attend 당하기도 한다**.
- **출력에서는 버린다.** 학습·추론 모두 [CLS]와 patch token만 표현으로 쓴다.
- 결정적으로, 이건 fine-tuning 트릭이 아니라 **DINOv2 / OpenCLIP / DeiT-III의 사전학습 자체를 register와 함께 다시 도는 것**이다.

동기는 task 적응이 아니라 **병리 제거**다. 충분히 크고 오래 학습한 ViT는 정보량이 적은(이웃과 비슷한) 배경 패치를 골라 거기에 global 정보를 저장·처리·회수하는 법을 스스로 배운다. 그 결과 그 패치들은 norm이 ~10배 튀고(전체의 약 2%), 원래의 위치·픽셀 정보를 잃는다. register는 이 "내부 스크래치패드" 역할을 **떠맡을 전용 자리**를 만들어 주는 것이다.

효과는 특정 task에 국한되지 않는다 — norm outlier가 완전히 사라지고, attention map과 feature map이 매끄러워지며, dense prediction(ADE-20k, NYUd)에서 SOTA를 갱신하고, LOST 같은 unsupervised object discovery가 큰 모델에서도 동작하게 된다(DINOv2 VOC2007 35.3 → 55.4 corloc). register 4개 기준 FLOP 증가는 2% 미만, 파라미터 증가는 무시할 수준이다.

![Figure 9: [CLS]와 register token의 attention map 비교](fig-9-register-attention.jpeg)

흥미롭게도 register들은 요구한 적 없는데도 서로 다른 영역에 attend하는 slot-attention 비슷한 행동을 자연스럽게 보인다. **task별로 나눠 놓은 게 아니라 학습 중 저절로 분화된 것**이라는 점이 Sandler의 task-specific memory와 대비된다.

---

## 3. 축별 비교표

| 축 | Sandler et al. 2022 (learnable memory) | Darcet et al. 2023 (registers) |
|---|---|---|
| **언제 토큰을 넣는가** | **fine-tuning 단계**. 사전학습된 ViT는 얼려 두고 memory + head + [CLS]만 학습 | **사전학습 단계**. 처음부터 register와 함께 backbone을 통째로 학습 |
| **어디에 넣는가** | **레이어마다** 새로 concat, 그 레이어 끝에서 truncate. 다음 레이어로 전파 안 됨 | patch embedding 직후 **입력 시퀀스에 한 번**. 전 레이어를 통과 |
| **attention 방식** | memory는 attend하지 **않음** (key/value로만 소비됨) | 일반 토큰과 동일하게 **양방향으로 attend** |
| **목적** | **과제별 적응** — parameter-efficient transfer, 한 backbone에 여러 task 얹기 | **표현 자체 개선** — high-norm artifact 제거, feature/attention map 정화 |
| **출력 사용 여부** | 출력은 안 쓰지만 **task별 [CLS] + head**가 판독기로 붙음. 즉 task 출력 경로가 토큰에 매여 있음 | 출력값을 **전혀 쓰지 않음**. 순수한 내부 스크래치패드. 표현은 여전히 [CLS] + patch |
| **전이성** | **나쁨.** task별로 특화되어 다른 task memory와 섞으면 간섭 (i-Nat 50.0 → 17.3). attention masking으로 **격리**해야만 공존 | **범용.** 하나의 사전학습 모델이 classification·segmentation·depth·object discovery 전부에서 개선 |
| **얻는 것** | head-only보다 높고 full fine-tuning에 근접한 정확도 + 멀티태스크 computation reuse | 모든 downstream task의 feature 품질, 해석 가능한 attention map, dense task SOTA, LOST 동작 |
| **비용** | task당 소수 파라미터 (backbone 재학습 없음) | 사전학습 재실행 필요. 대신 추론 비용은 +2% FLOPs 미만 |

한 줄 대비: **Sandler는 "고정된 모델을 새 task에 맞추는 손잡이"로 토큰을 썼고, Darcet은 "모델이 이미 몰래 하고 있던 짓을 위한 전용 공간"으로 토큰을 썼다.**

---

## 4. 마무리 — 이 논문의 진짜 논점

Related Work 문단에서 가장 중요한 문장은 차이 나열이 아니라 그 뒤에 붙은 insight다.

> our study contributes the following new insight: the mechanism implemented through memory tokens **already appears naturally in Vision Transformers**; our study shows that *such tokens allow us not to create but to **isolate** this existing behavior*, and thus avoid collateral side-effects.

Memory Transformer 계열(Burtsev, Bulatov, Sandler)은 모두 **"모델에 없던 memory 능력을 새로 부여한다"**는 프레임으로 토큰을 추가했다. Darcet et al.의 발견은 그 전제를 뒤집는다 — 충분히 크고 오래 학습한 ViT는 **아무도 시키지 않았는데 이미 memory 메커니즘을 자발적으로 구현하고 있었다.** 다만 그 저장소를 배경 patch token에서 훔쳐 쓰고 있었고, 그 대가로 해당 패치의 local 정보가 파괴되어 dense prediction 성능과 attention map 해석 가능성이 함께 무너진 것이다.

따라서 register가 하는 일은 **create가 아니라 isolate**다. 새 능력을 주는 게 아니라, 이미 벌어지고 있던 계산을 patch token에서 **분리해 전용 자리로 옮기는 것**이고, 그 결과 patch token은 본업(local 정보 보존)만 하면 되니 부수 피해(collateral side-effects)가 사라진다. 그래서 이 개선은 특정 task에 대한 적응이 아니라 표현 전반에 걸친 정화이며, 전이가 안 되는 Sandler의 task-specific memory와 달리 **모든 downstream task가 함께 이득을 본다**.

---

## 참고

- Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers*. ICLR 2024. arXiv:2309.16588
- Sandler, Zhmoginov, Vladymyrov, Jackson. *Fine-tuning Image Transformers using Learnable Memory*. CVPR 2022. arXiv:2203.15243
- Burtsev, Kuratov, Peganov, Sapunov. *Memory Transformer*. arXiv:2006.11527 (register 메커니즘의 원류)
- Bulatov, Kuratov, Burtsev. *Recurrent Memory Transformer*. NeurIPS 2022
