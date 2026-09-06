# %% [markdown]
# # DINO/iBOT의 teacher는 어떻게 만들어지나 — EMA(지수이동평균) teacher 실험
#
# DINO/iBOT/DINOv2에서 **학습(경사하강)되는 것은 student**뿐이고, teacher의 파라미터
# $\tau$는 student 파라미터 $s$의 **exponential moving average**다.
#
# $$\tau_t = m\,\tau_{t-1} + (1-m)\,s_t,\qquad \tau_0 = s_0$$
#
# - teacher는 student와 같은 상태로 초기화된다.
# - 매 학습 스텝이 끝난 직후 위 식으로 갱신된다.
# - $m$(momentum)은 DINOv2에서 0.994 → 1.0 코사인 스케줄.
#
# 이 스크립트는 numpy만으로 (1) 점화식 = 등비 가중합 확인, (2) 기억 길이 $1/(1-m)$,
# (3) 잡음 섞인 student 궤적을 EMA가 어떻게 매끈하게 만드는지, (4) 코사인 momentum 스케줄,
# (5) DINOv2 `update_teacher` 코드와 같은 파라미터-딕셔너리 EMA를 순서대로 보여준다.

# %%
# 필요 패키지: numpy, plotly, kaleido (expy.png 저장용)
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. 점화식으로 계산한 EMA == 등비 가중합으로 계산한 EMA
#
# 점화식을 풀면
# $$\tau_t = (1-m)\sum_{k=0}^{t-1} m^k s_{t-k} + m^t \tau_0$$
# 이므로 두 방법의 결과가 일치해야 한다.


# %%
def ema_recursive(s, m):
    """teacher 값 tau_t를 점화식으로 계산. tau_0 = s_0 (student와 같은 초기화)."""
    tau = np.empty_like(s, dtype=float)
    tau[0] = s[0]
    for t in range(1, len(s)):
        tau[t] = m * tau[t - 1] + (1 - m) * s[t]
    return tau


def ema_closed_form(s, m):
    """등비 가중합으로 직접 계산 (검산용)."""
    tau = np.empty_like(s, dtype=float)
    for t in range(len(s)):
        k = np.arange(t)  # k = 0..t-1 → s[t-k]
        tau[t] = (1 - m) * np.sum(m**k * s[t - k]) + m**t * s[0]
    return tau


s_demo = rng.normal(size=50).cumsum()  # 임의의 student iterate 수열
m_demo = 0.9
a, b = ema_recursive(s_demo, m_demo), ema_closed_form(s_demo, m_demo)
print("점화식 vs 등비가중합 최대 오차:", np.abs(a - b).max())
print("가중치 합 (1-m)Σm^k + m^t, t=49:", (1 - m_demo) * np.sum(m_demo ** np.arange(49)) + m_demo**49)
# 출력: 점화식 vs 등비가중합 최대 오차: 8.881784197001252e-16
# 출력: 가중치 합 (1-m)Σm^k + m^t, t=49: 1.0

# %% [markdown]
# ## 2. teacher는 얼마나 먼 과거까지 기억하나?
#
# 가중치 $(1-m)m^k$를 지연 $k$에 대한 확률분포(기하분포)로 보면 평균 지연은
# $$\mathbb{E}[k] = \frac{m}{1-m} \approx \frac{1}{1-m}$$
# 가중치가 절반이 되는 지연은 $k_{1/2} = \ln 2 / (-\ln m)$.

# %%
print(f"{'m':>6} | {'평균 지연 m/(1-m)':>18} | {'1/(1-m)':>8} | {'절반 지점':>9}")
for m in [0.9, 0.99, 0.994, 0.999]:
    k = np.arange(20000)
    w = (1 - m) * m**k
    mean_lag = np.sum(k * w)
    half = np.log(2) / (-np.log(m))
    print(f"{m:>6} | {mean_lag:>18.1f} | {1/(1-m):>8.1f} | {half:>9.1f}")
# 출력:      m |  평균 지연 m/(1-m) |  1/(1-m) |      절반 지점
# 출력:    0.9 |                9.0 |     10.0 |       6.6
# 출력:   0.99 |               99.0 |    100.0 |      69.0
# 출력:  0.994 |              165.7 |    166.7 |     115.2
# 출력:  0.999 |              999.0 |   1000.0 |     692.8

k_plot = np.arange(0, 600)
fig_w = go.Figure()
for m in [0.9, 0.99, 0.994]:
    fig_w.add_trace(go.Scatter(x=k_plot, y=(1 - m) * m**k_plot, name=f"m={m}"))
fig_w.update_layout(title="k 스텝 전 student iterate에 붙는 가중치 (1-m)m^k",
                    xaxis_title="지연 k (스텝)", yaxis_title="가중치", yaxis_type="log")
_show(fig_w)

# %% [markdown]
# ## 3. 잡음 섞인 student 궤적 vs 매끈한 teacher
#
# student는 미니배치 gradient 잡음 때문에 요동친다. 1차원 손실 $L(\theta)=\tfrac12(\theta-\theta^*)^2$를
# 잡음 섞인 SGD로 최소화하는 student를 흉내내고, 그 iterate에 EMA를 씌운 teacher를 비교한다.
# 초기값은 둘 다 같고(`tau[0]=s[0]`), 매 스텝 student 갱신 **직후** teacher를 갱신한다.

# %%
T = 2000
theta_star = 3.0
lr, noise = 0.02, 1.0
s = np.empty(T)
s[0] = -5.0  # student 초기값
for t in range(1, T):
    grad = (s[t - 1] - theta_star) + noise * rng.normal()  # 잡음 섞인 gradient
    s[t] = s[t - 1] - lr * grad

tau_994 = ema_recursive(s, 0.994)
tau_99 = ema_recursive(s, 0.99)

def rms_err(x):  # 후반 절반 구간에서 최적값과의 RMS 거리
    return np.sqrt(np.mean((x[T // 2:] - theta_star) ** 2))

print("초기값 일치 여부 s[0]==tau[0]:", s[0] == tau_994[0])
print(f"후반 RMS 오차  student: {rms_err(s):.4f}")
print(f"후반 RMS 오차  teacher(m=0.99):  {rms_err(tau_99):.4f}")
print(f"후반 RMS 오차  teacher(m=0.994): {rms_err(tau_994):.4f}")
# 출력: 초기값 일치 여부 s[0]==tau[0]: True
# 출력: 후반 RMS 오차  student: 0.1168
# 출력: 후반 RMS 오차  teacher(m=0.99):  0.0727
# 출력: 후반 RMS 오차  teacher(m=0.994): 0.0545

# %% [markdown]
# teacher(EMA)가 student보다 최적값에 더 가깝고 흔들림이 훨씬 작다 — 이것이 DINO 계열에서
# teacher가 student보다 성능이 좋고, 최종 배포 가중치로 teacher를 쓰는 이유(Polyak averaging 효과)다.
# 대신 초반에는 teacher가 student보다 **뒤처져서** 따라온다(지연 ≈ 1/(1-m) 스텝).

# %% [markdown]
# ## 4. DINOv2의 momentum 코사인 스케줄 0.994 → 1.0
#
# $$m(t) = m_{\text{final}} + \tfrac12\,(m_{\text{base}} - m_{\text{final}})\Big(1+\cos\frac{\pi t}{T}\Big)$$
# (`dinov2/train/train.py`의 `CosineScheduler(base_value=0.994, final_value=1.0)`와 동일)


# %%
def cosine_momentum(t, T, base=0.994, final=1.0):
    return final + 0.5 * (base - final) * (1 + np.cos(np.pi * t / T))


T_total = 625_000  # DINOv2 총 iteration 수
its = np.linspace(0, T_total, 7)
for t in its:
    m = cosine_momentum(t, T_total)
    print(f"iter {int(t):>7}: m = {m:.5f}, 기억 길이 1/(1-m) ≈ {1/(1-m) if m < 1 else float('inf'):>10.0f} 스텝")
# 출력: iter       0: m = 0.99400, 기억 길이 1/(1-m) ≈        167 스텝
# 출력: iter  104166: m = 0.99440, 기억 길이 1/(1-m) ≈        179 스텝
# 출력: iter  208333: m = 0.99550, 기억 길이 1/(1-m) ≈        222 스텝
# 출력: iter  312500: m = 0.99700, 기억 길이 1/(1-m) ≈        333 스텝
# 출력: iter  416666: m = 0.99850, 기억 길이 1/(1-m) ≈        667 스텝
# 출력: iter  520833: m = 0.99960, 기억 길이 1/(1-m) ≈       2488 스텝
# 출력: iter  625000: m = 1.00000, 기억 길이 1/(1-m) ≈        inf 스텝

# %% [markdown]
# ## 5. 실제 코드 형태: 파라미터 딕셔너리 전체에 같은 공식 적용
#
# DINOv2 `SSLMetaArch.update_teacher(m)`은 모든 파라미터 텐서에 대해
# `teacher *= m; teacher += (1-m) * student` 를 수행한다(`torch._foreach_mul_/_foreach_add_`).
# 초기화는 `teacher.load_state_dict(student.state_dict())`. numpy로 그대로 재현한다.


# %%
def init_teacher(student):
    """teacher.load_state_dict(student.state_dict()) 에 해당"""
    return {k: v.copy() for k, v in student.items()}


def update_teacher(teacher, student, m):
    """torch._foreach_mul_(teacher, m); torch._foreach_add_(teacher, student, alpha=1-m)"""
    for k in teacher:
        teacher[k] *= m
        teacher[k] += (1 - m) * student[k]


student_params = {"backbone.w": rng.normal(size=(4, 4)), "dino_head.w": rng.normal(size=(3,))}
teacher_params = init_teacher(student_params)
print("초기화 직후 teacher == student:",
      all(np.array_equal(student_params[k], teacher_params[k]) for k in student_params))

m = 0.994
for step in range(1, 501):
    for k in student_params:  # student만 gradient 스텝 (여기선 잡음 섞인 랜덤워크로 대체)
        student_params[k] += 0.01 * rng.normal(size=student_params[k].shape)
    update_teacher(teacher_params, student_params, m)  # 스텝 끝에 teacher EMA 갱신
    if step in (1, 100, 500):
        dist = np.sqrt(sum(np.sum((student_params[k] - teacher_params[k]) ** 2) for k in student_params))
        print(f"step {step:>3}: ||student - teacher||_2 = {dist:.4f}")
# 출력: 초기화 직후 teacher == student: True
# 출력: step   1: ||student - teacher||_2 = 0.0441
# 출력: step 100: ||student - teacher||_2 = 0.2707
# 출력: step 500: ||student - teacher||_2 = 0.2872

# %% [markdown]
# ## 6. 시각화 저장 (expy.png)

# %%
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "(a) 가중치 (1-m)m^k — 과거 iterate에 붙는 비중 (log y)",
        "(b) 잡음 섞인 student vs EMA teacher (m=0.994)",
        "(c) DINOv2 momentum 코사인 스케줄 0.994→1",
        "(d) 기억 길이 1/(1-m) (log y)",
    ),
)
for mm in [0.9, 0.99, 0.994]:
    fig.add_trace(go.Scatter(x=k_plot, y=(1 - mm) * mm**k_plot, name=f"가중치 m={mm}"), row=1, col=1)
fig.update_yaxes(type="log", range=[-6, 0], title_text="가중치", row=1, col=1)  # 1e-6 ~ 1
fig.update_xaxes(title_text="지연 k (스텝)", row=1, col=1)

fig.add_trace(go.Scatter(y=s, name="student s_t", line=dict(color="gray", width=1)), row=1, col=2)
fig.add_trace(go.Scatter(y=tau_994, name="teacher τ_t (m=0.994)", line=dict(color="crimson", width=2)), row=1, col=2)
fig.add_hline(y=theta_star, line_dash="dash", line_color="black", row=1, col=2)
fig.update_xaxes(title_text="학습 스텝 t", row=1, col=2)
fig.update_yaxes(title_text="파라미터 값", row=1, col=2)

t_grid = np.linspace(0, T_total, 400)
m_grid = cosine_momentum(t_grid, T_total)
fig.add_trace(go.Scatter(x=t_grid, y=m_grid, name="m(t)", line=dict(color="royalblue")), row=2, col=1)
fig.update_xaxes(title_text="iteration", row=2, col=1)
fig.update_yaxes(title_text="momentum m", row=2, col=1)

mem = 1 / (1 - m_grid[:-1])
fig.add_trace(go.Scatter(x=t_grid[:-1], y=mem, name="1/(1-m)", line=dict(color="seagreen")), row=2, col=2)
fig.update_yaxes(type="log", title_text="스텝 수", row=2, col=2)
fig.update_xaxes(title_text="iteration", row=2, col=2)

fig.update_layout(height=800, width=1200, title_text="EMA teacher: τ_t = m τ_{t-1} + (1-m) s_t",
                  legend=dict(orientation="h", y=-0.08))
_show(fig)

out_png = os.path.join(HERE, "expy.png")
fig.write_image(out_png, scale=1.5)
print("저장:", out_png)
# 출력: 저장: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/daf002b3-d58a-4603-ba0a-15f3771fe8c4/expy.png
