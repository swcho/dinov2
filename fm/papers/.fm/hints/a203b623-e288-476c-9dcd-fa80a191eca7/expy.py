# %% [markdown]
# # DINOv2-g 한 번의 사전학습: GPU-hours → 전력(MWh) → 탄소(tCO₂eq)
#
# 출처: DINOv2 논문(arXiv 2304.07193v2) §9 *Estimating the Environmental Impact of Training our Models*, Table 14.
#
# 논문은 Patterson et al. (2021) 공식을 따르되, 외생 변수(PUE, 탄소 집약도)를
# Touvron et al. (2023, LLaMA)과 같은 값으로 고정해 "미국 평균 데이터센터에서 다시 학습한다면"의
# 잠재 배출량을 보고한다.
#
# $$E_{\text{MWh}} = \frac{\text{GPU-hours} \times P_{GPU}[\text{W}] \times \text{PUE}}{10^6}$$
#
# $$\text{tCO}_2\text{eq} = \frac{E_{\text{MWh}} \times 1000\,[\text{kWh/MWh}] \times 0.385\,[\text{kgCO}_2\text{eq/kWh}]}{1000\,[\text{kg/t}]} = E_{\text{MWh}} \times 0.385$$
#
# 고정값: $P_{GPU} = 400\,\text{W}$ (A100 NVLink 시스템의 TDP), $\text{PUE} = 1.1$, 탄소 집약도 $= 0.385\,\text{kg CO}_2\text{eq/kWh}$ (미국 평균).

# %%
# 필요 패키지: numpy, plotly, kaleido
#   기본 python3에는 numpy가 없으므로 아래 인터프리터로 실행해 검증했다:
#   /home/sungwoo/miniforge3/envs/trellis/bin/python expy.py
import os
import numpy as np


def _show(fig):
    try:
        from IPython import get_ipython
        if get_ipython() is not None:  # VSCode 셀/Jupyter에서만 렌더링
            fig.show()
    except ImportError:
        pass


HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# 논문이 고정한 외생 변수
P_GPU_W = 400        # A100 NVLink 시스템 TDP [W]
PUE = 1.1            # Power Usage Effectiveness
CI_US = 0.385        # 탄소 집약도 [kg CO2eq / kWh], 미국 평균


def energy_mwh(gpu_hours, p_gpu_w=P_GPU_W, pue=PUE):
    """GPU-hours × GPU 전력[W] × PUE → MWh"""
    return gpu_hours * p_gpu_w * pue / 1e6


def carbon_t(e_mwh, ci_kg_per_kwh=CI_US):
    """MWh → kWh(×1000) → kg CO2eq(×CI) → t(÷1000)"""
    return e_mwh * 1000 * ci_kg_per_kwh / 1000


# %% [markdown]
# ## 1. DINOv2-g 한 번의 사전학습 — Table 14 값 재현
#
# 22,016 GPU-hours × 400 W × 1.1 을 단계별로 따라가며 9.7 MWh / 3.7 t 이 나오는지 확인한다.

# %%
GPU_HOURS_G = 22_016

gpu_wh = GPU_HOURS_G * P_GPU_W                  # GPU 자체 소비 [Wh]
dc_wh = gpu_wh * PUE                            # 데이터센터 오버헤드(냉각 등) 포함 [Wh]
e_mwh = dc_wh / 1e6
kg = e_mwh * 1000 * CI_US
t = kg / 1000

print(f"GPU-hours                 : {GPU_HOURS_G:,} h")
print(f"× 400 W                   = {gpu_wh:,.0f} Wh = {gpu_wh/1e6:.4f} MWh (GPU만)")
print(f"× PUE 1.1                 = {dc_wh:,.0f} Wh = {e_mwh:.4f} MWh (데이터센터 전체)")
print(f"→ 반올림                  : {e_mwh:.1f} MWh   (논문 Table 14: 9.7 MWh)")
print(f"× 1000 kWh/MWh × 0.385    = {kg:,.1f} kg CO2eq")
print(f"÷ 1000                    = {t:.4f} t → 반올림 {t:.1f} tCO2eq   (논문: 3.7)")
assert round(e_mwh, 1) == 9.7 and round(t, 1) == 3.7
# 출력:
# GPU-hours                 : 22,016 h
# × 400 W                   = 8,806,400 Wh = 8.8064 MWh (GPU만)
# × PUE 1.1                 = 9,687,040 Wh = 9.6870 MWh (데이터센터 전체)
# → 반올림                  : 9.7 MWh   (논문 Table 14: 9.7 MWh)
# × 1000 kWh/MWh × 0.385    = 3,729.5 kg CO2eq
# ÷ 1000                    = 3.7295 t → 반올림 3.7 tCO2eq   (논문: 3.7)

# %% [markdown]
# ## 2. Table 14 + 본문의 OpenCLIP 비교값 재현
#
# 논문 Table 14에는 **DINOv2-g 한 행만** 있다. OpenCLIP ViT-L / ViT-G는 텍스트 인코더까지 함께 학습하므로
# 표에 넣지 않고, 본문에서 "같은 데이터센터라면 각각 **22.4 MWh**, **118.9 MWh**가 필요하며 약 10× 더 많은
# 탄소를 배출한다"고만 언급한다. 여기서는 논문의 고정값(400 W, PUE 1.1)으로 MWh에서 GPU-hours를 역산하고,
# 0.385 kg/kWh로 tCO₂eq를 계산한다(역산값은 논문에 없는 추정치).

# %%
paper = {
    # 이름: (GPU 종류, GPU-hours(논문), 총 전력 MWh(논문), tCO2eq(논문))
    "DINOv2-g":      ("A100-40GB", 22_016, 9.7,   3.7),
    "OpenCLIP ViT-L": ("(본문만)",  None,   22.4,  None),
    "OpenCLIP ViT-G": ("(본문만)",  None,   118.9, None),
}

rows = []
print(f"{'모델':<16}{'GPU':<12}{'GPU-hours':>12}{'W':>6}{'PUE':>5}{'MWh(계산)':>11}{'MWh(논문)':>10}{'tCO2eq':>9}")
for name, (gpu, gh, mwh_paper, t_paper) in paper.items():
    if gh is None:  # MWh → GPU-hours 역산
        gh = mwh_paper * 1e6 / (P_GPU_W * PUE)
        src = "역산"
    else:
        src = "논문"
    e = energy_mwh(gh)
    c = carbon_t(e)
    rows.append((name, gh, e, c))
    print(f"{name:<16}{gpu:<12}{gh:>10,.0f}({src}){P_GPU_W:>4}{PUE:>5}{e:>11.2f}{mwh_paper:>10}{c:>9.1f}")

ratio_L = rows[1][2] / rows[0][2]
ratio_G = rows[2][2] / rows[0][2]
print(f"\nDINOv2-g 대비 에너지 비율: ViT-L {ratio_L:.1f}×, ViT-G {ratio_G:.1f}×  (논문: '10× more carbon emission')")
# 출력:
# 모델            GPU           GPU-hours     W  PUE  MWh(계산)  MWh(논문)   tCO2eq
# DINOv2-g        A100-40GB       22,016(논문) 400  1.1       9.69       9.7      3.7
# OpenCLIP ViT-L  (본문만)        50,909(역산) 400  1.1      22.40      22.4      8.6
# OpenCLIP ViT-G  (본문만)       270,227(역산) 400  1.1     118.90     118.9     45.8
#
# DINOv2-g 대비 에너지 비율: ViT-L 2.3×, ViT-G 12.3×  (논문: '10× more carbon emission')

# %% [markdown]
# ## 3. 22,016 GPU-hours 는 어느 정도 규모인가
#
# - GPU 수로 나누면 벽시계(wall-clock) 시간이 된다.
# - 논문은 프로젝트 전체를 "약 **200k GPU-days**, 0.5k–1k tCO₂eq"로 추정하며,
#   주 배출원은 자기지도 사전학습이라고 밝힌다. ViT-g 사전학습 1회(22k GPU-hours) = 3.7 t,
#   ImageNet-1k 파인튜닝 1회(1k GPU-hours) = 0.2 t.

# %%
for n_gpu in (128, 256, 512):
    hours = GPU_HOURS_G / n_gpu
    print(f"{n_gpu:>4} GPU 사용 시: {hours:>6.1f} 시간 = {hours/24:>4.1f} 일")

gpu_days_g = GPU_HOURS_G / 24
project_gpu_days = 200_000
print(f"\nDINOv2-g 1회 = {gpu_days_g:,.0f} GPU-days → 프로젝트 전체 {project_gpu_days:,} GPU-days 의 {gpu_days_g/project_gpu_days*100:.2f}%")
print(f"프로젝트 전체를 같은 공식으로 환산: {carbon_t(energy_mwh(project_gpu_days*24)):,.0f} tCO2eq  (논문 추정 0.5k–1k t 범위 안)")
print(f"ViT-g 사전학습 1회 = ImageNet-1k 파인튜닝({1_000} GPU-hours, {carbon_t(energy_mwh(1_000)):.2f} t → 논문 0.2 t) 약 {GPU_HOURS_G/1_000:.0f}회 분량")
print(f"참고: 보잉 777 런던-뉴욕 왕복 1편 ≈ 560 tCO2eq → DINOv2-g 1회는 그 {3.7/560*100:.2f}%")
# 출력:
#  128 GPU 사용 시:  172.0 시간 = 7.2 일
#  256 GPU 사용 시:   86.0 시간 = 3.6 일
#  512 GPU 사용 시:   43.0 시간 = 1.8 일
#
# DINOv2-g 1회 = 917 GPU-days → 프로젝트 전체 200,000 GPU-days 의 0.46%
# 프로젝트 전체를 같은 공식으로 환산: 813 tCO2eq  (논문 추정 0.5k–1k t 범위 안)
# ViT-g 사전학습 1회 = ImageNet-1k 파인튜닝(1000 GPU-hours, 0.17 t → 논문 0.2 t) 약 22회 분량
# 참고: 보잉 777 런던-뉴욕 왕복 1편 ≈ 560 tCO2eq → DINOv2-g 1회는 그 0.66%

# %% [markdown]
# ## 4. 민감도: 탄소 집약도와 PUE 가 바뀌면?
#
# GPU-hours 는 그대로여도 **어디서**(전력망) **어떤 데이터센터에서**(PUE) 학습하느냐에 따라 tCO₂eq는 크게 달라진다.
# 논문이 외생 변수를 고정한 이유가 여기 있다 — 사전학습 방식끼리 공정하게 비교하기 위함.
#
# $$\text{tCO}_2\text{eq} \propto \text{PUE} \times \text{CI}$$

# %%
carbon_intensity = {          # kg CO2eq / kWh (대략적 예시값)
    "프랑스(원전) ~0.05": 0.05,
    "미국 평균 0.385 (논문)": 0.385,
    "석탄 위주 ~0.8": 0.8,
}
pues = {"PUE 1.1 (논문)": 1.1, "PUE 1.6 (구형 DC)": 1.6}

sens = {}
print(f"{'':<26}" + "".join(f"{p:>20}" for p in pues))
for ci_name, ci in carbon_intensity.items():
    vals = [carbon_t(energy_mwh(GPU_HOURS_G, pue=pue), ci) for pue in pues.values()]
    sens[ci_name] = vals
    print(f"{ci_name:<26}" + "".join(f"{v:>18.2f} t" for v in vals))
worst, best = max(max(v) for v in sens.values()), min(min(v) for v in sens.values())
print(f"\n최악/최선 비율: {worst/best:.0f}×  (동일한 22,016 GPU-hours)")
# 출력:
#                                  PUE 1.1 (논문)   PUE 1.6 (구형 DC)
# 프랑스(원전) ~0.05                      0.48 t              0.70 t
# 미국 평균 0.385 (논문)                  3.73 t              5.42 t
# 석탄 위주 ~0.8                          7.75 t             11.27 t
#
# 최악/최선 비율: 23×  (동일한 22,016 GPU-hours)

# %% [markdown]
# ## 5. 시각화
#
# 왼쪽·중앙: 모델별 총 전력(MWh)과 배출량(tCO₂eq) — 단위가 다르므로 축을 나눈 별도 패널.
# 오른쪽: DINOv2-g(22,016 GPU-hours)의 배출량이 탄소 집약도 × PUE 에 따라 어떻게 변하는지.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
names = [r[0] for r in rows]
mwhs = [r[2] for r in rows]
tons = [r[3] for r in rows]

fig = make_subplots(
    rows=1, cols=3, horizontal_spacing=0.08,
    subplot_titles=("총 전력 소비 (MWh)", "탄소 배출 (tCO₂eq, 0.385 kg/kWh)",
                    "DINOv2-g 민감도: 탄소 집약도 × PUE"),
)
fig.add_trace(go.Bar(x=names, y=mwhs, marker_color=BLUE, showlegend=False,
                     text=[f"{v:.1f}" for v in mwhs], textposition="outside",
                     hovertemplate="%{x}<br>%{y:.2f} MWh<extra></extra>"), row=1, col=1)
fig.add_trace(go.Bar(x=names, y=tons, marker_color=AQUA, showlegend=False,
                     text=[f"{v:.1f}" for v in tons], textposition="outside",
                     hovertemplate="%{x}<br>%{y:.2f} tCO₂eq<extra></extra>"), row=1, col=2)
for (pue_name, _), color in zip(pues.items(), (BLUE, ORANGE)):
    idx = list(pues).index(pue_name)
    ys = [sens[ci][idx] for ci in carbon_intensity]
    fig.add_trace(go.Bar(name=pue_name, x=list(carbon_intensity), y=ys, marker_color=color,
                         text=[f"{v:.1f}" for v in ys], textposition="outside",
                         hovertemplate="%{x}<br>" + pue_name + "<br>%{y:.2f} tCO₂eq<extra></extra>"),
                  row=1, col=3)

fig.update_traces(marker_line_width=0, cliponaxis=False)
fig.update_layout(
    title="DINOv2-g 사전학습 1회: 22,016 GPU-hours × 400 W × PUE 1.1 = 9.7 MWh → 3.7 tCO₂eq (미국 평균 전력망)",
    barmode="group", bargap=0.35, template="plotly_white",
    width=1400, height=520, margin=dict(t=90, b=60),
    legend=dict(orientation="v", x=0.735, y=0.97, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)"),
)
fig.update_yaxes(title_text="MWh", row=1, col=1)
fig.update_yaxes(title_text="tCO₂eq", row=1, col=2)
fig.update_yaxes(title_text="tCO₂eq", row=1, col=3)
_show(fig)

png_path = os.path.join(HERE, "expy.png")
fig.write_image(png_path, scale=2)  # kaleido 필요
print("saved:", png_path)
# 출력:
# saved: /home/sungwoo/projects/swcho/dinov2/fm/papers/.fm/hints/a203b623-e288-476c-9dcd-fa80a191eca7/expy.png

# %% [markdown]
# ## 정리
#
# | 항목 | 값 |
# |---|---|
# | GPU | A100-40GB (전력 400 W = NVLink 시스템 TDP 가정) |
# | GPU-hours | **22,016** |
# | PUE | 1.1 |
# | 총 전력 | 22,016 × 400 × 1.1 / 10⁶ = **9.7 MWh** |
# | 탄소 배출 | 9.687 × 0.385 = **3.7 tCO₂eq** (미국 평균 0.385 kg/kWh) |
#
# 같은 조건에서 OpenCLIP ViT-L은 22.4 MWh, ViT-G는 118.9 MWh(≈ 12×) — 시각 특징만 필요하다면
# 자기지도 학습이 탄소 면에서 유리하다는 것이 논문의 결론. 프로젝트 전체는 약 200k GPU-days, 0.5k–1k tCO₂eq.
