# 로컬 환경 설정 (conda `trellis` 재사용)

업스트림 `conda.yaml`(python 3.9 / torch 2.0.0+cu117 / xformers 0.0.18)은 이 머신의
기존 `trellis` 환경과 맞지 않아, **별도 환경을 만들지 않고 `trellis`에 dinov2를 얹었다.**

- env: `/home/sungwoo/miniforge3/envs/trellis` (python 3.10.21)
- torch 2.4.0+cu121 / torchvision 0.19.0+cu121 / xformers 0.0.27.post2 (기존 그대로, 건드리지 않음)
- GPU: RTX 3090, driver 580.173.02, nvcc 12.1 (env 내장)

## 설치된 것

```bash
conda activate trellis

# 1) dinov2 코어 의존성 (torch/torchvision/xformers/numpy는 constraint로 고정하여 변경 금지)
pip install omegaconf fvcore iopath submitit "torchmetrics>=1.4,<2" "ipykernel<7"

# 2) dinov2 패키지 자체는 pip install 대신 .pth 로 경로만 추가
#    (setup.py 가 torch==2.0.0 등 옛 핀을 갖고 있어 pip 의존성 그래프를 오염시키기 때문)
echo "/home/sungwoo/projects/swcho/dinov2" \
  > /home/sungwoo/miniforge3/envs/trellis/lib/python3.10/site-packages/dinov2-dev.pth

# 3) notebooks 용 (depth / segmentation) — mmcv-full 은 CUDA ops 포함해 소스 빌드
pip install "setuptools==80.9.0"            # 84.0.0 은 pkg_resources 제거 → mmcv 빌드 불가
pip install --no-build-isolation "mmsegmentation==0.30.0" "yapf==0.40.1"
pip uninstall -y mmcv                        # mmseg 가 끌고 오는 lite 버전 제거
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" \
  CUDA_HOME=$CONDA_PREFIX MAX_JOBS=8 \
  pip install --no-build-isolation --no-deps "mmcv-full==1.7.2"
# 빌드된 wheel 은 ~/.cache/pip/wheels 에 캐시되어 재설치 시 재컴파일 불필요

# 4) jupyter 커널
python -m ipykernel install --user --name dinov2-trellis --display-name "Python (trellis / dinov2)"
```

## 업스트림 핀과 다른 점 / 이유

| 항목 | 업스트림 | 여기 | 이유 |
|---|---|---|---|
| python | 3.9 | 3.10 | trellis 환경 그대로 |
| torch | 2.0.0+cu117 | 2.4.0+cu121 | trellis 환경 그대로, 다운그레이드하면 trellis가 깨짐 |
| torchmetrics | 0.10.3 | 1.9.0 | 0.10.3 은 `pkg_resources` 를 import → setuptools 최신에서 동작 불가. 저장소가 쓰는 API(`Metric`, `MetricCollection`, `Multiclass*`, `dim_zero_cat`, `select_topk`, `MetricTracker`)는 1.x 에도 그대로 있음 |
| mmcv-full | 1.5.0 | 1.7.2 | 1.5.0 은 torch 2.x 에서 빌드 불가 |
| mmsegmentation | 0.27.0 | 0.30.0 | mmcv 1.7 을 허용하는 최소 버전 |
| cuml-cu11 | 설치 | **미설치** | CUDA 11 전용이라 cu121 과 불일치. `dinov2/eval/log_regression.py` 에서만 사용 |
| setuptools | - | 80.9.0 | 84.0.0 에서 `pkg_resources` 제거됨 (mmcv 빌드/구 패키지 import 에 필요) |

## 검증 완료 (실제로 돌려본 것)

- `torch.hub.load('.', 'dinov2_vits14', source='local', pretrained=True)` → CLS `[2,384]`, patch tokens `[2,256,384]`
- `notebooks/depth_estimation.ipynb` 전체 경로 (vits14 + NYU DPT head) → depth map `(480,640)`, min 0.965 / max 4.763
- `notebooks/semantic_segmentation.ipynb` linear head 경로 (vits14 + VOC2012) → seg map `(480,640)`
- `mmcv.ops` CUDA 커널 (`sigmoid_focal_loss`, `MultiScaleDeformableAttention`) 정상 로드
- `dinov2.eval.{depth,segmentation,segmentation_m2f,knn,linear,metrics}` import 정상
- `pip check` 클린 — trellis 기존 패키지 의존성 깨짐 없음

## 사용법

```bash
conda activate trellis
cd /home/sungwoo/projects/swcho/dinov2
python -c "import torch; m = torch.hub.load('.', 'dinov2_vits14', source='local'); print(m(torch.randn(1,3,224,224)).shape)"
```

노트북은 커널 `Python (trellis / dinov2)` 선택.

## 알려진 제약

- `dinov2/eval/log_regression.py` 는 `cuml` 필요 → 현재 미설치. 필요하면
  `pip install --extra-index-url https://pypi.nvidia.com cuml-cu12` (cu11 아님) 로 시도.
- 학습(`dinov2/train`)은 SLURM + submitit 전제. 단일 GPU 로는
  `torchrun --nproc_per_node=1 dinov2/train/train.py ...` 형태로 직접 실행해야 함.
