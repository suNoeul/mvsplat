#!/bin/bash
#SBATCH -J mvsplat-eval              # Job 이름
#SBATCH -o logs/mvsplat-eval.%j.out  # 표준 출력 로그 파일
#SBATCH -t 01:00:00                  # 최대 실행 시간
#SBATCH -p titanxp                   # 사용할 GPU 파티션
#SBATCH --gres=gpu:1                 # 사용할 GPU 수
#SBATCH --mem=16G                    # 메모리 용량

# Slurm 시스템 정보 출력
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_JOB_ID: $SLURM_JOB_ID"

# Conda 환경 설정
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mvsplat

# CUDA 라이브러리 우선 적용 (중요)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

# 작업 디렉토리로 이동
cd /home/rlatnsgh0708/research_project/mvsplat

# 평가 실행 (RealEstate10K 사전학습 모델 사용)
python -m src.main +experiment=re10k \
  checkpointing.load=checkpoints/re10k.ckpt \
  mode=test \
  dataset/view_sampler=evaluation \
  test.compute_scores=true

conda deactivate
echo "=== JOB FINISHED ==="
