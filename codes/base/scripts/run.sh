#!/usr/bin/env bash
#SBATCH --mem=20G                 # memory
#SBATCH --nodelist=compute001                 # memory
#SBATCH --gres=gpu:1              # Number of GPU(s): 1 for DTW, 3 for Feature extract.
#SBATCH --time=10-00:00:00          # time (DD-HH:MM:SS) 3 days by default; 5-00:00:00 (5 DAYS) / UNLIMITED;
#SBATCH --ntasks=1                # Number of "tasks”/#nodes (use with distributed parallelism).
#SBATCH --cpus-per-task=4         # Number of CPUs allocated to each task/node (use with shared memory parallelism).

hostname
whoami
echo "//////////////////////////////"

echo 'CUDA_VISIBLE_DEVICES:'
echo $CUDA_VISIBLE_DEVICES
echo "//////////////////////////////"

# name='10scifar100_ema_scaleonstu_scalelr100init-10_innerlr2_alphadiv4_noresetft'
# name='10scifar100_ema_alphadiv4_innerlr2_truncscalezerolinear_metalr1x'
# name='10scifar100_ema_alphadiv4on70-t-2x5__0325ole00_proj_1e-2reg'
# name='10scifar100_ema_alphadiv4on70-t-2x5_0325ole00_proj_lossemaonld'
# name='10scifar100_reg'
# name='b50_10splitcifar100_ema_alphadiv4on40_lrl1230.125_tr2'
name='base2022'
# name='20scifar100_ema_alphadiv4on70-t-2x5_0325ole_tr4_lr_t3l120.5_t4l1230.5_last'
# name='10simagenet_ema_0325ole_lr_t3l120.5_t3l1230.5_v2'
debug='1'
comments='None'
expid='2'




  # CUDA_VISIBLE_DEVICES=3 python -m main train with "../../exps/der_womask/imagenet-100/b0_10s/configs/1.yaml" \
  #       exp.name="${name}" \
  #       exp.savedir="./logs/" \
  #       exp.ckptdir="./logs/" \
  #       exp.tensorboard_dir="./tensorboard/" \
  #       exp.debug=True \
  #       --name="${name}" \
  #       -D \
  #       -p \
  #       --force \
  #       #--mongo_db=10.10.10.100:30620:debug

# cifar100
  CUDA_VISIBLE_DEVICES=4 python -m main train with "./configs/2.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        exp.debug=True \
        --name="${name}" \
        -D \
        -p \
        --force \
        #--mongo_db=10.10.10.100:30620:debug



# # cifar100 ablation
#   CUDA_VISIBLE_DEVICES=7 python -m main train with "./configs/ablation.yaml" \
#         exp.name="${name}" \
#         exp.savedir="./logs/" \
#         exp.ckptdir="./logs/" \
#         exp.tensorboard_dir="./tensorboard/" \
#         exp.debug=True \
#         --name="${name}" \
#         -D \
#         -p \
#         --force \
#         #--mongo_db=10.10.10.100:30620:debug
