#!/bin/bash

iter=30000
i=1
splmda=0
layer=1234


# Flower
lr=5e-3
wd=1e-4
mmt=0.9
lmda=1e-1
python -u train.py --iterations ${iter} --datapath /data/Flower_102/ --dataset Flower102Data --name resnet18_flower102_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i} --batch_size 64 --feat_lmda ${lmda} --lr ${lr} --network resnet18 --weight_decay ${wd} --beta 1e-2 --test_interval 1000 --feat_layers ${layer} --momentum ${mmt} --log

python -u eval_robustness.py --datapath /data/Flower_102/ --dataset Flower102Data --network resnet18 --checkpoint ckpt/resnet18_flower102_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.pth > eval/resnet18_flower102_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.log


# Indoor
lr=1e-2
wd=1e-4
mmt=0
lmda=5e-1
python -u train.py --iterations ${iter} --datapath /data/MIT_67/ --dataset MIT67Data --name resnet18_mit67_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i} --batch_size 64 --feat_lmda ${lmda} --lr ${lr} --network resnet18 --weight_decay ${wd} --beta 1e-2 --test_interval 1000 --feat_layers ${layer} --momentum ${mmt} --log

python -u eval_robustness.py --datapath /data/MIT_67/ --dataset MIT67Data --network resnet18 --checkpoint ckpt/resnet18_mit67_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.pth > eval/resnet18_mit67_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.log


# Stanford40
lr=5e-3
wd=0
mmt=0
lmda=1e0
python -u train.py --iterations ${iter} --datapath /data/stanford_40/ --dataset Stanford40Data --name resnet18_stanford40_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i} --batch_size 64 --feat_lmda ${lmda} --lr ${lr} --network resnet18 --weight_decay ${wd} --beta 1e-2 --test_interval 1000 --feat_layers ${layer} --momentum ${mmt} --log

python -u eval_robustness.py --datapath /data/stanford_40/ --dataset Stanford40Data --network resnet18 --checkpoint ckpt/resnet18_stanford40_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.pth > eval/resnet18_stanford40_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.log



# CUB200
lr=1e-2
wd=0
mmt=0
lmda=1e-1
python -u train.py --iterations ${iter} --datapath /data/CUB_200_2011/ --dataset CUB200Data --name resnet18_cub200_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i} --batch_size 64 --feat_lmda ${lmda} --lr ${lr} --network resnet18 --weight_decay ${wd} --beta 1e-2 --test_interval 1000 --feat_layers ${layer} --momentum ${mmt} --log

python -u eval_robustness.py --datapath /data/CUB_200_2011/ --dataset CUB200Data --network resnet18 --checkpoint ckpt/resnet18_cub200_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.pth > eval/resnet18_cub200_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.log


# DOG
lr=1e-2
wd=0
mmt=0
lmda=1e1
python -u train.py --iterations ${iter} --datapath /data/stanford_dog/ --dataset SDog120Data --name resnet18_sdog120_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i} --batch_size 64 --feat_lmda ${lmda} --lr ${lr} --network resnet18 --weight_decay ${wd} --beta 1e-2 --test_interval 1000 --feat_layers ${layer} --momentum ${mmt} --log

python -u eval_robustness.py --datapath /data/stanford_dog/ --dataset SDog120Data --network resnet18 --checkpoint ckpt/resnet18_sdog120_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.pth > eval/resnet18_sdog120_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.log
