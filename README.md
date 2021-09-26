# Ranking-Synthetic-Features-for-Generative-Zero-Shot-Learning

Ranking Synthetic Features for Generative Zero-Shot Learning [paper](https://ieeexplore.ieee.org/document/9420574)

## Data
The code uses the ResNet101 features provided by the paper: Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly, and follows its GZSL settings.

The features can be download here [data](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) .

## Run
python3 "GAN.py" --ratiox 0.1 --l_gamma 0.55 --u_gamma 0.99 --proto_param1 0 --proto_param2 0 --dataroot "xlsa17/data" --use_pretrain_s 0 --ensemble_ratio 0.5 --loss_syn_num 15 --cyc_seen_weight 1 --cyc_unseen_weight 1e-2 --dm_seen_weight 0.001 --dm_unseen_weight     1 --dm_weight 3e-2  --cls_syn_num 850  --cls_batch_size 750  --new_lr 0   --nepoch 60  --manualSeed 9182 --cls_weight 0.01 --preprocessing --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa

## Ciation

```
@inproceedings{Ramazi,
  title={Ranking Synthetic Features for Generative Zero-Shot Learning},
  author={Ramazi, Shayan and Nadian, Ali},
  booktitle={International Computer Conference, Computer Society of Iran (CSICC)},
  year={2021}
}
```
