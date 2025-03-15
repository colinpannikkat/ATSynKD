python train.py -dataset tiny-imagenet -kd -klat -llambda 0.98 -alpha 0.9 -weights tinyimagenet_teacher_train15.pt -n 50 -lr 0.1 -batch 128 -epochs 200 -sgd -momentum 0.9 -weight_decay 1e-4 -augment -reducer -lr_args '{"factor" : 0.1, "patience": 5}' -name atkd_sched_b128_e200_lr1_reduce_sgd_l98_a9