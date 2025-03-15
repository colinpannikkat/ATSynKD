python train.py -dataset tiny-imagenet -small -n -1 -lr 0.1 -batch 128 -epochs 200 -sgd -momentum 0.9 -weight_decay 1e-4 -augment -reducer -warmup -scheduler multistep -lr_args '{"milestones" : [20, 50, 110, 140], "gamma" : 0.1, "factor" : 0.5}' -name base_sched_b128_e200_sched_lr1_reduce_sgd_student