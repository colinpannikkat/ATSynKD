python tune_hyperparams.py -dataset tiny-imagenet -kd -ssimat -weights ~/ATSynKD/tinyimagenet_teacher_train15.pt -n 50 -lr 0.1 -batch 128 -epochs 200 -sgd -momentum 0.9 -weight_decay 1e-4 -augment -reducer -lr_args '{"factor" : 0.5, "patience": 5}' -name tune_ll_a_ssimkd_sched_b128_e200_lr1_reduce_sgd