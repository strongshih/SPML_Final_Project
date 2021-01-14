#! /bin/bash

for i in {1..40};
do
	python gen.py --bs=800 --teacher_weights=./ckpt.pth --r_feature_weight=10 --di_lr=0.1 --exp_descr="paper_parameters_better" --di_var_scale=0.001 --di_l2_scale=0.0
done
