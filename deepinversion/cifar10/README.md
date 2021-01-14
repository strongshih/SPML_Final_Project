```
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install numpy
pip install Pillow

wget https://csie.ntu.edu.tw/~r09922028/ckpt.pth
python gen.py --bs=256 --teacher_weights=./ckpt.pth --r_feature_weight=10 --di_lr=0.1 --exp_descr="paper_parameters_better" --di_var_scale=0.001 --di_l2_scale=0.0
bash ./run.sh
```
