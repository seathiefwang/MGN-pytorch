
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --datadir Market1501_dir --reset --re_rank --random_erasing --margin 1.2 --optimizer ADAM --epochs 160 --decay_type step_120_140 --test_every 20 --nGPU 2 --save MGN_save

