
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --datadir Market1501_dir --reset --epochs 160 --test_every 20 --batchid 16 --batchimage 4 --nGPU 2 --nThread --save MGN_save