#mAP: 0.9204 rank1: 0.9469 rank3: 0.9664 rank5: 0.9715 rank10: 0.9780 (Best: 0.9204 @epoch 4)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 160 --decay_type step_120_140 --loss 1*CrossEntropy+2*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_adam --nGPU 2  --lr 2e-4 --optimizer ADAM

#mAP: 0.9094 rank1: 0.9388 rank3: 0.9596 rank5: 0.9659 rank10: 0.9748 (Best: 0.9094 @epoch 4)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 160 --decay_type step_120_140 --loss 1*CrossEntropy+1*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_adam_1 --nGPU 2  --lr 1e-4 --optimizer ADAM

#mAP: 0.9217 rank1: 0.9460 rank3: 0.9653 rank5: 0.9706 rank10: 0.9801 (Best: 0.9217 @epoch 4)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 160 --decay_type step_120_140 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --re_rank --random_erasing --save MGN_adam_margin_1.2 --nGPU 2  --lr 2e-4 --optimizer ADAM

#mAP: 0.8986 rank1: 0.9356 rank3: 0.9567 rank5: 0.9620 rank10: 0.9727 (Best: 0.8986 @epoch 4)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 160 --decay_type step_120_140 --loss 1*CrossEntropy+2*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_adamax --nGPU 2  --lr 2e-4 --optimizer ADAMAX

#mAP: 0.5494 rank1: 0.7058 rank3: 0.7696 rank5: 0.8023 rank10: 0.8432 (Best: 0.5494 @epoch 4)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 160 --decay_type step_80_120 --loss 1*CrossEntropy+1*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_sgd --nGPU 2 --lr 1e-2 --optimizer SGD 

#mAP: 0.8480 rank1: 0.9008 rank3: 0.9317 rank5: 0.9436 rank10: 0.9555 (Best: 0.8480 @epoch 3)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 120 --decay_type step_60_80 --loss 1*CrossEntropy+1*Triplet --margin 0.3 --re_rank --random_erasing --save MGN_sgd_1 --nGPU 2 --lr 1e-2 --optimizer SGD 

#mAP: 0.8455 rank1: 0.9032 rank3: 0.9350 rank5: 0.9433 rank10: 0.9537 (Best: 0.8455 @epoch 3)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 40 --epochs 120 --decay_type step_60_80 --loss 1*CrossEntropy+1*Triplet --margin 1.2 --re_rank --random_erasing --save MGN_sgd_2 --nGPU 2 --lr 1e-2 --optimizer SGD 

#mAP: 0.8979 rank1: 0.9376 rank3: 0.9569 rank5: 0.9623 rank10: 0.9745 (Best: 0.8979 @epoch 200)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 50 --epochs 200 --decay_type step_130_170 --loss 1*CrossEntropy+1*Triplet --margin 1.2 --re_rank --random_erasing --save sgd_1 --nGPU 2 --lr 1e-2 --optimizer SGD --reset

#mAP: 0.8053 rank1: 0.9228 rank3: 0.9581 rank5: 0.9676 rank10: 0.9804 (Best: 0.8054 @epoch 190)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --reset --batchid 16 --batchtest 32 --test_every 10 --epochs 200 --decay_type step_240_250 --loss 1*CrossEntropy+1*Triplet --margin 1.2 --save sgd_2 --nGPU 2 --lr 1e-2 --optimizer SGD --save_models --random_erasing --reset

#mAP: 0.8251 rank1: 0.9353 rank3: 0.9679 rank5: 0.9783 rank10: 0.9866 (Best: 0.8251 @epoch 200)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --reset --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 10 --epochs 200 --decay_type step_240_250 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --random_erasing --save adam_1 --nGPU 2  --lr 2e-4 --optimizer ADAM --save_models

#mAP: 0.9097 rank1: 0.9442 rank3: 0.9614 rank5: 0.9679 rank10: 0.9751
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+1*Triplet --margin 1.2 --save sgd_3 --nGPU 2 --lr 1e-2 --optimizer SGD --save_models --random_erasing --reset --re_rank

#mAP: 0.9353 rank1: 0.9534 rank3: 0.9706 rank5: 0.9768 rank10: 0.9849
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save adam_2 --nGPU 2  --lr 2e-4 --optimizer ADAM --save_models --random_erasing --reset --re_rank

#mAP: 0.9174 rank1: 0.9433 rank3: 0.9617 rank5: 0.9679 rank10: 0.9754
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 20 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+1*Triplet --margin 1.2 --save sgd_3 --nGPU 2 --lr 1e-2 --optimizer SGD --random_erasing --reset --re_rank --nesterov

#mAP: 0.9376 rank1: 0.9558 rank3: 0.9712 rank5: 0.9765 rank10: 0.9816
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save adam_3 --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad

#mAP: 0.9323 rank1: 0.9513 rank3: 0.9700 rank5: 0.9745 rank10: 0.9813
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 100 --epochs 300 --decay_type step_250_290 --loss 1*CrossEntropy+2*Triplet --margin 0.3 --save adam_1 --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad

#mAP: 0.9270 rank1: 0.9510 rank3: 0.9691 rank5: 0.9751 rank10: 0.9810
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 50 --epochs 500 --decay_type step_300_420 --loss 1*CrossEntropy+1*Triplet --margin 1.2 --pool avg --save sgd_1 --nGPU 2 --lr 1e-2 --optimizer SGD --random_erasing --reset --re_rank --nesterov

#0.9383 rank1: 0.9578 rank3: 0.9721 rank5: 0.9783 rank10: 0.9843 (Best: 0.9383 @epoch 400)
#CUDA_VISIBLE_DEVICES=2,3 python3 main.py --datadir ../reid-mgn/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save adam_1 --nGPU 2  --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad