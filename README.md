# Multiple Granularity Network
Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- TorchVision
- Matplotlib
- Argparse
- Sklearn
- Pillow
- Numpy
- Scipy
- Tqdm

## Train

### Prepare training data

Download Market1501 training data.[here](http://www.liangzheng.org/Project/project_reid.html)

### Begin to train

In the demo.sh file, add the Market1501 directory to --datadir

run `sh demo.sh`

##  Result

|  | mAP | rank1 | rank3 | rank5 | rank10 |
| :------: | :------: | :------: | :------: | :------: | :------: |
| 2018-7-22 | 92.17 | 94.60 | 96.53 | 97.06 | 98.01 |
| 2018-7-24 | 93.53 | 95.34 | 97.06 | 97.68 | 98.49 |
| last | 93.83 | 95.78 | 97.21 | 97.83 | 98.43 |

Download model file in [here](https://pan.baidu.com/s/1DbZsT16yIITTkmjRW1ifWQ)


## The architecture of Multiple Granularity Network (MGN)
![Multiple Granularity Network](https://pic2.zhimg.com/80/v2-90a8763a0b7aa86d9152492eb3f85899_hd.jpg)

Figure . Multiple Granularity Network architecture.

```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1804.01438},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = apr,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```