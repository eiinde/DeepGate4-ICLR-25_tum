# DeepGate4: Efficient and Effective Representation Learning for Circuit Design at Scale

## Overview
Our paper is avaiable at [Arixv](https://www.arxiv.org/abs/2502.01681) and [OpenReview](https://openreview.net/forum?id=b10lRabU9W).
![Overall Pipeline](./Overall_pipline.png)

## Environment
To install the library `deepgate`, please refer to [python-deepgate](https://github.com/zshi0616/python-deepgate).

## Dataset Preparation
We provide sample raw data and corresponding processed data in `./raw_data` and `./raw_sample_data` respectively.

**To prepare your own data:**
* 1. `cd ./simulator` and `bash ./build.sh`
* 2. prepare your own raw aig data in `./YOUR_RAW_DATA`
* 3. `python ./src/dg_dataset/data_preparation.py --aig_dir ./YOUR_RAW_DATA --save_path ./YOUR_DATASET_DIR`
    
## Training 
You can run experiment with `./run/train_large.sh` and `./run/train_large_baseline.sh`
* `./run/train_large.sh` denotes running model with our updating strategy
* `./run/train_large_baseline.sh` denotes running model with its original strategy

We further offer various baseline models:
* `./run/train_large.sh` offers models with **baseline**(DeepGate2), **plain**(DeepGate3), **sparse**(DeepGate4), **GraphGPS**, **Exphormer** and **DAGformer**
* `./run/train_large_baseline.sh` offers models with **PolarGate**, **DeepGate2**, **GraphGPS**, **Exphormer**, **DAGformer**, **GCN**, **GraphSAGE**, **GAT** and **PNA**

