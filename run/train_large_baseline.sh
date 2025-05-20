NUM_PROC=1
GPUS=0
DATE=0920
DATA=sample_data
MODE=your_exp_name
MODEL=baseline

ENCODERS=(PolarGate DeepGate2 GraphGPS Exphormer DAGformer GCN GraphSAGE GAT PNA) 
ENCODER=DeepGate2

python -u ./src/main.py \
 --exp_id ${DATE}_${MODEL}_${MODE}_${ENCODER} \
 --data_dir ./inmemory/${DATA} \
 --pkl_path ./${DATA} \
 --pretrained_model_path ./trained/model_last_workload.pth \
 --tf_arch ${MODEL} --hidden 128 --lr 1e-4  \
 --workload --gpus ${GPUS} --batch_size 1 --mini_batch_size 256 --epoch 200  \
 --skip_path --encoder ${ENCODER}

