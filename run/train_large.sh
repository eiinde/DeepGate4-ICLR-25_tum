NUM_PROC=1
GPUS=0
DATE=0923
DATA=sample_data
MODE=${DATA}

MODELs=(baseline sparse plain GraphGPS Exphormer DAGformer)
MODEL=DAGformer

python -u ./src/main.py \
 --exp_id ${DATE}_${MODEL}_${MODE} \
 --data_dir ./inmemory/${DATA} \
 --pkl_path ./${DATA} \
 --pretrained_model_path ./trained/model_last_workload.pth \
 --tf_arch ${MODEL} --hidden 128 --lr 1e-4 --enable_cut \
 --workload --gpus ${GPUS} --batch_size 1 --mini_batch_size 256 --epoch 200 

