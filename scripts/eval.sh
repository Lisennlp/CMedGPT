
MASTER_ADDR=127.0.0.1
MASTER_PORT=8912
export OMP_NUM_THREADS=3
WORLD_SIZE=4
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE  --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
ADD_POS=0
ADD_NER=0
ADD_PROMPT=0
SIZE=base
LR=1.5e-4

# LOAD_MODEL=pretrain_$SIZE/step43000
LOAD_MODEL=/nas2/lishengping/other_models/gpt2-chinese/$SIZE
# rotary emb
# LOAD_MODEL=/nas2/lishengping/caiyun_projects/gpt2-chinese/pretrain_t2023_12_06_13_pos0_ner0_prompt0_Rotary_base/step7000
# position emb
# LOAD_MODEL=/nas2/lishengping/caiyun_projects/gpt2-chinese/pretrain_t2023_12_05_12_pos0_ner0_prompt0_base/step20000
# LOAD_MODEL=/nas2/lishengping/caiyun_projects/gpt2-chinese/pretrain_t2023_12_06_12_pos0_ner0_prompt0_base/step1000_pos

# TRAIN_PATH=/nas2/lishengping/datas/med_data/data_with_pos_ner/1205/train.jsonl
TRAIN_PATH=/nas2/lishengping/datas/med_data/data_with_pos_ner/train.jsonl
EVAL_PATH=/nas2/lishengping/datas/med_data/data_with_pos_ner/test.jsonl

# TRAIN_PATH=/nas2/lishengping/datas/med_data/data_with_pos_ner/IMCS.jsonl
# EVAL_PATH=/nas2/lishengping/datas/med_data/data_with_pos_ner/IMCS.jsonl

current_year=$(date +%Y)
current_month=$(date +%m)
current_day=$(date +%d)
current_hour=$(date +%H)
# POS_EMB=Rotary
POS_EMB=Postion
MAX_LEN=1536
skip_step=0
BASE=10000
# OUT_DIR=pretrain_t"${current_year}_${current_month}_${current_day}_${current_hour}"_pos"$ADD_POS"_ner"$ADD_NER"_prompt"$ADD_PROMPT"_"$POS_EMB"_"$SIZE"/
# LOG_NAME="${current_year}_${current_month}_${current_day}_${current_hour}$POS_EMB"
LOG_NAME="$skip_step"step_"$POS_EMB"_"$MAX_LEN"_"$BASE"

# 512: 5118, 1024: 4590, 1536:3259, 2048: 2317, 3072:
if [[ $1 == 'train' ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py  --train_data_path $TRAIN_PATH --eval_data_path $EVAL_PATH --epochs 10  --log_step 2 --output_dir $OUT_DIR  --perrank_batch_size 4 --gradient_accumulation 1 --lr $LR --warmup_steps 1000 --pretrained_model $LOAD_MODEL --save_model_interval_steps 1000 --save_model --eval_step 1000 --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16  --use_rotary_emb 2>&1 --max_len 512 --skip_step $skip_step| tee logs/$SIZE.$LOG_NAME.$1.log
elif [[ $1 == 'eval' ]]; then
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py --eval_data_path $EVAL_PATH --log_step 2 --perrank_batch_size 1 --pretrained_model $LOAD_MODEL --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16 --eval --use_position_emb --max_len $MAX_LEN 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
elif [[ $1 == 'test_train' ]]; then
    CUDA_VISIBLE_DEVICES=1,2,4,5 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py --train_data_path $TRAIN_PATH --eval_data_path $EVAL_PATH --epochs 10  --log_step 2 --output_dir test_$SIZE/  --perrank_batch_size 4 --gradient_accumulation 4 --lr $LR --warmup_steps 1000 --pretrained_model $LOAD_MODEL  --save_model_interval_steps 1000 --eval_step 50 --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16  --use_position_emb 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
elif [[ $1 == 'test_eval' ]]; then
    CUDA_VISIBLE_DEVICES=1,2,4,5 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py --eval_data_path $EVAL_PATH --log_step 2 --perrank_batch_size 8 --pretrained_model $LOAD_MODEL --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16 --eval --use_rotary_emb  --use_position_emb 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
else
    echo "Unknow parameter"
fi