
MASTER_ADDR=127.0.0.1
MASTER_PORT=8915
export OMP_NUM_THREADS=3
WORLD_SIZE=4
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE  --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
ADD_POS=0
ADD_NER=0
ADD_PROMPT=0
SIZE=base
LR=1.5e-4

# LOAD_MODE=pretrain_$SIZE/step43000
LOAD_MODE=/nas2/lishengping/other_models/gpt2-chinese/$SIZE
TRAIN_PATH=/nas2/qzj/datas/med_data/data_with_pos_ner/train.jsonl
EVAL_PATH=/nas2/qzj/datas/med_data/data_with_pos_ner/test.jsonl

current_year=$(date +%Y)
current_month=$(date +%m)
current_day=$(date +%d)
current_hour=$(date +%H)

LOG_NAME="${current_year}_${current_month}_${current_day}_${current_hour}.txt"


if [[ $1 == 'train' ]]; then
    CUDA_VISIBLE_DEVICES=1,2,4,5 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py  --train_data_path $TRAIN_PATH --eval_data_path $EVAL_PATH --epochs 10  --log_step 2 --output_dir pretrain_$SIZE/  --perrank_batch_size 4 --gradient_accumulation 4 --lr $LR --warmup_steps 1000 --pretrained_model $LOAD_MODE --save_model_interval_steps 5000 --save_model --eval_step 5000 --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16 --use_rotary_emb 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
elif [[ $1 == 'eval' ]]; then
    CUDA_VISIBLE_DEVICES=1,2,4,5 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py --eval_data_path $EVAL_PATH --log_step 2 --perrank_batch_size 8 --pretrained_model $LOAD_MODE --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16 --eval 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
elif [[ $1 == 'test_train' ]]; then
    CUDA_VISIBLE_DEVICES=1,2,4,5 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py  --train_data_path $TRAIN_PATH --eval_data_path $EVAL_PATH data_with_pos_ner/CMDD.jsonl --epochs 10  --log_step 2 --output_dir test_$SIZE/  --perrank_batch_size 4 --gradient_accumulation 4 --lr $LR --warmup_steps 1000 --pretrained_model $LOAD_MODE  --save_model_interval_steps 1000 --eval_step 1000 --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16 --use_rotary_emb --use_position_emb 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
elif [[ $1 == 'test_eval' ]]; then
    CUDA_VISIBLE_DEVICES=1,2,4,5 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_ddp.py --eval_data_path $EVAL_PATH --log_step 2 --perrank_batch_size 8 --pretrained_model $LOAD_MODE --add_pos $ADD_POS --add_ner $ADD_NER --add_prompt $ADD_PROMPT --fp16 --eval --use_rotary_emb --use_position_emb 2>&1 | tee logs/$SIZE.$LOG_NAME.$1.log
else
    echo "Unknow parameter"
fi