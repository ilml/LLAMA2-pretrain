#python3  run_clm.py \
deepspeed --num_gpus=1  run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --output_dir ./tmp \
    --overwrite_output_dir true\
    --deepspeed ds_config.json \
    --fp16 true 
