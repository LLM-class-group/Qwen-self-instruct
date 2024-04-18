batch_dir=~/LLMs/lima/qwen_self_instruct/data/model_generations/

python3 ~/LLMs/lima/qwen_self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --input_file is_clf_or_not.jsonl \
    --output_file machine_generated_instances.jsonl \
    --num_instructions $INSTRUCTION_NUMS \
    --max_instances_to_gen 5 \
    --request_batch_size 5