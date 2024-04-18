batch_dir=~/LLMs/lima/qwen_self_instruct/data/model_generations/

python3 ~/LLMs/lima/qwen_self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate $INSTRUCTION_NUMS \
    --seed_tasks_path ~/LLMs/lima/qwen_self_instruct/data/seed_tasks.jsonl \
