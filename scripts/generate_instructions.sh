batch_dir=my_self_instruct/data/model_generations/

python3 my_self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 100 \
    --seed_tasks_path my_self_instruct/data/seed_tasks.jsonl \
