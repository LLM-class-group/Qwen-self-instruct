batch_dir=~/LLMs/lima/qwen_self_instruct/data/model_generations

python ~/LLMs/lima/qwen_self_instruct/prepare_for_finetuning.py \
    --instance_files ${batch_dir}/machine_generated_instances.jsonl \
    --classification_type_files ${batch_dir}/is_clf_or_not.jsonl \
    --output_dir ${batch_dir}/finetuning_data \
    --seed_tasks_path ~/LLMs/lima/qwen_self_instruct/data/seed_tasks.jsonl