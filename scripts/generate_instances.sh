batch_dir=my_self_instruct/data/model_generations/

python3 my_self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --input_file is_clf_or_not.jsonl \
    --output_file machine_generated_instances_test2.jsonl \
    --max_instances_to_gen 5 \
    --request_batch_size 5