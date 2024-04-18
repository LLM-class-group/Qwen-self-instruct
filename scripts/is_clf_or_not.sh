batch_dir=~/LLMs/lima/qwen_self_instruct/data/model_generations/
output=~/LLMs/lima/qwen_self_instruct/data/model_generations/is_clf_or_not.jsonl

if [ -f "$output" ]; then
    > $output
fi

python ~/LLMs/lima/qwen_self_instruct/identify_clf_or_not.py \
    --batch_dir ${batch_dir} \
    --num_instructions $INSTRUCTION_NUMS \
    --template template_2 \
    --request_batch_size $BATCH_SIZE \
    