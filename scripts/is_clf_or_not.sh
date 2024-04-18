batch_dir=/home/jiahe/LLMs/lima/my_self_instruct/data/model_generations/
output=/home/jiahe/LLMs/lima/my_self_instruct/data/model_generations/is_clf_or_not_test.jsonl

if [ -f "$output" ]; then
    > $output
fi

python /home/jiahe/LLMs/lima/my_self_instruct/identify_clf_or_not.py \
    --batch_dir ${batch_dir} \
    --num_instructions 175 \
    --template template_2 \
    --request_batch_size 20 \
    