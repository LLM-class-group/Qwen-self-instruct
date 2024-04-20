#!/bin/bash

date # print begin time

export INSTRUCTION_NUMS=5000
export BATCH_SIZE=1
export DATA_GEN_PATH=~/LLMs/lima/qwen_self_instruct/data/model_generations3/

echo "-------------------------Running generate_instances: -------------------------"
./generate_instances.sh 
echo "-------------------------Running prepare_for_finetuning: -------------------------"
./prepare_for_finetuning.sh 
echo "-------------------------finish pipeline2 ---------------------------"

date # print end time
