#!/bin/bash

date # print begin time

export INSTRUCTION_NUMS=5000
export BATCH_SIZE=1
export DATA_GEN_PATH=~/LLMs/lima/qwen_self_instruct/data/model_generations2/

echo "-------------------------Running generate_instructions: -------------------------"
./generate_instructions.sh 
echo "-------------------------Running classifications: -------------------------"
./is_clf_or_not.sh 
echo "-------------------------finish pipeline1 -----------------------------"

date # print end time
