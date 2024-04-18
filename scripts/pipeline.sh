#!/bin/bash

export INSTRUCTION_NUMS=20
export BATCH_SIZE=1

# 定义颜色和样式
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

echo "-------------------------Running generate_instructions: -------------------------"
./generate_instructions.sh 
echo "-------------------------Running classifications: -------------------------"
./is_clf_or_not.sh 
echo "-------------------------Running generate_instances: -------------------------"
./generate_instances.sh 
echo "-------------------------Running prepare_for_finetuning: -------------------------"
./prepare_for_finetuning.sh 
echo "-------------------------finish pipline-------------------------"

