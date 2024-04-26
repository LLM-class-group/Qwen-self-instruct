# Qwen Self-Instruct

## 项目结构

- `api/`: 封装了本地模型（Qwen1.5）和其他LLMs（如OpenAI API）的统一调用接口
- `data/`: 数据目录
- `scripts/`: 脚本目录
- `templates/`: 模板目录
- `history_data.jsonl`: 多轮对话数据
- `merge.py`: 合并和启发式过滤程序
- `bootstrap_instructions.py`: 指令生成程序
- `identify_clf_or_not.py`: 指令分类程序
- `generate_instances.py`: 实例生成程序
- `prepare_for_finetuning.py`: 过滤和格式化程序

## 使用方法

生成候选数据：

运行`./scripts/pipeline.sh`，在`data/model_generations?/`下生成指令精调的候选数据

合并和启发式过滤：

运行`python3 merge.py`，合并所有`data/model_generations?/`下的数据并启发式过滤，同时会加入`history_data.jsonl`中的数据

我们的数据：

- `data/final_data/qwen_self_instruct_full.jsonl` 我们生成的所有候选数据
- `data/final_data/qwen_self_instruct.jsonl` 我们最终指令微调使用的数据
