import os
import json
import random
import tqdm
import re
import time
import argparse
import pandas as pd
from collections import OrderedDict
from templates.clf_task_template import template_1
from templates.clf_task_template_short import template_2
from api.qwen_1__8B_api import response


random.seed(time.time())

sample_prompts = ""


templates = {
    "template_1": template_1,
    "template_2": template_2
}

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", type=str, default="template_1", help="Which template to use.")
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument( #the number of instructions to classification
        "--num_instructions",
        type=int,
        default=20,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="template_1",
        help="Which template to use. Currently only `template_1` is supported.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin: #read instructions in lines
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
    print("Total instruction nums: ", len(lines))

    output_path = os.path.join(args.batch_dir, f"is_clf_or_not.jsonl") #output file
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing instructions that have been classified before.")

    yes_success = 0 # identified yes classification instructions
    no_success = 0 # identified no classification instructions
    idx = 0


    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size): #process data in batches
            print(f"Processing batch {batch_idx} to {batch_idx + args.request_batch_size}")
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            if all(d["instruction"] in existing_requests for d in batch): #check if all instructions in the batch are already processed
                idx += len(batch)
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = templates[args.template]
                prompts = [prefix + " " + d["instruction"].strip() +
                           "\n" + "Is it classification?" for d in batch]
                results = []
                for prompt in prompts:
                    # print("-----------prompt is: ---------------\n",prompt)
                    idx += 1
                    if idx < 5:
                        sample_prompts += f"prompt {idx}:" + prompt +"\n"
                    result = response(prompt, 5)
                    if result.strip():
                        first_result_word = result.split()[0]
                    else:
                        first_result_word = "empty result!"
                    if (first_result_word == "no" or first_result_word == "No" or first_result_word == "NO" or first_result_word =="not" or first_result_word =="Not" or first_result_word =="NOT" or first_result_word == "negative" or first_result_word == "Negative" or first_result_word == "NEGATIVE" or first_result_word == "不是" or first_result_word == "false"): 
                        no_success += 1
                        result="no"
                    elif (first_result_word == "yes" or first_result_word == "Yes" or first_result_word == "YES" or first_result_word == "positive" or first_result_word == "Positive" or first_result_word == "POSITIVE" or first_result_word == "是" or first_result_word == "true"): 
                        yes_success += 1
                        result="yes"
                    else: result=None
                    print("-----------classify result {}: ---------------\n".format(idx),result)
                    results.append(result)
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i] is not None:
                        data = {
                        "instruction": data["instruction"],
                        "is_classification": results[i]
                        }
                        data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "is_classification"]
                        )
                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))

    print(f"Processed {idx} instructions, {yes_success+no_success} of them are classified successfully.")
    print(f"Identification rate: {(yes_success+no_success) / idx}")



