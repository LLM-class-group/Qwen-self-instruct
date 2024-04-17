import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from templates.clf_task_template import template_1
from qwen_api import response


random.seed(42)


templates = {
    "template_1": template_1
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
    print("instruction nums: ", len(lines))

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

    success = 0 #number of instructions that is classified successfully
    yes_correct = 0 #number of classification instructions that is classified correctly
    no_correct = 0 #number of not classification instructions that is classified correctly
    yes_success = 0 # identified yes classification instructions
    no_success = 0 # identified no classification instructions
    idx = 0

    # strange_prompts = ""

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
                    result = response(prompt, 30)
                    # if(idx <= 5): strange_prompts += "prompt"+str(idx)+ ":\n"+prompt+"\n\n\n"
                    if result.strip():
                        first_result_word = result.split()[0]
                    else:
                        first_result_word = "empty result!"
                    # print("-----------classification result of : ---------------\n",result)
                    if (first_result_word == "no" or first_result_word == "No" or first_result_word == "NO" or first_result_word == "negative" or first_result_word == "Negative" or first_result_word == "NEGATIVE" or first_result_word == "不是" or first_result_word == "false"): 
                        success += 1
                        if(idx <= 148): 
                            no_correct += 1
                            no_success += 1
                        else:
                            yes_success += 1    
                    if (first_result_word == "yes" or first_result_word == "Yes" or first_result_word == "YES" or first_result_word == "positive" or first_result_word == "Positive" or first_result_word == "POSITIVE" or first_result_word == "是" or first_result_word == "true"): 
                        success += 1
                        if (idx > 148): 
                            yes_correct += 1
                            yes_success += 1
                        else:
                            no_success += 1
                    print("-----------first word of instruction {}: ---------------\n".format(idx),first_result_word)
                    results.append(first_result_word)
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i] is not None:
                        data["is_classification"] = results[i]
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))

    print(f"Processed {idx} instructions, {success} of them are classified successfully.")
    print(f"Identification rate: {success / idx}")
    epsilon = 0.0001
    print(f"Classification problem correct rate: {yes_correct / (yes_success + epsilon)}")
    print(f"Not classification problem correct rate: {no_correct / (no_success + epsilon)}")
    # with open("../clf_prompt_1-5.txt", "w") as file:
    #      print("open file successfully!")
    #      file.write(strange_prompts)

