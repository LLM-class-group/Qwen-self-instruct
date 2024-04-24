# merge all generated instances to one with extra filtering rules
# also concatenate the history data to the end of the file 

import json
import random

input_filter_list = ["program", "function", "python", "code", "java"]

history = "history_data.jsonl"


def to_be_filter(data, wordlist):
    for word in wordlist:
        if word in data:
            return True
    return False


def filter_some_programming_mission(data):
    if to_be_filter(data["instruction"], input_filter_list):
        if random.randint(0, 5) != 0:
            data["instruction"] = ""


def filter_short_output(data):
    if len(data["output"]) < 15:
        data["output"] = ""


def filter_wrong_input(data):
    if "Input" in data["output"] or "Output" in data["input"] or "Output" in data["output"] or "Input" in data["input"]:
        data["output"] = ""


def filter_some_table_mission(data):
    if "table" in data["instruction"]:
        if random.randint(0, 1) != 0:
            data["instruction"] = ""
    if "Column" in data["input"]:
        data["instruction"] = ""


def filter_fri(data):
    if data["output"].endswith("fri"):
        data["instruction"] = ""


def filter_short_list(data):
    if "1)" in data["output"] and "3)" not in data["output"]:
        data["output"] = ""


def filter_long_list(data):
    if "9)" in data["output"] and " )" in data["output"]:
        data["output"] = ""


def filter_dep(data):
    if data["output"].endswith("dep"):
        data["instruction"] = ""


def filter_prime_number(data):
    if "prime" in data["instruction"]:
        data["instruction"] = ""


def merge_and_filt(file_list, output_file):
    with open(output_file, 'w') as outfile:
        for fname in file_list:
            with open(fname, 'r') as infile:
                for line in infile:
                    json_object = json.loads(line)
                    filter_some_programming_mission(json_object)
                    filter_short_output(json_object)
                    filter_wrong_input(json_object)
                    filter_some_table_mission(json_object)
                    filter_fri(json_object)
                    filter_dep(json_object)
                    filter_short_list(json_object)
                    filter_long_list(json_object)
                    filter_prime_number(json_object)
                    if json_object["instruction"] != "" and json_object["output"] != "":
                        json_object['history'] = []
                        json_line = json.dumps(json_object)
                        outfile.write(json_line + '\n')
        with open(history, 'r') as infile:
            for line in infile:
                json_object = json.loads(line)
                json_line = json.dumps(json_object)
                outfile.write(json_line + '\n')


prefix = '/home/jiahe/LLMs/lima/qwen_self_instruct/data/model_generations'
postfix = '/finetuning_data/all_generated_instances.jsonl'
files_to_merge = []
for i in range(1, 4):
    files_to_merge.append(prefix + str(i) + postfix)

output_filename = 'qwen_self_instruct.jsonl'
merge_and_filt(files_to_merge, output_filename)
