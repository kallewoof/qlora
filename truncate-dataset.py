# truncate the prepared-dataset.jsonl by the given percentage

import os
import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("proportion", help="proportion of dataset to keep", type=float)

args = parser.parse_args()
with open("prepared-dataset.jsonl", "r") as file:
    with open("prepared-dataset-truncated.jsonl", "w") as output_file:
        lines = file.readlines()
        for line in lines:
            if random.random() < args.proportion:
                output_file.write(line)

# move truncated dataset over original dataset
os.remove("prepared-dataset.jsonl")
os.rename("prepared-dataset-truncated.jsonl", "prepared-dataset.jsonl")
