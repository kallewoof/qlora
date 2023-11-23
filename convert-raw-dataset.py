# Convert a bunch of text files into [{"text": "story1"}, {"text":"story2"}, ...]
# The text files are stored as .txt files in the ./curr-dataset/ directory.

import os
import json
import re
import random
from transformers import AutoTokenizer

def quickfmt(e):
    e = json.loads(e)
    if "input" in e:
        return """### Instruction: {instruction}

### Input: {input}

### Response: {output}""".format(instruction=e["instruction"], input=e["input"], output=e["output"])
    else:
        return """### Instruction: {instruction}

### Response: {output}""".format(instruction=e["instruction"], output=e["output"])

eos = "</s>" # default

tokenizer = AutoTokenizer.from_pretrained("./curr-model", trust_remote_code=True)
# Hard coded EOS? args.tokenizer + "/fixed-eos.txt" exists?
if os.path.exists("./curr-model/fixed-eos.txt"):
    with open("./curr-model/fixed-eos.txt", "r") as file:
        eos = file.read()
else:
    # Try to derive from tokenizer_config.json
    # {
    #     "add_bos_token": true,
    #     "add_eos_token": false,
    #     "eos_token": {
    #         "__type": "AddedToken",
    #         "content": "</s>",
    # ...
    #     },
    # ...
    tokcfg = json.read("./curr-model/tokenizer_config.json")
    if tokcfg["add_eos_token"]:
        eos = tokcfg["eos_token"]["content"]

print(f"EOS token: {eos}")

# Get the directory of the current file
curr_dir = os.path.dirname(os.path.realpath(__file__))
# Get the directory of the dataset
dataset_dir = os.path.join(curr_dir, "curr-dataset")
# Get the directory of the output
output_jsonl = os.path.join(curr_dir, "prepared-dataset.jsonl")
instr_output_jsonl = os.path.join(curr_dir, "prepared-instr-dataset.jsonl")
instr_output = open(instr_output_jsonl, "w")

results = []

# Iterate over each file in the dataset directory
entries = 0
for filename in os.listdir(dataset_dir):
    # Get the path of the file
    filepath = os.path.join(dataset_dir, filename)
    # Open the file
    with open(filepath, "r") as file:
        # Read the file
        text = file.read()
        # The file may already by jsonl, in which case it is an instruction finetune component, and we need to merge it with the instruction set
        if filepath.endswith(".jsonl"):
            # Write the contents as is into the instruction set
            entries += len(text.split("\n"))
            instr_output.write(text)
            continue
        # Clean the file up a little
        # Consider triple newline as new story
        stories = text.split("\n\n\n")
        for story in stories:
            # Remove leading and trailing whitespace
            story = story.strip()
            # Skip empty stories
            if story == "":
                continue
            assert len(tokenizer.tokenize(story)) > 455, "Story too short (" + str(len(tokenizer.tokenize(story))) + "): " + story
            # Replace existing triple newlines with double newlines
            story = story.replace("\n\n\n", "\n\n")
            # Replace 4+ dots with 3 dots
            story = re.sub(r"\.{4,}", "...", story)
            # Replace 4+ dashes with 3 dashes
            story = re.sub(r"-{4,}", "---", story)
            results.append(story)
            entries += 1

# Include instruction sets
# - we include the entire camelv set as is
extentries = 0
tinyentries = 0
with open('public-datasets/camelv/camelv.jsonl', 'r') as file:
    text = file.read()
    # Reject tiny entries
    for entry in text.split("\n"):
        if len(entry.strip()) == 0: continue
        if len(tokenizer.tokenize(quickfmt(entry))) < 456:
            tinyentries += 1
            continue
        instr_output.write(entry + "\n")
        extentries += 1

# Include a shuffled count of 1x the entries from the oasst1 dataset
with open('public-datasets/oasst1/oasst1-train.jsonl', 'r') as file:
    text = file.read().split("\n")
    random.shuffle(text)
    entries2x = entries
    # Read and add entries until we have the desired amount
    included = 0
    for entry in text:
        if len(entry.strip()) == 0: continue
        if len(tokenizer.tokenize(quickfmt(entry))) < 456:
            tinyentries += 1
            continue
        instr_output.write(entry + "\n")
        included += 1
        extentries += 1
        if included >= entries2x:
            break

print(f"Private entries: {entries}, external entries (oasst1, camelv): {extentries}, tiny entries (skipped): {tinyentries}")

# # Shuffle results
# random.shuffle(results)

# Open the output file
with open(output_jsonl, "w") as output_file:
    # Iterate over each text
    for text in results:
        # Write file to output with escaped newlines as {"text": "CONTENT"}
        output_file.write(json.dumps({"text": text}))
        # Write newline
        output_file.write("\n")

# Print the output file path
print(output_jsonl)
