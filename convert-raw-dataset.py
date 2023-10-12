# Convert a bunch of text files into [{"text": "story1"}, {"text":"story2"}, ...]
# The text files are stored as .txt files in the ./curr-dataset/ directory.

import os
import json
import re
import random

# Get the directory of the current file
curr_dir = os.path.dirname(os.path.realpath(__file__))
# Get the directory of the dataset
dataset_dir = os.path.join(curr_dir, "curr-dataset")
# Get the directory of the output
output_jsonl = os.path.join(curr_dir, "prepared-dataset.jsonl")

results = []

# Iterate over each file in the dataset directory
for filename in os.listdir(dataset_dir):
    # Get the path of the file
    filepath = os.path.join(dataset_dir, filename)
    # Open the file
    with open(filepath, "r") as file:
        # Read the file
        text = file.read()
        # Clean the file up a little
        # Consider triple newline as new story
        stories = text.split("\n\n\n")
        for story in stories:
            # Remove leading and trailing whitespace
            story = story.strip()
            # Skip empty stories
            if story == "":
                continue
            # Replace existing triple newlines with double newlines
            story = story.replace("\n\n\n", "\n\n")
            # Replace 4+ dots with 3 dots
            story = re.sub(r"\.{4,}", "...", story)
            # Replace 4+ dashes with 3 dashes
            story = re.sub(r"-{4,}", "---", story)
            results.append(story)

# Shuffle results
random.shuffle(results)

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
