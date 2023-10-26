#!/bin/bash

# Train 256, 512, 1024, and 2048 context length models
for i in 256 512 1024; do
    echo "Training $1 model with context length $i"
    python -m axolotl.cli.train ./axolotl-$1-$i.yml
    # Determine checkpoint
    last_checkpoint_dir=$(ls -d qlora-out/checkpoint-* -v | tail -n 1)
done
