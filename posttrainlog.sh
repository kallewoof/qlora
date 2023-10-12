#!/bin/bash

# Write final information such as loss, final adapter, etc., to the log file

logdir=tlogs/latest/
logfile=$logdir/log.txt

if [ ! -e $logfile ]; then
    echo "Log file $logfile does not exist"
    exit 1
fi

echo -e "\n\n==========================================\nPost-training information:" >> $logfile

# Find the last checkpoint
last_checkpoint_dir=$(ls -d qlora-out/checkpoint-* -v | tail -n 1)
if [ -z "$last_checkpoint_dir" ]; then
    echo "No checkpoints found"
    exit 1
fi

cp $last_checkpoint_dir/trainer_state.json $logdir

# Copy the lora adapter
cp qlora-out/adapter* $logdir

touch $logdir/.posttrainlog
