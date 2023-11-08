#!/bin/bash
f=$(ls -d qlora-out/checkpoint-* -v | tail -n 1)/trainer_state.json
cat $f|jq ".log_history[].eval_loss" | grep -v null
