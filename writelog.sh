#!/bin/bash

# Write information about the next training session

logdir="tlogs/$(date +%Y%m%d%H%M%S)"
logfile="$logdir/log.txt"

mkdir -p $logdir
echo "Training session started at $(date)" > $logfile

rm tlogs/latest
ln -s $logdir tlogs/latest

# current model
echo -r -e "==========================================\nCurrent model:" >> $logfile
ls -ld curr-model >> $logfile

# current data
echo -r -e "==========================================\nCurrent data:" >> $logfile
cp prepared-dataset.jsonl $logdir
ls -l prepared-dataset.jsonl >> $logfile

# current training script
echo -r -e "==========================================\nCurrent training script:" >> $logfile
cp axolotl-$1.yml $logdir
ls -l axolotl-$1.yml >> $logfile

# python module versions
echo -r -e "==========================================\nPython module versions:" >> $logfile
pip freeze >> $logfile

# axolotl git commit and changes
echo -r -e "==========================================\nAxolotl git commit and changes:" >> $logfile
cd ../axolotl
git log -1 >> ../qlora/$logfile
git diff >> ../qlora/$logfile
cd ../qlora

# current git commit and changes
echo -r -e "==========================================\nCurrent git commit and changes:" >> $logfile
git log -1 >> $logfile
git diff >> $logfile

