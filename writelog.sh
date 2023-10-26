#!/bin/bash

# If we have existing log, see if we can fill it in with training data
if [ -e "tlogs/latest/log.txt" ] && [ ! -e "tlogs/latest/.posttrainlog" ] && [ -e qlora-out ]; then
    ./posttrainlog.sh
fi

# Write information about the next training session

logname="$(date +%Y%m%d%H%M%S)"
logdir="tlogs/$logname"
logfile="$logdir/log.txt"

mkdir -p $logdir
echo "$logname" > $logdir/.logname
echo "Training session started at $(date)" > $logfile

rm tlogs/latest
ln -s $logname tlogs/latest

# current model
echo -n -e "==========================================\nCurrent model:" >> $logfile
ls -ld curr-model >> $logfile

# current data
echo -n -e "==========================================\nCurrent data:" >> $logfile
cp prepared-dataset.jsonl $logdir
ls -l prepared-dataset.jsonl >> $logfile

# current training script
echo -n -e "==========================================\nCurrent training script:" >> $logfile
cp axolotl-"$1"*.yml $logdir
ls -l axolotl-"$1"*.yml >> $logfile

# python module versions
echo -n -e "==========================================\nPython module versions:" >> $logfile
pip freeze >> $logfile

# axolotl git commit and changes
echo -n -e "==========================================\nAxolotl git commit and changes:" >> $logfile
cd ../axolotl
git log -1 >> ../qlora/$logfile
git diff >> ../qlora/$logfile
cd ../qlora

# current git commit and changes
echo -n -e "==========================================\nCurrent git commit and changes:" >> $logfile
git log -1 >> $logfile
git diff >> $logfile

