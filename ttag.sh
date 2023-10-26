#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tag>"
    echo "Tags the latest training session with the given tag"
    exit 1
fi

cd tlogs
logname=$(cat latest/.logname)
ln -s $logname $logname-"$@"
cd ..
ls -l tlogs/$logname-"$@"
