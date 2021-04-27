#!/bin/bash

let count=0

while IFS=' ' read -r col1 col2 col3 col4; do
    (( count++ ))
    python ./images_whole_combine_on_folder.py -t $col1 -p $col2 -o $col3 -r $col4 -j 6 &
    if (( count % 3 == 0)); then
        echo "$count"
        wait
    fi
done < "$1"