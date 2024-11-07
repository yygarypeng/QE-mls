#!/bin/bash
for f in DNN*; do
    new_name="${f/DNN/DNN_}"  # Add underscore after DNN
    new_name="${new_name%_}"  # Remove trailing underscore
    mv -- "$f" "$new_name"
done
