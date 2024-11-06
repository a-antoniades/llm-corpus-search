#!/bin/bash

PID=78193
# COMMAND="
# python wimbd_preprocess.py
#         --task "mmlu"
#         --base_dir "./results/n-grams/mmlu/exp3/test-set/exp_full_None"
#         --method common
#         --sub_tasks marketing management
#                     high_school_world_history
#                     high_school_european_history
#                     miscellaneous
#         --n_grams 3 4 5

# "
COMMAND="
python wimbd_preprocess.py
                    --task "mmlu"
                    --base_dir "./results/n-grams/mmlu/pile/exp4_nofilter/test-set/exp_full_None" 
                    --method common 
                    --sub_tasks marketing management 
                                high_school_world_history 
                                high_school_european_history 
                                miscellaneous 
                    --n_grams 5 6 7

"

# Loop to check if the process with the given PID exists
while true; do
    if ! ps -p $PID > /dev/null; then
        echo "Process $PID has terminated. Launching program..."
        $COMMAND
        break
    fi
    sleep 10  # Wait for 10 seconds before checking again
    echo "Waiting..."
done