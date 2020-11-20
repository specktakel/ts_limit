#!/bin/bash

# Testing script, should activate conda env "fermi3".
# Then call sim.py in some directory with arg $1.
# Copy stuff from Franziska's bash script.

__conda_setup="$('/nfs/astrop/n1/kuhlmann/miniconda/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
        eval "$__conda_setup"
        echo "Conda is set up!"
else
        if [ -f "/nfs/astrop/n1/kuhlmann/miniconda/etc/profile.d/conda.sh" ]; then
                . "/nfs/astrop/n1/kuhlmann/miniconda/etc/profile.d/conda.sh"
                echo "Conda also set up!"
        else
                export PATH="/nfs/astrop/n1/kuhlmann/miniconda/bin:$PATH"
                echo "No idea what this actually does!"
        fi
fi
unset __conda_setup


int=60


timer=$(($(($1 % $int ))*3))
echo "sleeping for $timer"
sleep $timer
conda activate fermi3
echo "which conda:"
conda info --base
echo "conda path"
echo "$CONDA_PREFIX"
echo "STARTING THE PYTHON SCRIPT with argument:"
echo $1
python /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/gen_probs.py $1 1
python /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_survival_prob/gen_probs.py $1 2
