#!/bin/bash

# Testing script, should activate conda env "fermi".
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


# timer=$(($(($1 % $int ))*3))
timer=$(($1 % $int ))
echo "sleeping for $timer"
sleep $timer
# sleep 60
conda activate fermi3
echo "which conda:"
conda info --base
echo "conda path"
echo "$CONDA_PREFIX"
echo "Copying needed files..."
# cp -r /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/fits_01 .
cp -r /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/roi_$2 .
echo "STARTING THE PYTHON SCRIPT with arguments:"
echo $1 $2 $3
python /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/grid_ts/ts_pixel_refactored.py $1 $2 $3

