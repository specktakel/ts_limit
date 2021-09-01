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

### CLA:
### $1: gm
### $2: start of roi

### num is fixed to some integer

conda activate fermi3
echo "which conda:"
conda info --base
echo "conda path"
echo "$CONDA_PREFIX"
# simulates 5 data sets: saves them at correct directory, removes working directory and repeats


number=5
begin=$(($2 * $number))
end=$(($(($2 + 1)) * $number))

for ((i=$begin; i<$end; i+=1))
do
  echo "Copying needed files..."
  cp -r /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/fits_01 .
  echo "STARTING THE PYTHON SCRIPT with argument:"
  echo $1 $i
  python /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/structured_field/h1/simulate_roi.py $1 $i
  cp -r ./fits_01 /nfs/astrop/n1/kuhlmann/NGC_1275/ts_limit/roi_simulation/roi_files/struc_h1/gm_$1/roi_$i
  rm -r ./fits_01
done
