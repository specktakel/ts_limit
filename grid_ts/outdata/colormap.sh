#!/bin/bash

# script to get python to do some colormaps
SUBDIR=ts_95
for i in {20..50}
do
  files=($(ls roi_"$i"/"$SUBDIR"))
  # echo $files
  num_of_files=${#files[@]}
  echo "$i, $num_of_files"
  if [ $num_of_files -eq 900 ] #  && [ ! -f colormaps/roi_"$i"_color_map.png ]
  then
    echo "dir with index $i has all data"
    if [ ! -f colormaps/roi_"$i"_color_map.png ]
      then
        echo "invoking script with arg $i"
        python loglike_mac.py "$i"
    else
      echo "data is there, png already done"
    fi
  else
    echo "not all data is there"
  fi


done
