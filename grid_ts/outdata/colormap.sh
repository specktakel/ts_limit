#!/bin/bash

# script to get python to do some colormaps
SUBDIR=ts_95
for i in {49..70}
do
  files=($(ls roi_"$i"/"$SUBDIR"))
  # echo $files
  num_of_files=${#files[@]}
  echo "$i, $num_of_files"
  if [ ! -f colormaps/roi_"$i"_color_map.png ]
  then 
    if [ $num_of_files -eq 900 ] #  && [ ! -f colormaps/roi_"$i"_color_map.png ]
    then
      echo "dir with index $i has all data"
      echo "invoking script with arg $i"
      python loglike_mac.py "$i"
    else
      echo "data not all done"
    fi
  else
    echo "png is already done"
  fi


done
