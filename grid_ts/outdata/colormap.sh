#!/bin/bash

# script to get python to do some colormaps
SUBDIR=ts_95
declare -a incomplete
for i in {95..99}
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
    elif [ $num_of_files -gt 0 ]
    then
      echo "data not all done"
      incomplete+=($i)
    else
       echo "ROI not touched yet"
    fi
  else
    echo "png is already done"
  fi

printf "%s\n" "${incomplete[@]}" > ../to_be_submitted.txt
done
