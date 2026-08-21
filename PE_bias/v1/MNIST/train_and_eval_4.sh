#!/bin/bash

dimensionalities=(10 2 3 4 5 6 7 8 9)
for n_cl in "${dimensionalities[@]}"
do
  for r in {1..25}
  do
    python3 ./train_and_eval_4.py --run $r --n_classes $n_cl --new_n_classes 4
  done
done

