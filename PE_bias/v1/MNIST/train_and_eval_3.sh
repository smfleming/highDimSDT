#!/bin/bash

dimensionalities=(6 7 8 9 10)
for n_cl in "${dimensionalities[@]}"
do
  for r in {1..25}
  do
    python3 ./train_and_eval_3.py --run $r --n_classes $n_cl
  done
done
