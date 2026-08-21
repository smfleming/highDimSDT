#!/bin/bash

# Run train_and_eval_5.py with different n_classes values
# This script tests the fixed architecture with proper output dimensions

dimensionalities=(10 2 3 4 5 6 7 8 9)
for n_cl in "${dimensionalities[@]}"
do
  for new_n_cl in {2..10}
  do
    for r in {1..25}
    do
      python3 ./train_and_eval_5.py --run $r --n_classes $n_cl --new_n_classes $new_n_cl --train-batch-size 256
    done
  done
done

