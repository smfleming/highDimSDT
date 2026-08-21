#!/bin/bash

# Run train_and_eval_6.py with different n_classes values
# This script tests the optimized architecture with significant performance improvements

dimensionalities=(10 2 3 4 5 6 7 8 9)
for n_cl in "${dimensionalities[@]}"
do
  for r in {1..25}
  do
    python3 ./train_and_eval_6.py --run $r --n_classes $n_cl --new_n_classes 2
  done
done

