#!/bin/bash

dimensionalities=(10 2 3 4 5 6 7 8 9)
target_accs=(0.55 0.60 0.65 0.70 0.75)
for target_acc in "${target_accs[@]}"
do
  for n_cl in "${dimensionalities[@]}"
  do
    python3 ./PE_test_4.py --N_runs 25 --n_classes $n_cl --new_n_classes 4 --target_acc $target_acc
  done
done


