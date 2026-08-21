#!/bin/bash

dimensionalities=(2 3 4 5 6 7 8 9 10)
for n_cl in "${dimensionalities[@]}"
do
  python3 ./PE_test_3.py --N_runs 25 --n_classes $n_cl
done

