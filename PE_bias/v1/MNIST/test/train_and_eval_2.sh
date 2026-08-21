#!/bin/bash

for r in {1..25}
do
	python3 ./train_and_eval.py --run $r
done
