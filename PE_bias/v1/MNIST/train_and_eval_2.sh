#!/bin/bash

for r in {1..10}
do
	python3 ./train_and_eval_2.py --run $r
done
