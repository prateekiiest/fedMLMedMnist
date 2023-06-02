#! /bin/bash

for INDEX in '1' '2' '3' '4' '5' '6'
do
    file="results/Baseline2/oct_baseline2_run$INDEX-epochs=8.txt"
    python3 src/baseline_main.py --dataset=octmnist --epochs=8 --model=cnn > $file
done

