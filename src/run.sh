#! /bin/bash

for INDEX in '1' '2' '3' '4' '5' '6'
do
    file="results/Baseline3/octmnist/oct_dispsum_baseline3_run$INDEX-epochs=10_8.txt"
    python3 src/fed_main.py --dataset=octmnist --epochs=10 --local_ep=8 --model=cnn --subset=True --random=False --algo=dispsum > $file
done

for INDEX in '1' '2' '3' '4' '5' '6'
do
    file="results/Baseline3/octmnist/oct_dispmin_baseline3_run$INDEX-epochs=10_8.txt"
    python3 src/fed_main.py --dataset=octmnist --epochs=10 --local_ep=8 --model=cnn --subset=True --random=False --algo=dispmin > $file
done