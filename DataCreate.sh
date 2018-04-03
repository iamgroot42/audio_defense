#!/bin/bash

for file in ./perturbed_subset/*
do
	python3 createTrainingData.py $file >> perturbed.txt
done
