#!/bin/bash

for file in ./perturbed_subset/*
do
	python mansi_features.py $file >> word_one.txt
done
