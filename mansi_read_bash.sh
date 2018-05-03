#!/bin/bash

for file in ./commonvoice_subset2/*
do
	python mansi_features.py $file >> word_zero.txt
done
