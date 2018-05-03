#!/bin/bash

for file in ./perturbed_subset/*
do
	python check_old.py $file >> old_one.txt
done
