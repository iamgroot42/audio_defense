#!/bin/bash

for file in ./commonvoice_subset2/*
do
	python check_old.py $file >> old_zero.txt
	exit
done
