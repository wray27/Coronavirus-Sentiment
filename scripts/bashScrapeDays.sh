#!/bin/bash

for i in {02..02}
do
  for j in {06..11}
  do
    python mineData.py "2020-$i-$j"
  done
done
