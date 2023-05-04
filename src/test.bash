#!/bin/bash

for i in {1..10}
do
  echo "Running command: $i"
  ./gt -i -r 5 6 1235 1
  # ./gt -e -t -s 10000000 -r 8 9 1235 1
#   ./gt -e -r 8 9 1235 1
#   ./gt symm2.gam 8 9 1235 1
done

echo "Done!"