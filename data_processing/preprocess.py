#!/usr/bin/python3

# Input format: Two header lines. 10e-5 timestep.
# Data lines: sequence,voltage,
# The one voltage is between two phases.

import csv
import sys

DT = 10e-5

with open(sys.argv[1], 'r') as in_file:
  with open(sys.argv[2], 'w') as out_file:
    reader = csv.reader(in_file)
    writer = csv.writer(out_file, lineterminator='\n')
    time = 0
    # Skip the header rows.
    next(reader)
    next(reader)
    for row in reader:
      writer.writerow(('{0:.5f}'.format(time), float(row[1])))
      time += DT
