#!/usr/bin/python3

# Input format: Two header lines.
# Data lines: relative_time,voltage_a,voltage_b,voltage_c,noise
# Subtracts voltage_a and voltage_b to get a line-to-line voltage.

import csv
import sys

with open(sys.argv[1], 'r') as in_file:
  with open(sys.argv[2], 'w') as out_file:
    reader = csv.reader(in_file)
    writer = csv.writer(out_file, lineterminator='\n')
    start_time = None
    # Skip the header rows.
    next(reader)
    next(reader)
    for row in reader:
      if start_time is None:
        start_time = float(row[0])
      relative_time = float(row[0]) - start_time
      voltage_a, voltage_b = float(row[1]), float(row[2])
      writer.writerow(('{0:.6f}'.format(relative_time), voltage_a - voltage_b))
