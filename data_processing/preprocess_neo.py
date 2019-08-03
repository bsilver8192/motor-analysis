#!/usr/bin/python3

# Input format: Two header lines.
# Data lines: time,voltage,
# The one voltage is between two phases.

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
      writer.writerow(('{0:.6f}'.format(relative_time), float(row[1])))
