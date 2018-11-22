#!/usr/bin/python3

import numpy
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]
file_data = numpy.loadtxt(filename, delimiter=',').T
data = file_data[1]
timesteps = file_data[0]

# How many FFT coefficients we'll use to approximate it.
number_coefficients = 2

def approximate(fft, num):
  '''Returns the inverse transform of fft, after dropping all but the biggest
  num coefficients.'''
  fft = numpy.copy(fft)
  biggest_indices = numpy.argpartition(abs(fft), -num)[-num:]
  fft[list(i for i in range(len(fft)) if i not in biggest_indices)] = 0
  return numpy.fft.irfft(fft), max(biggest_indices)

zero_noise = 0.15
def find_zero_crossing(start):
  '''Returns the index into data of the first zero crossing after start.'''
  i = start
  while data[i] > 0:
    i += 1
  i += 20
  while data[i] < -zero_noise:
    i += 1
  start = i
  while data[i] < zero_noise:
    i += 1
  end = i
  assert end - start < 20
  return (end + start) // 2

round_angle_multiple = numpy.pi / 6
def round_angle(angle):
  '''Returns angle rounded to a multiple of pi/6.'''
  return round(angle / round_angle_multiple) * round_angle_multiple

first_zero = find_zero_crossing(0)
second_zero = find_zero_crossing(first_zero + 100)
abs_first_zero, abs_second_zero = first_zero, second_zero

# Chop off the last one to make sure the length is even, so FFTs use all the
# points.
if (second_zero - first_zero) % 2:
  second_zero -= 1

one_cycle = data[first_zero:second_zero]
cycle_timesteps = timesteps[first_zero:second_zero]
omega = (numpy.pi * 2) / (cycle_timesteps[-1] - cycle_timesteps[0])

fft = numpy.fft.rfft(one_cycle)
approximated, max_index = approximate(fft, number_coefficients)

# Now grab more and more coefficients until the next-biggest is after 100. We
# stop at this arbitrary point to avoid matching the noise, but still get a
# close approximation.
# I know there's a way less wasteful way to do this without recalculating things
# all time, but whatever...
number_precise_coefficients = number_coefficients
while abs(fft[max_index]) >= abs(fft[1]) / 30:
  number_precise_coefficients += 1
  precise_approximated, max_index = approximate(fft, number_precise_coefficients)
number_precise_coefficients -= 1
precise_approximated, _ = approximate(fft, number_precise_coefficients)
print('Mostly found noise after %d' % (number_precise_coefficients,))

# Grab just our one cycle of data.
fft_offset = -int(round(numpy.angle(fft[1]) / (2 * numpy.pi) *
                        len(cycle_timesteps)))
first_zero += fft_offset
second_zero += fft_offset
one_cycle = data[first_zero:second_zero]
cycle_timesteps = timesteps[first_zero:second_zero]

# Shuffle the functions around so they start in the same place in a cycle.
approximated = numpy.roll(approximated, -fft_offset)
precise_approximated = numpy.roll(precise_approximated, -fft_offset)

cycle_x = cycle_timesteps - cycle_timesteps[0]
assert min(cycle_x) == 0
cycle_time = max(cycle_x)

coefficients = list(sorted(numpy.argpartition(abs(fft), -number_coefficients)[-number_coefficients:]))
assert coefficients[0] == 1
f_scale = 1 / (len(cycle_x) / 2)
linkage_message = 'Flux linkage = %f * cos(theta)' % (abs(fft[1]) * f_scale / omega)
zero_angle = numpy.angle(fft[1])
cos_arg = 2 * numpy.pi * cycle_x / cycle_time - zero_angle
f = abs(fft[1]) * numpy.cos(cos_arg + zero_angle)
f_rounded = numpy.copy(f)
for coefficient in coefficients[1:]:
  scalar = abs(fft[coefficient])
  angle = numpy.angle(fft[coefficient])
  rounded_angle = round_angle(angle)
  linkage_message += ' + %f * cos(%d * theta + %f)' % (
      scalar * f_scale / omega, coefficient, rounded_angle)
  f += scalar * numpy.cos(cos_arg * coefficient + angle)
  f_rounded += scalar * numpy.cos(cos_arg * coefficient + rounded_angle)
f *= f_scale
f_rounded *= f_scale
linkage_message += ' V/(rad/s) aka N*m/A'
print(linkage_message)

plt.subplot(2, 1, 2)
plt.plot(cycle_x, one_cycle, label='raw')
# This should precisely overlap f, but put it on here anyways to allow visually
# double checking.
plt.plot(cycle_x, approximated, label='course')
if (approximated != precise_approximated).any() or True:
  # Avoid overlapping lines because they're confusing.
  plt.plot(cycle_x, precise_approximated, label='precise')
plt.plot(cycle_x, f, label='f')
plt.plot(cycle_x, f_rounded, label='f_rounded')
plt.legend()
plt.title('%s one cycle' % (filename,))
plt.xlabel('time (s)')
plt.ylabel('volts (line-to-line)')

plt.subplot(2, 1, 1)
plt.plot(timesteps, data, label='raw')
plt.title('%s all data' % (filename,))
plt.axvline(x=timesteps[first_zero], linestyle=':', color='r')
plt.axvline(x=timesteps[second_zero], linestyle='-', color='r')
plt.axvline(x=timesteps[abs_first_zero], linestyle=':', color='g')
plt.axvline(x=timesteps[abs_second_zero], linestyle='-', color='g')
plt.axhline(y=0)
plt.xlabel('time (s)')
plt.ylabel('volts (line-to-line)')

plt.show()
