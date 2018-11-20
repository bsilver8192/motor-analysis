#!/usr/bin/python3

import matplotlib
import numpy
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]
data = numpy.loadtxt(filename, delimiter=',', skiprows=2, usecols=(1,),
                     comments='\r')
dt = 10e-5

# How many FFT coefficients we'll use to approximate it.
number_coefficients = 2

def approximate(fft, num):
  fft = numpy.copy(fft)
  biggest_indices = numpy.argpartition(abs(fft), -num)[-num:]
  fft[list(i for i in range(len(fft)) if i not in biggest_indices)] = 0
  return numpy.fft.irfft(fft), max(biggest_indices)

zero_noise = 0.15
def find_zero_crossing(start):
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
  return (end + start) // 2

plt.subplot(2, 1, 1)
plt.plot(numpy.linspace(start=0, stop=dt * len(data), num=len(data)), data)
plt.title('%s raw data' % (filename,))
plt.xlabel('time (s)')
plt.ylabel('volts (line-to-line)')

first_zero = find_zero_crossing(0)
second_zero = find_zero_crossing(first_zero + 100)

rpm = 1 / ((second_zero - first_zero) * dt) * 60

# Chop off the last one to make sure the length is even, so FFTs use all the
# points.
if (second_zero - first_zero) % 2:
  second_zero -= 1

one_cycle = data[first_zero:second_zero]
cycle_x = numpy.linspace(start=0, stop=dt * len(one_cycle), num=len(one_cycle))

fft = numpy.fft.rfft(one_cycle)
approximated, max_index = approximate(fft, number_coefficients)

# Now grab more and more coefficients until the next-biggest is after 100. We
# stop at this arbitrary point to avoid matching the noise, but still get a
# close approximation.
# I know there's a way less wasteful way to do this without recalculating things
# all time, but whatever...
number_precise_coefficients = number_coefficients
while max_index < 100:
  number_precise_coefficients += 1
  precise_approximated, max_index = approximate(fft, number_precise_coefficients)
number_precise_coefficients -= 1
precise_approximated, _ = approximate(fft, number_precise_coefficients)
print('Mostly found noise after %d' % number_precise_coefficients)

sin_arg = 2 * numpy.pi * cycle_x / max(cycle_x)
f = numpy.real(fft[1]) * numpy.sin(sin_arg) + numpy.real(fft[7]) * numpy.sin(sin_arg * 7) / 7
f /= -(max(f) - min(f)) / (max(precise_approximated) - min(precise_approximated))

plt.subplot(2, 1, 2)
plt.plot(cycle_x, one_cycle, label='raw')
plt.plot(cycle_x, approximated, label='course')
plt.plot(cycle_x, precise_approximated, label='precise')
plt.plot(cycle_x, f, label='f')
plt.legend()
plt.title('%s one cycle' % (filename,))
plt.xlabel('time (s)')
plt.ylabel('volts (line-to-line)')

peak_peak_voltage = max(precise_approximated) - min(precise_approximated)
# Calculate the overall peak-peak KV, which is what your battery voltage limits.
kv = rpm / peak_peak_voltage / 1.5
print('KV = %f RPM/V' % kv)
print('Speed at 48V = %f RPM' % (kv * 48))

assert number_coefficients == 2
print('Flux linkage = %f * sin(theta) + %f / 7 * sin(7 * theta)' %
      (abs(fft[1]), abs(fft[7])))

plt.show()
