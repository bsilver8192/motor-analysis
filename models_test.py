#!/usr/bin/python3

import unittest
import numpy

import models

class TestCosSum(unittest.TestCase):
  '''A sanity test of the CosSum functionality. This mostly just makes sure that
  converting between line_line and phase values gives the same result both
  directions.'''

  def assertEqualRotated(self, a, b):
    '''Asserts that two functions (a and b) which are periodic with a period
    which divides 2*pi are equal after rotation by some reasonable number.'''

    theta = numpy.arange(0, numpy.pi * 2, 0.01)
    epsilon = 0.000001
    # First verify they're both periodic.
    self.assertTrue((abs(a(theta) - a(theta + numpy.pi * 2)) < epsilon).all())
    self.assertTrue((abs(b(theta) - b(theta + numpy.pi * 2)) < epsilon).all())
    for angle in numpy.arange(0, numpy.pi * 2, numpy.pi / 6):
      if ((abs(a(theta) - b(theta + angle)) < epsilon).all()):
        return
    self.fail('Did not find a way these functions are equal: %s vs %s' % (a, b))

  def testAssertEqualRotated(self):
    self.assertEqualRotated(numpy.sin,
                            lambda theta: numpy.sin(theta + numpy.pi / 3))
    self.assertEqualRotated(numpy.sin, numpy.cos)
    with self.assertRaisesRegex(AssertionError,
                                'Did not find a way these functions are equal'):
      self.assertEqualRotated(numpy.sin, lambda theta: numpy.sin(theta) * 0.9)

  def check_phase_line_line_conversions(self, f):
    with self.subTest(line_line=f):
      from_line_line = models.CosSum(line_line = f)
      from_phase = models.CosSum(phase = from_line_line.phase_coeff)
      self.assertEqualRotated(from_phase.line_line, from_line_line.line_line)
      self.assertEqualRotated(from_line_line.phase, from_phase.phase)
    with self.subTest(phase=f):
      from_phase = models.CosSum(phase = f)
      from_line_line = models.CosSum(line_line = from_phase.line_line_coeff)
      self.assertEqualRotated(from_phase.line_line, from_line_line.line_line)
      self.assertEqualRotated(from_line_line.phase, from_phase.phase)

  def test_one_cos(self):
    self.check_phase_line_line_conversions({1: (1, 0)})
    self.check_phase_line_line_conversions({1: (13.3, 0)})
    self.check_phase_line_line_conversions({1: (13.3, 0.1)})
    self.check_phase_line_line_conversions({1: (1, 15)})

  def test_two_cos(self):
    self.check_phase_line_line_conversions({1: (1, 0), 5: (0.5, 0.23)})
    self.check_phase_line_line_conversions({1: (0.1, 0), 5: (0.5, 0.23)})
    self.check_phase_line_line_conversions({5: (0.1, 0), 7: (0.5, 0.23)})

  def test_many_cos(self):
    self.check_phase_line_line_conversions({1: (1, 0), 5: (0.5, 0.23),
                                            7: (0.1, 0.3)})

if __name__ == '__main__':
  unittest.main()
