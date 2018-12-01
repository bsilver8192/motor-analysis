#!/usr/bin/python3

import unittest
import numpy

import models

class TestCase(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._epsilon = numpy.pi / 500
    self._theta = numpy.arange(0, numpy.pi * 2, self.epsilon) + self.epsilon / 2

  @property
  def epsilon(self):
    return self._epsilon

  @property
  def theta(self):
    return self._theta

  def assertFClose(self, a, b, offset=0):
    if not numpy.allclose(a(self.theta), b(self.theta + offset)):
      raise AssertionError('%r != %r' % (a, b))

class TestCosSum(TestCase):
  '''A sanity test of the CosSum functionality. This mostly just makes sure that
  converting between line_line and phase values gives the same result both
  directions.'''

  def assertEqualRotated(self, a, b):
    '''Asserts that two functions (a and b) which are periodic with a period
    which divides 2*pi are equal after rotation by some reasonable number.'''

    # First verify they're both periodic.
    self.assertFClose(a, a, numpy.pi * 2)
    self.assertFClose(b, b, numpy.pi * 2)
    for angle in numpy.arange(0, numpy.pi * 2, numpy.pi / 6):
      if ((abs(a(self.theta) - b(self.theta + angle)) < self.epsilon).all()):
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

class TestFunctions(TestCase):
  '''A sanity test of various commutation patterns we define. This verifies
  they are continuous and all three phases add up to a constant.

  This is helpful for checking that they're not mirrored or anything silly.'''

  def assertFunction(self, f):
    '''Asserts that f meets the minimum requirements for being a voltage
    waveform, current waveform, or flux linkage derivative.

    Note that most of this doesn't matter in theory, but we want it to be true
    for all the functions we use to remove unnecessary DOFs in the definitions
    and make subsequent math less confusing.'''
    self.assertGreater(max(f(self.theta)) - min(f(self.theta)), 0.5)
    with self.subTest('isPeriodic', f=f):
      self.assertFClose(f, f, numpy.pi * 2)
      self.assertFClose(f, f, numpy.pi * 4)
    with self.subTest('isOddSymmetric', f=f):
      # Note that this deliberately verifies it has odd symmetry about pi, and
      # not some other point.
      self.assertFClose(f, lambda t: -f(numpy.pi * 2 - t))
    # We don't actually need the function to have zeros. We just need it to
    # average zero about the interesting points, to make sure it's properly
    # aligned.
    for point in (0, numpy.pi, numpy.pi * 2):
      with self.subTest('zeros', f=f, point=point):
        before = f(point - self.epsilon)
        after = f(point + self.epsilon)
        self.assertAlmostEqual((before + after) / 2, 0)

  def assertContinuous(self, f):
    '''Asserts that f is continuous. This is true for some waveforms, but not
    others.'''
    max_slope = 5
    for offset in (self.epsilon, self.epsilon * 0.9, self.epsilon * 1.1,
                   self.epsilon * 1.9, self.epsilon * 2, self.epsilon * 2.1):
      self.assertTrue((abs(f(self.theta) - f(self.theta + offset)) < self.epsilon * max_slope).all())

  def assertConstant(self, f, motor):
    '''Asserts that f*motor for all three phases is constant throughout a
    whole revolution.'''
    def all_three_raw(theta):
      a_theta = theta
      b_theta = theta + numpy.pi * 2 / 3
      c_theta = theta - numpy.pi * 2 / 3
      return (f(a_theta)*motor(a_theta) +
              f(b_theta)*motor(b_theta) +
              f(c_theta)*motor(c_theta))
    zero = all_three_raw(0)
    def all_three(theta):
      return all_three_raw(theta) - zero
    self.assertFClose(all_three, lambda _: 0)

  def testTrapezoid(self):
    self.assertFunction(models.trapezoid)
    self.assertContinuous(models.trapezoid)

  def testTrapezoid6Step(self):
    self.assertFunction(models.trapezoid_6step)

  def testTrapezoid4Step(self):
    self.assertFunction(models.trapezoid_4step)
    self.assertConstant(models.trapezoid_4step, models.trapezoid)
    self.assertConstant(models.trapezoid_4step, models.trapezoid_4step)

  def testSquare(self):
    self.assertFunction(models.square)

if __name__ == '__main__':
  unittest.main()
