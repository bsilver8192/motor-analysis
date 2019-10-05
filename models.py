'''This module has models of various aspects of motors and control algorithms.

These models are all simple linear/sinusoidal functions at runtime. There is
some moderately more complex math at initialization time, but everything in this
module is simple at runtime.'''

import numpy
import scipy.optimize
import types

_frozendict = types.MappingProxyType

def _vectorize_float(f):
  return numpy.vectorize(f, otypes=(numpy.float,))

def _offset_coeff(coeff):
  coeff = dict(coeff)
  offset = -numpy.pi / 2 - coeff[1][1]
  for i in coeff:
    coeff[i] = (coeff[i][0], coeff[i][1] + offset)
  return coeff

class CosSum(object):
  def __init__(self,
               line_line = None, phase = None):
    if phase is not None:
      self._phase_coeff = _frozendict(_offset_coeff(phase))
      self._line_line_coeff = _frozendict(
          {a: (self.phase_coeff[a][0] * numpy.sqrt(3),
               self.phase_coeff[a][1])
           for a in self.phase_coeff})

    if line_line is not None:
      self._line_line_coeff = _frozendict(_offset_coeff(line_line))
      self._phase_coeff = _frozendict(
          {a: (self.line_line_coeff[a][0] / numpy.sqrt(3),
               self.line_line_coeff[a][1])
           for a in self.line_line_coeff})

    self._phase = CosSum.make_function(self._phase_coeff)
    self._line_line = CosSum.make_function(self._line_line_coeff)

  @property
  def phase(self):
    return self._phase

  @property
  def line_line(self):
    return self._line_line

  @property
  def phase_coeff(self):
    return self._phase_coeff

  @property
  def line_line_coeff(self):
    return self._line_line_coeff

  @staticmethod
  def make_function(coeff):
    terms = tuple(CosSum._do_cos(a, *coeff[a]) for a in coeff)
    def r(theta):
      return numpy.sum(term(theta) for term in terms)
    return r

  @staticmethod
  def _do_cos(a, b, c):
    def r(theta):
      return b * numpy.cos(a * theta + c)
    return r

_RPM_TO_RAD_S = numpy.pi * 2 / 60
"""RPM / (rad/s)"""

class Motor(object):
  r'''Represents one type of motor.

  This currently only supports motors with flux linkages which are well-modeled
  as sums of sinusoids, but the API is designed to support other shapes in the
  future too.

  All internal constants are phase-neutral. All angles are electrical radians.
  All constants are in standard SI units (specifically: volts, seconds,
  newton*meters, amps, and ohms).

  The mutual inductance is assumed to be 0. This is true by definition for
  wye-connected motors (with floating neutral) due to the linear dependence
  between the phase currents.

  Flux linkage coefficients are represented as a mapping from a to (b, c) such
  that the final function is $\sum b * cos(a * \theta + c)$.

  The advertised_* numbers are the ones from the manufacturer, or some other
  source. They should not be used for any calculations, because it's unclear
  which kinds of currents and voltages they're using.

  electrical_ratio is the number of electrical rotations per mechanical one.
  It doesn't affect any electrical calculations, but it is useful for
  calculating mechanical gear ratios etc.
  '''
  def __init__(self,
               phase_resistance = None, line_line_resistance = None,
               phase_f_coeff = None, line_line_f_coeff = None,
               phase_self_inductance = None, line_line_self_inductance = None,
               advertised_rpm = None, advertised_voltage = None,
               advertised_kv = None,
               electrical_ratio = 1):
    '''Callers must specify each quantity in exactly one way.

    phase_resistance is the resistance of one phase.
    line_line_resistance is the resistance in one phase and out another.
    line_line_self_inductance is the inductance in one phase and out another.
    line_line_f_coeff is the coefficients for the flux linkage (derivative)
      function between two phases, in V/(rad/s) aka N*m/A.
    '''
    if phase_resistance is not None:
      self._resistance = phase_resistance
    elif line_line_resistance is not None:
      self._resistance = line_line_resistance / 2
    else:
      raise ValueError('Must specify the resistance')

    if phase_self_inductance is not None:
      self._self_inductance = phase_self_inductance
    elif line_line_self_inductance is not None:
      self._self_inductance = line_line_self_inductance / 2
    else:
      raise ValueError('Must specify the inductance')

    if phase_f_coeff is not None:
      self._f = CosSum(phase = phase_f_coeff)
    elif line_line_f_coeff is not None:
      self._f = CosSum(line_line = line_line_f_coeff)
    else:
      raise ValueError('Must specify the flux linkage')

    assert self._f.phase_coeff == _offset_coeff(self._f.phase_coeff)
    assert self._f.line_line_coeff == _offset_coeff(self._f.line_line_coeff)

    if advertised_rpm is not None:
      self._advertised_omega = advertised_rpm * _RPM_TO_RAD_S * electrical_ratio
    else:
      self._advertised_omega = None
    self._advertised_voltage = advertised_voltage
    if advertised_kv is not None:
      self._advertised_kv = advertised_kv * _RPM_TO_RAD_S * electrical_ratio
    else:
      self._advertised_kv = None

    self._electrical_ratio = electrical_ratio

  @property
  def f(self):
    return self._f.phase

  @property
  def line_line_f(self):
    return self._f.line_line

  @property
  def f_coeff(self):
    return self._f.phase_coeff

  @property
  def line_line_f_coeff(self):
    return self._f.line_line_coeff

  @property
  def resistance(self):
    return self._resistance

  @property
  def self_inductance(self):
    return self._self_inductance

  @property
  def advertised_kv(self):
    if self._advertised_kv is not None:
      return self._advertised_kv
    if (self._advertised_omega is not None and
        self._advertised_voltage is not None):
      return self._advertised_omega / self._advertised_voltage
    return None

  @property
  def electrical_ratio(self):
    return self._electrical_ratio

  def __repr__(self):
    return 'Motor(%s)' % ', '.join([
        'phase_resistance=%f' % self.resistance,
        'phase_f_coeff=%r' % dict(self.f_coeff),
        'phase_self_inductance=%f' % self.self_inductance,
        'advertised_omega=%r' % self._advertised_omega,
        'advertised_voltage=%r' % self._advertised_voltage,
        'advertised_kv=%r' % self._advertised_kv,
        'electrical_ratio=%f' % self.electrical_ratio,
        ])

def _cos_harmonic(cos_scalar, harmonic_scalar, harmonic_index, harmonic_offset):
  def r(theta):
    return (cos_scalar * numpy.cos(theta) +
            harmonic_scalar * numpy.cos(
                theta * harmonic_index + harmonic_offset))
  return r

class Waveform(object):
  def __init__(self, scalar_f):
    self.__doc__ = scalar_f.__doc__
    self._f = _vectorize_float(scalar_f)
    one_sixth_min = min(self._f(numpy.linspace(0, numpy.pi * 2, 12)))
    continuous_min = scipy.optimize.minimize(self._f,
                                             x0=(numpy.pi * 3 / 2),
                                             bounds=((numpy.pi, numpy.pi * 2),))
    assert continuous_min.success
    global_min = scipy.optimize.differential_evolution(
        self._f, bounds=((numpy.pi, numpy.pi * 2),), polish=True)
    assert global_min.success
    self._min = numpy.array([one_sixth_min,
                             float(self._f(continuous_min.x[0])),
                             float(self._f(global_min.x[0])),
                             ]).min()

  def __call__(self, *args):
    return self._f(*args)

  @property
  def min(self):
    return self._min

  @property
  def max(self):
    return -self._min

def _trapezoid(theta):
  '''A trapezoid with 120-degree flat regions.

  This is the ideal flux linkage for a trapezoidal motor.'''
  one_sixth = numpy.pi / 3
  theta = (theta - one_sixth / 2) % (numpy.pi * 2)
  if theta < one_sixth * 2:
    return 1
  elif theta < one_sixth * 3:
    return (theta - one_sixth * 2.5) / one_sixth * -2
  elif theta < one_sixth * 5:
    return -1
  else:
    return (theta - one_sixth * 5.5) / one_sixth * 2
trapezoid = Waveform(_trapezoid)

def _trapezoid_6step(theta):
  '''A 6-step "trapezoid".

  When people talk about "trapezoidal commutation", they usually mean this as
  the idealized phase current waveform.'''
  one_sixth = numpy.pi / 3
  theta = theta % (numpy.pi * 2)
  if theta < one_sixth * 1:
    return 0.5
  elif theta < one_sixth * 2:
    return 1
  elif theta < one_sixth * 3:
    return 0.5
  elif theta < one_sixth * 4:
    return -0.5
  elif theta < one_sixth * 5:
    return -1
  else:
    return -0.5
trapezoid_6step = Waveform(_trapezoid_6step)

def _trapezoid_4step(theta):
  '''A 4-step kind-of-trapezoid. This only has the 120-degree flat regions, and
  0 elsewhere.

  Some people call this a "modified square wave", and others call it
  "trapezoidal".
  When people talk about "trapezoidal commutation", they  usually mean this as
  the phase voltage waveform.'''
  one_sixth = numpy.pi / 3
  theta = (theta - one_sixth / 2) % (numpy.pi * 2)
  if theta < one_sixth * 2:
    return 1
  elif theta < one_sixth * 3:
    return 0
  elif theta < one_sixth * 5:
    return -1
  else:
    return 0
trapezoid_4step = Waveform(_trapezoid_4step)

def _square(theta):
  '''A 2-step square wave.

  This is a really crude commutation scheme. Not many people talk about actually
  using it.'''
  theta = theta % (numpy.pi * 2)
  if theta < numpy.pi:
    return 1
  else:
    return -1
square = Waveform(_square)

sin = Waveform(numpy.sin)

def make_sin_constant(coeff):
  '''Returns a Waveform which will produce constant torque for the given motor
  waveform. Passing coeff=Motor.f_coeff often makes sense.'''
  assert coeff == _offset_coeff(coeff)
  coeff = dict(coeff)
  assert len(coeff) <= 2
  for i in sorted(coeff.keys())[1:]:
    coeff[i] = (-coeff[i][0], coeff[i][1])
  return Waveform(CosSum.make_function(coeff))

# TODO(Brian): Actually measure the inductance.
BOMA = Motor(line_line_resistance = 0.638 / 3.77,
             line_line_self_inductance = 0.38e-3,
             line_line_f_coeff = {1: (0.03382623, 0), 7: (0.00343913, 0)},
             advertised_rpm = 4800, advertised_voltage = 48,
             electrical_ratio = 3)
# TODO(Brian): Actually measure the inductance.
MY1020 = Motor(line_line_resistance = 0.650 / 3.77,
               line_line_self_inductance = 0.38e-3,
               line_line_f_coeff = {1: (0.03202452, 0), 7: (0.00242868, 0)},
               advertised_rpm = 4500, advertised_voltage = 48,
               electrical_ratio = 3)
# TODO(Brian): Verify resistance and inductance on a power supply.
T20 = Motor(phase_resistance = 0.0065,
            phase_self_inductance = 5.0e-6,
            line_line_f_coeff = {1: (0.00660802, 0), 5: (0.00097149, 0)},
            advertised_kv = 730, advertised_voltage = 41,
            electrical_ratio = 2)
