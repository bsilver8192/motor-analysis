import numpy

def _vectorize_float(f):
  return numpy.vectorize(f, otypes=(numpy.float,))

class CosSum(object):
  def __init__(self,
               line_line = None, phase = None):
    if phase is not None:
      self._phase_coeff = phase
      self._line_line_coeff = {a: (phase[a][0] * numpy.sqrt(3), phase[a][1])
                               for a in phase}

    if line_line is not None:
      self._line_line_coeff = line_line
      self._phase_coeff = {a: (line_line[a][0] / numpy.sqrt(3), line_line[a][1])
                           for a in line_line}

    self._phase = CosSum._make_function(self._phase_coeff)
    self._line_line = CosSum._make_function(self._line_line_coeff)

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
  def _make_function(coeff):
    terms = tuple(CosSum._do_cos(a, *coeff[a]) for a in coeff)
    def r(theta):
      return numpy.sum(term(theta) for term in terms)
    return r

  @staticmethod
  def _do_cos(a, b, c):
    def r(theta):
      return b * numpy.cos(a * theta + c)
    return r

class Motor(object):
  '''Represents one type of motor.

  All internal constants are phase-neutral. All angles are electrical radians.
  All constants are in standard SI units (specifically: volts, seconds,
  newton*meters, amps, and ohms).

  Flux linkages are represented as a mapping from a to (b, c) such that the
  final function is $\sum b * cos(a * \theta + c)$.
  '''
  def __init__(self,
               phase_resistance = None, line_line_resistance = None,
               line_line_f = None):
    '''Callers must specify each quantity in exactly one way.

    phase_resistance is the resistance of one phase.
    line_line_resistance is the resistance in one phase and out another.
    line_line_f is the flux linkage (derivative) function between
      two phases, in V/(rad/s) aka N*m/A.
    '''
    if phase_resistance is not None:
      self._resistance = phase_resistance
    if line_line_resistance is not None:
      self._resistance = line_line_resistance / 2

    if line_line_f is not None:
      self._f = CosSum(line_line_f)

  @property
  def f(self):
    return self._f.phase

  @property
  def line_line_f(self):
    return self._f.line_line

  @property
  def resistance(self):
    return self.resistance

def _cos_harmonic(cos_scalar, harmonic_scalar, harmonic_index, harmonic_offset):
  def r(theta):
    return (cos_scalar * numpy.cos(theta) +
            harmonic_scalar * numpy.cos(
                theta * harmonic_index + harmonic_offset))
  return r

BOMA = Motor(line_line_resistance = 0.638 / 3.77,
             line_line_f = {1: (0.033826, 0), 7: (0.003439, 1.5707960)})
MY1020 = Motor(line_line_resistance = 0.650 / 3.77,
               line_line_f = {1: (0.032025, 0), 7: (0.002429, 1.047198)})
T20 = Motor(phase_resistance = 0.0079,
            line_line_f = {1: (0.006595, 0), 5: (0.000970, -1.570796)})
