r'''This module supports simulating various aspects of motors and control
algorithms.

Simple linear/sinusoidal functions helpful for these simulations are in the
models module instead.

We model a motor's torque as
$\tau = 3 \int_0^{2\pi} f(\theta) g(\theta) \mathrm(d)\theta$
where $f$ is the (phase) flux linkage (derivative) and $g$ is the (phase)
current.
We only model motors at a steady operating point. This means we can say
$V(\theta) = g(\theta) \times R + \frac{dg}{dt}(\theta) + f(\theta)$ aka
$V(\theta) = g(\theta) \times R + \frac{dg}{d\theta}(\theta) \times \omega + f(\theta)$
where $V$ is the phase voltage.
We then know that $V(\theta) + V(\theta + \frac{2\pi}{3}) + V(\theta - \frac{2\pi}{3}) < V_{bat}$
and the RMS current is
$3 \sqrt{\int_0^{2\pi} g(\theta)^2}$.

It is common to drive motors with step changes in voltage. Ignoring motor
inductance, these result in step changes to the current too, but that results in
infinities when trying to simulate that make the math fall apart.
However, we also want to support driving known current waveforms, because that
is an easier domain to optimize in.
To support both of these, we do simulations both with a current waveform (backed
out to a voltage to bound it) or a voltage waveform (used to calculate a current
waveform to calculate torque).

We ignore the PWM switching effects. At steady state, and assuming the switching
is much faster than the motors rotation, it doesn't matter. It's possible to
make any shape, and a maximum duty cycle is effectively a reduction in the
input voltage. Even if the duty cycle vs average voltage is non-linear, it still
doesn't affect the simulations we're doing here, because we're not attempting to
test controls algorithms which actually drive the currents/voltages we simulate.

We model the relative efficiency of different gearing choices with a simple
multipler of the torque. This should be sufficient to capture tradeoffs between
one and two reductions, for example.
'''

import numpy
import scipy.optimize
import scipy.integrate

def average_circle(f):
  return scipy.integrate.quad(f, 0, numpy.pi * 2)[0] / (numpy.pi * 2)

def rms_circle(f):
  return numpy.sqrt(average_circle(lambda t: f(t)**2))

def max_circle(f):
  n = 200
  results = numpy.empty((n,))
  epsilon = numpy.pi * 2 / n / 2
  i = 0
  for theta in numpy.linspace(0, numpy.pi * 2, n):
    result = scipy.optimize.minimize(lambda t: -f(t), x0=(theta,),
                                     bounds=((theta - epsilon,
                                              theta + epsilon),))
    if result.success:
      results[i] = float(result.x)
    i += 1
  return f(numpy.nanmin(results))

def _differentiate(f, theta):
  """Calculates the first derivative of f at theta.

  Arguments
  ---------
  f : callable
      Function from theta to a scalar.
  theta : float
      The point to evaluate the derivative at.

  Notes
  -----
  f must be a function of theta for the scale of the region we calculate the
  derivative over to make sense.
  """
  epsilon = numpy.pi * 2 / 10000
  values = f((theta + epsilon, theta - epsilon))
  return (values[1] - values[0]) / (epsilon * 2)
differentiate = numpy.vectorize(_differentiate, otypes=(numpy.float,),
                                excluded = ['f'])

class OperatingPoint(object):
  """Represents one operating point of a motor.

  This is intended to include all of the useful output metrics for an operating
  point (speed, torque, etc) of a motor.

  Attributes
  ----------
  omega : float
      The speed of the motor in rad/s.
  motor_power : float
      The RMS power dissipated in the motor in W.
  torque : float
      The average torque (for all phases) in N*m.
  """
  def __init__(self, omega, motor_power, torque):
    self._omega = omega
    self._motor_power = motor_power
    self._torque = torque

  @property
  def omega(self):
    return self._omega

  @property
  def input_power(self):
    """Calculates the power fed into the motor controller.

    Returns
    -------
    float
        The RMS input power in W.
    """
    return self.motor_power + self.output_power

  @property
  def torque(self):
    return self._torque

  @property
  def motor_power(self):
    return self._motor_power

  def input_current(self, input_voltage):
    """
    Arguments
    ---------
    input_voltage : float
        The input voltage (in V).

    Returns
    -------
    float
        The input current (in A).
    """
    return self.input_power / input_voltage

  @property
  def output_power(self):
    """
    Returns
    -------
    float
        The output power (in W).
    """
    return self.omega * self.torque

  @property
  def efficiency(self):
    """
    Returns
    -------
    float
        The efficiency, in [0, 1].
    """
    return self.output_power / (self.input_power + self.output_power)

class MotorController(object):
  """
  Attributes
  ----------
  motor : models.Motor
  """
  def __init__(self, motor):
    self._motor = motor

  @property
  def motor(self):
    return self._motor

  def max_speed(self):
    """
    Calculates the maximum speed at 1V.

    For subclasses to implement.

    This speed scales linearly with voltage.

    Returns
    -------
    float
        The maximum speed with 1V in rad/s.
    """
    pass

  def operating_point(self, omega,
                      max_torque = None,
                      max_motor_current = None,
                      max_input_power = None,
                      max_voltage = None,
                      ):
    """
    Calculates one operating point for the motor (at a given speed).

    For subclasses to implement.

    TODO(Brian): Define all this in quadrants II and IV (motor braking) and
    make sure all the implementations do it correctly.

    At least one of the max_* arguments must be specified.

    Arguments
    ---------
    omega : float
        The speed the motor is spinning at in rad/s.
    max_torque : float, optional
        The maximum average torque to develop (across all phases) in N*m.
    max_motor_current : float, optional
        The maximum motor current to use (RMS for one phase in A).
    max_input_power : float, optional
        The maximum input power to use (RMS in W).
    max_voltage : float, optional
        The maximum input voltage to use in V.

    Returns
    -------
    OperatingPoint
    """
    pass
