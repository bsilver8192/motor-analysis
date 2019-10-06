import simulation
import numpy
import scipy

class SimpleController(simulation.MotorController):
  def __init__(self, motor, phase_g):
    super().__init__(motor)

    self._phase_g = phase_g
    def line_line_g(theta):
      return phase_g(theta) - phase_g(theta + numpy.pi * 2 / 3)
    self._line_line_g = line_line_g

    def torque(theta):
      r = self.motor.f(theta) * self.phase_g(theta)
      assert r >= -1e-20, '%f: %f * %f = %f' % (theta, self.motor.f(theta), self.phase_g(theta), r)
      return r
    self._unit_average_torque = simulation.average_circle(torque) * 3
    self._unit_rms_torque = numpy.sqrt(
        simulation.average_circle(lambda t: torque(t)**2)) * 3
    self._unit_average_current = simulation.average_circle(
        lambda t: numpy.abs(self.phase_g(t)))
    self._unit_rms_current = simulation.rms_circle(self.phase_g)

    thetas = numpy.linspace(0, numpy.pi * 2, 1000)
    self._max_speed = 1 / numpy.amax(self.motor.line_line_f(thetas))

    def input_voltage(theta):
      thetas = (theta, theta + numpy.pi * 2 / 3,
                theta - numpy.pi * 2 / 3)
      from_resistance = self.phase_g(thetas) * self.motor.resistance
      from_inductance = (simulation.differentiate(self.phase_g, thetas) *
                          self.motor.self_inductance)
      voltages = from_resistance + from_inductance
      return numpy.amax(numpy.abs((voltages[0] - voltages[1],
                                    voltages[0] - voltages[2],
                                    voltages[1] - voltages[2])))
    self._unit_voltage = simulation.max_circle(input_voltage)

  def max_speed(self):
    return self._max_speed

  def operating_point(self, omega,
                      max_torque = None,
                      max_motor_current = None,
                      max_input_power = None,
                      max_voltage = None,
                      ):
    current_scale = numpy.inf

    if max_torque is not None:
      current_scale = min(current_scale, max_torque / self._unit_average_torque)

    if max_motor_current is not None:
      current_scale = min(current_scale,
                          max_motor_current / self._unit_rms_current)

    unit_rms_electrical_power = (self._unit_rms_current**2 *
                                 self.motor.resistance * 3)
    unit_rms_mechanical_power = self._unit_rms_torque * omega
    if max_input_power is not None:
      # input_power = unit_mechanical_power*x + unit_electrical_power*x^2
      # 0 = unit_electrical * x^2 + unit_mechanical * x + -input_power
      # Then, use the quadratic formula.
      # TODO(Brian): Use the other solution for motor braking.
      new_scale = ((-unit_rms_mechanical_power +
                    numpy.sqrt(unit_rms_mechanical_power ** 2 -
                               4 * unit_rms_electrical_power * -max_input_power)) /
                   (2 * unit_rms_electrical_power))
      current_scale = min(current_scale, new_scale)

    if max_voltage is not None:
      bemf_voltage = omega / self.max_speed()
      assert bemf_voltage <= max_voltage, 'TODO(Brian): braking not supported yet'
      current_scale = min(current_scale, (max_voltage - bemf_voltage) / self._unit_voltage)

    assert current_scale < numpy.inf, 'Need to specify at least one limit'

    rms_motor_power = unit_rms_electrical_power * current_scale**2
    rms_output_power = unit_rms_mechanical_power * current_scale
    average_motor_power = (((self._unit_average_current * current_scale) ** 2) *
                           self.motor.resistance * 3)
    return simulation.OperatingPoint(omega=omega,
                                     rms_motor_power=rms_motor_power,
                                     rms_output_power=rms_output_power,
                                     average_motor_power=average_motor_power,
                                     torque=self._unit_average_torque * current_scale)

  @property
  def phase_g(self):
    return self._phase_g

  @property
  def line_line_g(self):
    return self._line_line_g

  def __repr__(self):
    return 'SimpleController(%r, %r)' % (self.motor, self.phase_g)
