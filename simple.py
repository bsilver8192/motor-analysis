import simulation
import numpy
import scipy

class SimpleController(simulation.MotorController):
  def __init__(self, motor, phase_g):
    super().__init__(motor)

    self._phase_g = phase_g
    def line_line(theta):
      return phase_g(theta) - phase_g(theta + numpy.pi * 2 / 3)
    self._line_line_g = line_line

    def torque(theta):
      return self.motor.f(theta) * self.phase_g(theta)
    self._unit_torque = simulation.average_circle(torque) * 3
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
      current_scale = min(current_scale, max_torque / self._unit_torque)

    if max_motor_current is not None:
      current_scale = min(current_scale,
                          max_motor_current / self._unit_rms_current)

    unit_mechanical_power = self._unit_torque * omega
    unit_rms_electrical_power = (self._unit_rms_current**2 *
                                 self.motor.resistance * 3)
    if max_input_power is not None:
      # input_power = unit_mechanical_power*x + unit_rms_electrical_power*x^2
      # 0 = unit_electrical * x^2 + unit_mechanical * x + -input_power
      new_scale = ((-unit_mechanical_power +
                    numpy.sqrt(unit_mechanical_power ** 2 -
                               4 * unit_rms_electrical_power * -max_input_power)) /
                   (2 * unit_rms_electrical_power))
      current_scale = min(current_scale, new_scale)

    if max_voltage is not None:
      bemf_voltage = omega / self.max_speed()
      assert bemf_voltage <= max_voltage, 'braking not supported yet'
      current_scale = min(current_scale, (max_voltage - bemf_voltage) / self._unit_voltage)

    motor_power = unit_rms_electrical_power * current_scale**2
    return simulation.OperatingPoint(omega=omega,
                                     motor_power=motor_power,
                                     torque=self._unit_torque * current_scale)

  @property
  def phase_g(self):
    return self._phase_g

  @property
  def line_line_g(self):
    return self._line_line_g
