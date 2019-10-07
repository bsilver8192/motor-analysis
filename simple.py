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

    def phase_average_sane(phase, total):
      return round(phase * 3 - total, 7) == 0

    def phase_torque(theta):
      r = self.motor.f(theta) * self.phase_g(theta)
      assert (r >= -1e-20).all(), '%f: %f * %f = %f' % (theta, self.motor.f(theta), self.phase_g(theta), r)
      return r
    def total_torque(theta):
      return simulation.three_phases(phase_torque, theta)
    # N*m for one phase.
    self._unit_phase_average_torque = simulation.average_circle(phase_torque)
    assert phase_average_sane(self._unit_phase_average_torque, simulation.average_circle(total_torque))
    # N*m for all phases *at once*.
    self._unit_total_rms_torque = simulation.rms_circle(total_torque)
    # N*m for one phase.
    self._unit_phase_rms_torque = simulation.rms_circle(phase_torque)

    def abs_phase_current(theta):
      return numpy.abs(self.phase_g(theta))
    def total_current(theta):
      return simulation.three_phases(abs_phase_current, theta)
    # A (absolute value) for a single phase.
    self._unit_phase_average_current = simulation.average_circle(abs_phase_current)
    assert phase_average_sane(self._unit_phase_average_current, simulation.average_circle(total_current))
    # A (absolute value) for all phases *at once*.
    self._unit_total_rms_current = simulation.rms_circle(total_current)
    # A for a single phase.
    # Taking the absolute value changes nothing for RMS of a single quantity.
    self._unit_phase_rms_current = simulation.rms_circle(self.phase_g)

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
    scale_limits = []

    if max_torque is not None:
      scale_limits.append(max_torque / (self._unit_phase_average_torque * 3))

    if max_motor_current is not None:
      scale_limits.append(max_motor_current / self._unit_phase_rms_current)

    # W/A burned as heat for all phases.
    unit_rms_electrical_power = (self._unit_total_rms_current**2 *
                                 self.motor.resistance)
    # W/A turned into torque for all phases.
    unit_rms_mechanical_power = self._unit_total_rms_torque * omega
    if max_input_power is not None:
      # input_power = unit_mechanical_power*x + unit_electrical_power*x^2
      # 0 = unit_electrical * x^2 + unit_mechanical * x + -input_power
      # Then, use the quadratic formula.
      # TODO(Brian): Use the other solution for motor braking.
      new_scale = ((-unit_rms_mechanical_power +
                    numpy.sqrt(unit_rms_mechanical_power ** 2 -
                               4 * unit_rms_electrical_power * -max_input_power)) /
                   (2 * unit_rms_electrical_power))
      scale_limits.append(new_scale)

    if max_voltage is not None:
      bemf_voltage = omega / self.max_speed()
      assert bemf_voltage <= max_voltage, 'TODO(Brian): braking not supported yet'
      scale_limits.append((max_voltage - bemf_voltage) / self._unit_voltage)

    assert scale_limits, 'Need to specify at least one limit'
    final_scale = numpy.nanmin(scale_limits)

    rms_motor_power = (((self._unit_phase_rms_current * final_scale) ** 2) *
                       self.motor.resistance * 3)
    rms_output_power = unit_rms_mechanical_power * final_scale
    rms_input_power = rms_output_power + unit_rms_electrical_power * final_scale**2
    average_motor_power = (((self._unit_phase_average_current * final_scale) ** 2) *
                           self.motor.resistance * 3)
    return simulation.OperatingPoint(
        omega=omega,
        rms_motor_power=rms_motor_power,
        rms_output_power=rms_output_power,
        rms_input_power=rms_input_power,
        average_motor_power=average_motor_power,
        torque=self._unit_phase_average_torque * final_scale * 3,
      )

  @property
  def phase_g(self):
    return self._phase_g

  @property
  def line_line_g(self):
    return self._line_line_g

  def __repr__(self):
    return 'SimpleController(%r, %r)' % (self.motor, self.phase_g)
