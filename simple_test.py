#!/usr/bin/python3

import unittest
import numpy

import simple
import models

_one_offset = -numpy.pi / 2

_MOTOR1 = models.Motor(phase_resistance = 1,
                       phase_self_inductance = 1,
                       phase_f_coeff = {1: (1, 0), 5: (0.2, 0)},
                       electrical_ratio = 1)

_MOTOR2 = models.Motor(phase_resistance = 1,
                       phase_self_inductance = 1,
                       phase_f_coeff = {1: (1, 0), 5: (0.003, 0)},
                       electrical_ratio = 1)

_MOTOR3 = models.Motor(phase_resistance = 1,
                       phase_self_inductance = 1,
                       phase_f_coeff = {1: (1, 0), 7: (0.05, 0)},
                       electrical_ratio = 1)

_CONTROLLERS = (
    simple.SimpleController(models.Motor(phase_resistance = 1,
                                         phase_self_inductance = 1,
                                         phase_f_coeff = {1: (1, 0)},
                                         electrical_ratio = 1),
                            models.sin),
    simple.SimpleController(models.Motor(phase_resistance = 1,
                                         phase_self_inductance = 1,
                                         phase_f_coeff = {1: (1, 0)},
                                         electrical_ratio = 1),
                            lambda t: models.sin(t) * 13),
    simple.SimpleController(models.Motor(phase_resistance = 1,
                                         phase_self_inductance = 1,
                                         phase_f_coeff = {1: (1, 0), 5: (0.5, 0)},
                                         electrical_ratio = 1),
                            models.sin),
    simple.SimpleController(models.Motor(phase_resistance = 1,
                                         phase_self_inductance = 1,
                                         phase_f_coeff = {1: (1, 0), 5: (0.003, 0)},
                                         electrical_ratio = 1),
                            models.sin),
    simple.SimpleController(models.Motor(phase_resistance = 1,
                                         phase_self_inductance = 1,
                                         phase_f_coeff = {1: (1, 0)},
                                         electrical_ratio = 1),
                            models.Waveform(models.CosSum.make_function(
                                {1: (1, _one_offset), 5: (0.5, _one_offset)}))),
    simple.SimpleController(models.Motor(phase_resistance = 1,
                                         phase_self_inductance = 1,
                                         phase_f_coeff = {1: (1, 0)},
                                         electrical_ratio = 1),
                            models.Waveform(models.CosSum.make_function(
                                {1: (1, _one_offset), 5: (0.003, _one_offset)}))),
    simple.SimpleController(_MOTOR1,
                            models.make_sin_constant(_MOTOR1.f_coeff)),
    simple.SimpleController(_MOTOR2,
                            models.make_sin_constant(_MOTOR2.f_coeff)),
    simple.SimpleController(_MOTOR3,
                            models.make_sin_constant(_MOTOR3.f_coeff)),
    simple.SimpleController(_MOTOR3,
                            models.make_sin_constant(_MOTOR1.f_coeff)),
    )

class SimpleControllerTest(unittest.TestCase):
  def test_max_torque(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_torque = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.rms_input_power, 0)
        self.assertEqual(stopped.average_input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.rms_motor_power, 0)
        self.assertEqual(stopped.rms_input_current(1), 0)
        self.assertEqual(stopped.average_output_power, 0)
        self.assertEqual(stopped.rms_output_power, 0)
        self.assertEqual(stopped.average_motor_power, 0)

        fast_none = controller.operating_point(100, max_torque = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertEqual(fast_none.rms_input_power, 0)
        self.assertEqual(fast_none.average_input_power, 0)
        self.assertEqual(fast_none.torque, 0)
        self.assertEqual(fast_none.rms_motor_power, 0)
        self.assertEqual(fast_none.rms_input_current(1), 0)
        self.assertEqual(fast_none.average_output_power, 0)
        self.assertEqual(fast_none.rms_output_power, 0)
        self.assertEqual(fast_none.average_motor_power, 0)

        stall_low = controller.operating_point(0, max_torque = 0.001)
        self.assertEqual(stall_low.omega, 0)
        self.assertGreater(stall_low.rms_input_power, 0)
        self.assertGreater(stall_low.average_input_power, 0)
        self.assertAlmostEqual(stall_low.torque, 0.001)
        self.assertGreater(stall_low.rms_motor_power, 0)
        self.assertGreater(stall_low.rms_input_current(1), 0)
        self.assertEqual(stall_low.average_output_power, 0)
        self.assertEqual(stall_low.rms_output_power, 0)
        self.assertGreater(stall_low.average_motor_power, 0)
        self.assertEqual(stall_low.rms_input_power, stall_low.rms_motor_power)
        self.assertEqual(stall_low.average_input_power, stall_low.average_motor_power)
        self.assertGreaterEqual(stall_low.rms_input_power, stall_low.average_input_power)

        stall_high = controller.operating_point(0, max_torque = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertGreater(stall_high.rms_input_power, 0)
        self.assertGreater(stall_high.average_input_power, 0)
        self.assertAlmostEqual(stall_high.torque, 1000)
        self.assertGreater(stall_high.rms_motor_power, 0)
        self.assertGreater(stall_high.rms_input_current(1), 0)
        self.assertEqual(stall_high.average_output_power, 0)
        self.assertEqual(stall_high.rms_output_power, 0)
        self.assertGreater(stall_high.average_motor_power, 0)
        self.assertEqual(stall_high.rms_input_power, stall_high.rms_motor_power)
        self.assertEqual(stall_high.average_input_power, stall_high.average_motor_power)
        self.assertGreaterEqual(stall_high.rms_input_power, stall_high.average_input_power)

        fast_low = controller.operating_point(10000, max_torque = 0.001)
        self.assertEqual(fast_low.omega, 10000)
        self.assertGreater(fast_low.rms_input_power, 0)
        self.assertGreater(fast_low.average_input_power, 0)
        self.assertAlmostEqual(fast_low.torque, 0.001)
        self.assertGreater(fast_low.rms_motor_power, 0)
        self.assertGreater(fast_low.rms_input_current(1), 0)
        self.assertGreater(fast_low.average_output_power, 0)
        self.assertGreater(fast_low.rms_output_power, 0)
        self.assertGreater(fast_low.average_motor_power, 0)
        self.assertGreaterEqual(fast_low.rms_output_power, fast_low.average_output_power)
        self.assertGreaterEqual(fast_low.rms_input_power, fast_low.average_input_power)
        self.assertEqual(fast_low.rms_input_power,
                         fast_low.rms_motor_power + fast_low.rms_output_power)
        self.assertEqual(fast_low.average_input_power,
                         fast_low.average_motor_power + fast_low.average_output_power)

        fast_high = controller.operating_point(10000, max_torque = 1000)
        self.assertEqual(fast_high.omega, 10000)
        self.assertGreater(fast_high.rms_input_power, 0)
        self.assertGreater(fast_high.average_input_power, 0)
        self.assertAlmostEqual(fast_high.torque, 1000)
        self.assertGreater(fast_high.rms_motor_power, 0)
        self.assertGreater(fast_high.rms_input_current(1), 0)
        self.assertGreater(fast_high.average_output_power, 0)
        self.assertGreater(fast_high.average_motor_power, 0)
        self.assertGreaterEqual(fast_high.rms_output_power, fast_high.average_output_power)
        self.assertGreaterEqual(fast_high.rms_input_power, fast_high.average_input_power)
        self.assertEqual(fast_high.rms_input_power,
                         fast_high.rms_motor_power + fast_high.rms_output_power)
        self.assertEqual(fast_high.average_input_power,
                         fast_high.average_motor_power + fast_high.average_output_power)

        self.assertAlmostEqual(stall_high.rms_motor_power,
                               stall_low.rms_motor_power * (
                                   (stall_high.torque / stall_low.torque) ** 2))
        self.assertAlmostEqual(fast_high.rms_motor_power,
                               fast_low.rms_motor_power * (
                                   (fast_high.torque / fast_low.torque) ** 2))

  def test_max_motor_current(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_motor_current = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.rms_input_power, 0)
        self.assertEqual(stopped.average_input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.rms_motor_power, 0)
        self.assertEqual(stopped.rms_input_current(1), 0)
        self.assertEqual(stopped.average_output_power, 0)
        self.assertEqual(stopped.rms_output_power, 0)
        self.assertEqual(stopped.average_motor_power, 0)

        fast_none = controller.operating_point(100, max_motor_current = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertEqual(fast_none.rms_input_power, 0)
        self.assertEqual(fast_none.average_input_power, 0)
        self.assertEqual(fast_none.torque, 0)
        self.assertEqual(fast_none.rms_motor_power, 0)
        self.assertEqual(fast_none.rms_input_current(1), 0)
        self.assertEqual(fast_none.average_output_power, 0)
        self.assertEqual(fast_none.rms_output_power, 0)
        self.assertEqual(fast_none.average_motor_power, 0)

        stall_low = controller.operating_point(0, max_motor_current = 0.001)
        self.assertEqual(stall_low.omega, 0)
        self.assertGreater(stall_low.rms_input_power, 0)
        self.assertGreater(stall_low.average_input_power, 0)
        self.assertGreater(stall_low.torque, 0)
        self.assertAlmostEqual(stall_low.rms_motor_power / 3,
                               0.001**2 * controller.motor.resistance)
        self.assertGreater(stall_low.rms_input_current(1), 0)
        self.assertEqual(stall_low.average_output_power, 0)
        self.assertEqual(stall_low.rms_output_power, 0)
        self.assertGreater(stall_low.average_motor_power, 0)
        self.assertEqual(stall_low.rms_input_power, stall_low.rms_motor_power)
        self.assertEqual(stall_low.average_input_power, stall_low.average_motor_power)
        self.assertGreaterEqual(stall_low.rms_input_power, stall_low.average_input_power)

        stall_high = controller.operating_point(0, max_motor_current = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertGreater(stall_high.rms_input_power, 0)
        self.assertGreater(stall_high.average_input_power, 0)
        self.assertGreater(stall_high.torque, 0)
        self.assertAlmostEqual(stall_high.rms_motor_power / 3,
                               1000**2 * controller.motor.resistance)
        self.assertGreater(stall_high.rms_input_current(1), 0)
        self.assertEqual(stall_high.average_output_power, 0)
        self.assertEqual(stall_high.rms_output_power, 0)
        self.assertGreater(stall_high.average_motor_power, 0)
        self.assertEqual(stall_high.rms_input_power, stall_high.rms_motor_power)
        self.assertEqual(stall_high.average_input_power, stall_high.average_motor_power)
        self.assertGreaterEqual(stall_high.rms_input_power, stall_high.average_input_power)

        fast_low = controller.operating_point(10000, max_motor_current = 0.001)
        self.assertEqual(fast_low.omega, 10000)
        self.assertGreater(fast_low.rms_input_power, 0)
        self.assertGreaterEqual(stall_high.rms_input_power, stall_high.average_input_power)
        self.assertGreater(fast_low.torque, 0)
        self.assertAlmostEqual(fast_low.rms_motor_power / 3,
                               0.001**2 * controller.motor.resistance)
        self.assertGreater(fast_low.rms_input_current(1), 0)
        self.assertGreater(fast_low.average_output_power, 0)
        self.assertGreater(fast_low.rms_output_power, 0)
        self.assertGreater(fast_low.average_motor_power, 0)
        self.assertGreaterEqual(fast_low.rms_output_power, fast_low.average_output_power)
        self.assertGreaterEqual(fast_low.rms_input_power, fast_low.average_input_power)
        self.assertEqual(fast_low.rms_input_power,
                         fast_low.rms_motor_power + fast_low.rms_output_power)
        self.assertEqual(fast_low.average_input_power,
                         fast_low.average_motor_power + fast_low.average_output_power)

        fast_high = controller.operating_point(10000, max_motor_current = 1000)
        self.assertEqual(fast_high.omega, 10000)
        self.assertGreater(fast_high.rms_input_power, 0)
        self.assertGreater(fast_high.average_input_power, 0)
        self.assertGreater(fast_high.torque, 0)
        self.assertAlmostEqual(fast_high.rms_motor_power / 3,
                               1000**2 * controller.motor.resistance)
        self.assertGreater(fast_high.rms_input_current(1), 0)
        self.assertGreater(fast_high.average_output_power, 0)
        self.assertGreater(fast_high.average_motor_power, 0)
        self.assertGreaterEqual(fast_high.rms_output_power, fast_high.average_output_power)
        self.assertGreaterEqual(fast_high.rms_input_power, fast_high.average_input_power)
        self.assertEqual(fast_high.rms_input_power,
                         fast_high.rms_motor_power + fast_high.rms_output_power)
        self.assertEqual(fast_high.average_input_power,
                         fast_high.average_motor_power + fast_high.average_output_power)

        self.assertAlmostEqual(stall_high.rms_motor_power,
                               stall_low.rms_motor_power * (
                                   (stall_high.torque / stall_low.torque) ** 2))
        self.assertAlmostEqual(fast_high.rms_motor_power,
                               fast_low.rms_motor_power * (
                                   (fast_high.torque / fast_low.torque) ** 2))

  def test_max_input_power(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_input_power = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.rms_input_power, 0)
        self.assertEqual(stopped.average_input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.rms_motor_power, 0)
        self.assertEqual(stopped.rms_input_current(1), 0)
        self.assertEqual(stopped.average_output_power, 0)
        self.assertEqual(stopped.rms_output_power, 0)
        self.assertEqual(stopped.average_motor_power, 0)

        fast_none = controller.operating_point(100, max_input_power = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertEqual(fast_none.rms_input_power, 0)
        self.assertEqual(fast_none.average_input_power, 0)
        self.assertEqual(fast_none.torque, 0)
        self.assertEqual(fast_none.rms_motor_power, 0)
        self.assertEqual(fast_none.rms_input_current(1), 0)
        self.assertEqual(fast_none.average_output_power, 0)
        self.assertEqual(fast_none.rms_output_power, 0)
        self.assertEqual(fast_none.average_motor_power, 0)

        stall_low = controller.operating_point(0, max_input_power = 0.001)
        self.assertEqual(stall_low.omega, 0)
        self.assertAlmostEqual(stall_low.rms_input_power, 0.001)
        self.assertGreater(stall_low.average_input_power, 0)
        self.assertGreater(stall_low.torque, 0)
        self.assertAlmostEqual(stall_low.rms_motor_power, 0.001)
        self.assertGreater(stall_low.rms_input_current(1), 0)
        self.assertEqual(stall_low.average_output_power, 0)
        self.assertEqual(stall_low.rms_output_power, 0)
        self.assertGreater(stall_low.average_motor_power, 0)
        self.assertEqual(stall_low.rms_input_power, stall_low.rms_motor_power)
        self.assertEqual(stall_low.average_input_power, stall_low.average_motor_power)
        self.assertGreaterEqual(stall_low.rms_input_power, stall_low.average_input_power)

        stall_high = controller.operating_point(0, max_input_power = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertAlmostEqual(stall_high.rms_input_power, 1000)
        self.assertGreater(stall_high.average_input_power, 0)
        self.assertGreater(stall_high.torque, 0)
        self.assertAlmostEqual(stall_high.rms_motor_power, 1000)
        self.assertGreater(stall_high.rms_input_current(1), 0)
        self.assertEqual(stall_high.average_output_power, 0)
        self.assertEqual(stall_high.rms_output_power, 0)
        self.assertGreater(stall_high.average_motor_power, 0)
        self.assertEqual(stall_high.rms_input_power, stall_high.rms_motor_power)
        self.assertEqual(stall_high.average_input_power, stall_high.average_motor_power)
        self.assertGreaterEqual(stall_high.rms_input_power, stall_high.average_input_power)

        fast_low = controller.operating_point(10000, max_input_power = 0.001)
        self.assertEqual(fast_low.omega, 10000)
        self.assertAlmostEqual(fast_low.rms_input_power, 0.001)
        self.assertGreater(fast_low.average_input_power, 0)
        self.assertGreater(fast_low.torque, 0)
        self.assertGreater(fast_low.rms_motor_power, 0)
        self.assertGreater(fast_low.rms_input_current(1), 0)
        self.assertGreater(fast_low.average_output_power, 0)
        self.assertGreater(fast_low.rms_output_power, 0)
        self.assertGreater(fast_low.average_motor_power, 0)
        self.assertGreaterEqual(fast_low.rms_output_power, fast_low.average_output_power)
        self.assertGreaterEqual(fast_low.rms_input_power, fast_low.average_input_power)
        self.assertEqual(fast_low.rms_input_power,
                         fast_low.rms_motor_power + fast_low.rms_output_power)
        self.assertEqual(fast_low.average_input_power,
                         fast_low.average_motor_power + fast_low.average_output_power)

        fast_high = controller.operating_point(10000, max_input_power = 1000)
        self.assertEqual(fast_high.omega, 10000)
        self.assertAlmostEqual(fast_high.rms_input_power, 1000)
        self.assertGreater(fast_high.average_input_power, 0)
        self.assertGreater(fast_high.torque, 0)
        self.assertGreater(fast_high.rms_motor_power, 0)
        self.assertGreater(fast_high.rms_input_current(1), 0)
        self.assertGreater(fast_high.average_output_power, 0)
        self.assertGreater(fast_high.average_motor_power, 0)
        self.assertGreaterEqual(fast_high.rms_output_power, fast_high.average_output_power)
        self.assertGreaterEqual(fast_high.rms_input_power, fast_high.average_input_power)
        self.assertEqual(fast_high.rms_input_power,
                         fast_high.rms_motor_power + fast_high.rms_output_power)
        self.assertEqual(fast_high.average_input_power,
                         fast_high.average_motor_power + fast_high.average_output_power)

        self.assertLess(fast_low.rms_motor_power, stall_low.rms_motor_power)
        self.assertLess(fast_high.rms_motor_power, stall_high.rms_motor_power)
        self.assertAlmostEqual(
            numpy.sqrt(stall_high.rms_motor_power / stall_low.rms_motor_power),
            stall_high.torque / stall_low.torque)
        self.assertAlmostEqual(
            numpy.sqrt(fast_high.rms_motor_power / fast_low.rms_motor_power),
            fast_high.torque / fast_low.torque)

  def test_max_voltage(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_voltage = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.rms_input_power, 0)
        self.assertEqual(stopped.average_input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.rms_motor_power, 0)
        self.assertEqual(stopped.rms_input_current(1), 0)
        self.assertEqual(stopped.average_output_power, 0)
        self.assertEqual(stopped.rms_output_power, 0)
        self.assertEqual(stopped.average_motor_power, 0)

        '''
        TODO(Brian): Part of the TODO in MotorController.operating_point.
        fast_none = controller.operating_point(100, max_voltage = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertLess(fast_none.rms_input_power, 0)
        self.assertEqual(fast_none.average_input_power, 0)
        self.assertLess(fast_none.torque, 0)
        self.assertGreater(fast_none.rms_motor_power, 0)
        self.assertLess(fast_none.rms_input_current(1), 0)
        self.assertLess(fast_none.average_output_power, 0)
        self.assertLess(fast_none.rms_output_power, 0)
        self.assertGreater(fast_none.average_motor_power, 0)
        self.assertLessEqual(fast_none.rms_output_power, fast_none.average_output_power)
        self.assertLessEqual(fast_none.rms_input_power, fast_none.average_input_power)
        '''

        stall_low = controller.operating_point(0, max_voltage = 0.01)
        self.assertEqual(stall_low.omega, 0)
        self.assertGreater(stall_low.rms_input_power, 0)
        self.assertGreater(stall_low.average_input_power, 0)
        self.assertGreater(stall_low.torque, 0)
        self.assertGreater(stall_low.rms_motor_power, 0)
        self.assertGreater(stall_low.rms_input_current(1), 0)
        self.assertEqual(stall_low.average_output_power, 0)
        self.assertEqual(stall_low.rms_output_power, 0)
        self.assertGreater(stall_low.average_motor_power, 0)
        self.assertEqual(stall_low.rms_input_power, stall_low.rms_motor_power)
        self.assertEqual(stall_low.average_input_power, stall_low.average_motor_power)
        self.assertGreaterEqual(stall_low.rms_input_power, stall_low.average_input_power)

        stall_high = controller.operating_point(0, max_voltage = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertGreater(stall_high.rms_input_power, 0)
        self.assertGreater(stall_high.average_input_power, 0)
        self.assertGreater(stall_high.torque, 0)
        self.assertGreater(stall_high.rms_motor_power, 0)
        self.assertGreater(stall_high.rms_input_current(1), 0)
        self.assertEqual(stall_high.average_output_power, 0)
        self.assertEqual(stall_high.rms_output_power, 0)
        self.assertGreater(stall_high.average_motor_power, 0)
        self.assertEqual(stall_high.rms_input_power, stall_high.rms_motor_power)
        self.assertEqual(stall_high.average_input_power, stall_high.average_motor_power)
        self.assertGreaterEqual(stall_high.rms_input_power, stall_high.average_input_power)

        '''
        TODO(Brian): Part of the TODO in MotorController.operating_point.
        fast_low = controller.operating_point(10000, max_voltage = 0.01)
        self.assertEqual(fast_low.omega, 10000)
        self.assertLess(fast_low.rms_input_power, 0)
        self.assertLess(fast_low.average_input_power, 0)
        self.assertLess(fast_low.torque, 0)
        self.assertGreater(fast_low.rms_motor_power, 0)
        self.assertLess(fast_low.rms_input_current(1), 0)
        self.assertLess(fast_low.average_output_power, 0)
        self.assertGreater(fast_low.rms_output_power, 0)
        self.assertGreater(fast_low.average_motor_power, 0)
        self.assertLessEqual(fast_low.rms_output_power, fast_low.average_output_power)
        self.assertLessEqual(fast_low.rms_input_power, fast_low.average_input_power)
        self.assertEqual(fast_low.rms_input_power,
                         fast_low.rms_motor_power + fast_low.rms_output_power)
        self.assertEqual(fast_low.average_input_power,
                         fast_low.average_motor_power + fast_low.average_output_power)
        '''

        fast_high = controller.operating_point(100, max_voltage = 1000)
        self.assertEqual(fast_high.omega, 100)
        self.assertGreater(fast_high.rms_input_power, 0)
        self.assertGreater(fast_high.average_input_power, 0)
        self.assertGreater(fast_high.torque, 0)
        self.assertGreater(fast_high.rms_motor_power, 0)
        self.assertGreater(fast_high.rms_input_current(1), 0)
        self.assertGreater(fast_high.average_output_power, 0)
        self.assertGreater(fast_high.average_motor_power, 0)
        self.assertGreaterEqual(fast_high.rms_output_power, fast_high.average_output_power)
        self.assertGreaterEqual(fast_high.rms_input_power, fast_high.average_input_power)
        self.assertEqual(fast_high.rms_input_power,
                         fast_high.rms_motor_power + fast_high.rms_output_power)
        self.assertEqual(fast_high.average_input_power,
                         fast_high.average_motor_power + fast_high.average_output_power)

        free_1v = controller.operating_point(controller.max_speed(), max_voltage = 1)
        self.assertEqual(free_1v.omega, controller.max_speed())
        self.assertGreaterEqual(free_1v.rms_input_power, 0)
        self.assertGreaterEqual(free_1v.average_input_power, 0)
        self.assertEqual(free_1v.torque, 0)
        self.assertGreaterEqual(free_1v.rms_motor_power, 0)
        self.assertGreaterEqual(free_1v.rms_input_current(1), 0)
        self.assertEqual(free_1v.average_output_power, 0)
        self.assertEqual(free_1v.rms_output_power, 0)
        self.assertEqual(free_1v.average_motor_power, 0)
        self.assertGreaterEqual(free_1v.rms_output_power, free_1v.average_output_power)
        self.assertGreaterEqual(free_1v.rms_input_power, free_1v.average_input_power)
        self.assertEqual(free_1v.rms_input_power,
                         free_1v.rms_motor_power + free_1v.rms_output_power)
        self.assertEqual(free_1v.average_input_power,
                         free_1v.average_motor_power + free_1v.average_output_power)

        self.assertLess(fast_high.rms_motor_power, stall_high.rms_motor_power)
        self.assertAlmostEqual(
            numpy.sqrt(stall_high.rms_motor_power / stall_low.rms_motor_power),
            stall_high.torque / stall_low.torque)

if __name__ == '__main__':
  unittest.main()
