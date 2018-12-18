#!/usr/bin/python3

import unittest
import numpy

import simple
import models

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
    )

class SimpleControllerTest(unittest.TestCase):
  def test_max_torque(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_torque = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.motor_power, 0)
        self.assertEqual(stopped.input_current(1), 0)
        self.assertEqual(stopped.output_power, 0)

        fast_none = controller.operating_point(100, max_torque = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertEqual(fast_none.input_power, 0)
        self.assertEqual(fast_none.torque, 0)
        self.assertEqual(fast_none.motor_power, 0)
        self.assertEqual(fast_none.input_current(1), 0)
        self.assertEqual(fast_none.output_power, 0)

        stall_low = controller.operating_point(0, max_torque = 0.001)
        self.assertEqual(stall_low.omega, 0)
        self.assertGreater(stall_low.input_power, 0)
        self.assertAlmostEqual(stall_low.torque, 0.001)
        self.assertGreater(stall_low.motor_power, 0)
        self.assertGreater(stall_low.input_current(1), 0)
        self.assertEqual(stall_low.output_power, 0)
        self.assertEqual(stall_low.input_power, stall_low.motor_power)

        stall_high = controller.operating_point(0, max_torque = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertGreater(stall_high.input_power, 0)
        self.assertAlmostEqual(stall_high.torque, 1000)
        self.assertGreater(stall_high.motor_power, 0)
        self.assertGreater(stall_high.input_current(1), 0)
        self.assertEqual(stall_high.output_power, 0)
        self.assertEqual(stall_high.input_power, stall_high.motor_power)

        fast_low = controller.operating_point(10000, max_torque = 0.001)
        self.assertEqual(fast_low.omega, 10000)
        self.assertGreater(fast_low.input_power, 0)
        self.assertAlmostEqual(fast_low.torque, 0.001)
        self.assertGreater(fast_low.motor_power, 0)
        self.assertGreater(fast_low.input_current(1), 0)
        self.assertGreater(fast_low.output_power, 0)

        fast_high = controller.operating_point(10000, max_torque = 1000)
        self.assertEqual(fast_high.omega, 10000)
        self.assertGreater(fast_high.input_power, 0)
        self.assertAlmostEqual(fast_high.torque, 1000)
        self.assertGreater(fast_high.motor_power, 0)
        self.assertGreater(fast_high.input_current(1), 0)
        self.assertGreater(fast_high.output_power, 0)

        self.assertAlmostEqual(stall_high.motor_power,
                               stall_low.motor_power * (
                                   (stall_high.torque / stall_low.torque) ** 2))
        self.assertAlmostEqual(fast_high.motor_power,
                               fast_low.motor_power * (
                                   (fast_high.torque / fast_low.torque) ** 2))

  def test_max_motor_current(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_motor_current = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.motor_power, 0)
        self.assertEqual(stopped.input_current(1), 0)
        self.assertEqual(stopped.output_power, 0)

        fast_none = controller.operating_point(100, max_motor_current = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertEqual(fast_none.input_power, 0)
        self.assertEqual(fast_none.torque, 0)
        self.assertEqual(fast_none.motor_power, 0)
        self.assertEqual(fast_none.input_current(1), 0)
        self.assertEqual(fast_none.output_power, 0)

        stall_low = controller.operating_point(0, max_motor_current = 0.001)
        self.assertEqual(stall_low.omega, 0)
        self.assertGreater(stall_low.input_power, 0)
        self.assertGreater(stall_low.torque, 0)
        self.assertAlmostEqual(stall_low.motor_power / 3,
                               0.001**2 * controller.motor.resistance)
        self.assertGreater(stall_low.input_current(1), 0)
        self.assertEqual(stall_low.output_power, 0)
        self.assertEqual(stall_low.input_power, stall_low.motor_power)

        stall_high = controller.operating_point(0, max_motor_current = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertGreater(stall_high.input_power, 0)
        self.assertGreater(stall_high.torque, 0)
        self.assertAlmostEqual(stall_high.motor_power / 3,
                               1000**2 * controller.motor.resistance)
        self.assertGreater(stall_high.input_current(1), 0)
        self.assertEqual(stall_high.output_power, 0)
        self.assertEqual(stall_high.input_power, stall_high.motor_power)

        fast_low = controller.operating_point(10000, max_motor_current = 0.001)
        self.assertEqual(fast_low.omega, 10000)
        self.assertGreater(fast_low.input_power, 0)
        self.assertGreater(fast_low.torque, 0)
        self.assertAlmostEqual(fast_low.motor_power / 3,
                               0.001**2 * controller.motor.resistance)
        self.assertGreater(fast_low.input_current(1), 0)
        self.assertGreater(fast_low.output_power, 0)

        fast_high = controller.operating_point(10000, max_motor_current = 1000)
        self.assertEqual(fast_high.omega, 10000)
        self.assertGreater(fast_high.input_power, 0)
        self.assertGreater(fast_high.torque, 0)
        self.assertAlmostEqual(fast_high.motor_power / 3,
                               1000**2 * controller.motor.resistance)
        self.assertGreater(fast_high.input_current(1), 0)
        self.assertGreater(fast_high.output_power, 0)

        self.assertAlmostEqual(stall_high.motor_power,
                               stall_low.motor_power * (
                                   (stall_high.torque / stall_low.torque) ** 2))
        self.assertAlmostEqual(fast_high.motor_power,
                               fast_low.motor_power * (
                                   (fast_high.torque / fast_low.torque) ** 2))

  def test_max_input_power(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_input_power = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.motor_power, 0)
        self.assertEqual(stopped.input_current(1), 0)
        self.assertEqual(stopped.output_power, 0)

        fast_none = controller.operating_point(100, max_input_power = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertEqual(fast_none.input_power, 0)
        self.assertEqual(fast_none.torque, 0)
        self.assertEqual(fast_none.motor_power, 0)
        self.assertEqual(fast_none.input_current(1), 0)
        self.assertEqual(fast_none.output_power, 0)

        stall_low = controller.operating_point(0, max_input_power = 0.001)
        self.assertEqual(stall_low.omega, 0)
        self.assertAlmostEqual(stall_low.input_power, 0.001)
        self.assertGreater(stall_low.torque, 0)
        self.assertAlmostEqual(stall_low.motor_power, 0.001)
        self.assertGreater(stall_low.input_current(1), 0)
        self.assertEqual(stall_low.output_power, 0)
        self.assertEqual(stall_low.input_power, stall_low.motor_power)

        stall_high = controller.operating_point(0, max_input_power = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertAlmostEqual(stall_high.input_power, 1000)
        self.assertGreater(stall_high.torque, 0)
        self.assertAlmostEqual(stall_high.motor_power, 1000)
        self.assertGreater(stall_high.input_current(1), 0)
        self.assertEqual(stall_high.output_power, 0)

        fast_low = controller.operating_point(10000, max_input_power = 0.001)
        self.assertEqual(fast_low.omega, 10000)
        self.assertAlmostEqual(fast_low.input_power, 0.001)
        self.assertGreater(fast_low.torque, 0)
        self.assertGreater(fast_low.motor_power, 0)
        self.assertGreater(fast_low.input_current(1), 0)
        self.assertGreater(fast_low.output_power, 0)

        fast_high = controller.operating_point(10000, max_input_power = 1000)
        self.assertEqual(fast_high.omega, 10000)
        self.assertAlmostEqual(fast_high.input_power, 1000)
        self.assertGreater(fast_high.torque, 0)
        self.assertGreater(fast_high.motor_power, 0)
        self.assertGreater(fast_high.input_current(1), 0)
        self.assertGreater(fast_high.output_power, 0)

        self.assertLess(fast_low.motor_power, stall_low.motor_power)
        self.assertLess(fast_high.motor_power, stall_high.motor_power)
        self.assertAlmostEqual(
            numpy.sqrt(stall_high.motor_power / stall_low.motor_power),
            stall_high.torque / stall_low.torque)
        self.assertAlmostEqual(
            numpy.sqrt(fast_high.motor_power / fast_low.motor_power),
            fast_high.torque / fast_low.torque)

  def test_max_voltage(self):
    for controller in _CONTROLLERS:
      with self.subTest(controller=controller):
        stopped = controller.operating_point(0, max_voltage = 0)
        self.assertEqual(stopped.omega, 0)
        self.assertEqual(stopped.input_power, 0)
        self.assertEqual(stopped.torque, 0)
        self.assertEqual(stopped.motor_power, 0)
        self.assertEqual(stopped.input_current(1), 0)
        self.assertEqual(stopped.output_power, 0)

        '''
        TODO(Brian): Part of the TODO in MotorController.operating_point.
        fast_none = controller.operating_point(100, max_voltage = 0)
        self.assertEqual(fast_none.omega, 100)
        self.assertLess(fast_none.input_power, 0)
        self.assertLess(fast_none.torque, 0)
        self.assertGreater(fast_none.motor_power, 0)
        self.assertLess(fast_none.input_current(1), 0)
        self.assertLess(fast_none.output_power, 0)
        '''

        stall_low = controller.operating_point(0, max_voltage = 0.01)
        self.assertEqual(stall_low.omega, 0)
        self.assertGreater(stall_low.input_power, 0)
        self.assertGreater(stall_low.torque, 0)
        self.assertGreater(stall_low.motor_power, 0)
        self.assertGreater(stall_low.input_current(1), 0)
        self.assertEqual(stall_low.output_power, 0)
        self.assertEqual(stall_low.input_power, stall_low.motor_power)

        stall_high = controller.operating_point(0, max_voltage = 1000)
        self.assertEqual(stall_high.omega, 0)
        self.assertGreater(stall_high.input_power, 0)
        self.assertGreater(stall_high.torque, 0)
        self.assertGreater(stall_high.motor_power, 0)
        self.assertGreater(stall_high.input_current(1), 0)
        self.assertEqual(stall_high.output_power, 0)

        '''
        TODO(Brian): Part of the TODO in MotorController.operating_point.
        fast_low = controller.operating_point(10000, max_voltage = 0.01)
        self.assertEqual(fast_low.omega, 10000)
        self.assertLess(fast_low.input_power, 0)
        self.assertLess(fast_low.torque, 0)
        self.assertGreater(fast_low.motor_power, 0)
        self.assertLess(fast_low.input_current(1), 0)
        self.assertLess(fast_low.output_power, 0)
        '''

        fast_high = controller.operating_point(100, max_voltage = 1000)
        self.assertEqual(fast_high.omega, 100)
        self.assertGreater(fast_high.input_power, 0)
        self.assertGreater(fast_high.torque, 0)
        self.assertGreater(fast_high.motor_power, 0)
        self.assertGreater(fast_high.input_current(1), 0)
        self.assertGreater(fast_high.output_power, 0)

        free_1v = controller.operating_point(controller.max_speed(), max_voltage = 1)
        self.assertEqual(free_1v.omega, controller.max_speed())
        self.assertGreaterEqual(free_1v.input_power, 0)
        self.assertEqual(free_1v.torque, 0)
        self.assertGreaterEqual(free_1v.motor_power, 0)
        self.assertGreaterEqual(free_1v.input_current(1), 0)
        self.assertEqual(free_1v.output_power, 0)

        self.assertLess(fast_high.motor_power, stall_high.motor_power)
        self.assertAlmostEqual(
            numpy.sqrt(stall_high.motor_power / stall_low.motor_power),
            stall_high.torque / stall_low.torque)

if __name__ == '__main__':
  unittest.main()
