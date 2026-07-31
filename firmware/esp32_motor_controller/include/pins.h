/**
 * Pin assignment for the NeuroGrip motor-controller board (ESP32-S3).
 *
 * Change this file, not the driver code, when the board is revised. Nothing
 * else in the firmware refers to a GPIO number.
 *
 * Board revision: NG-MC-A (reference build). See docs/hardware.md.
 */

#pragma once

#include <stdint.h>

/** Servo PWM frequency. 50 Hz is the standard hobby-servo rate; digital servos
 *  accept 200–333 Hz, which reduces control latency. Verify with the datasheet
 *  before raising it — an analogue servo will overheat at 333 Hz. */
#define SERVO_PWM_HZ 50

/** Finger order matches Finger in the host: thumb, index, middle, ring, pinky.
 *  These must be PWM-capable pins on a timer group not used by the ADC. */
static const uint8_t PIN_SERVO[5] = {4, 5, 6, 7, 15};

/** INA181 shunt-amplifier outputs, one per finger. ADC1 only: ADC2 is
 *  unavailable while Wi-Fi is active. */
static const uint8_t PIN_CURRENT_SENSE[5] = {1, 2, 3, 8, 9};

/** Battery rail through a 2:1 divider. */
static const uint8_t PIN_BUS_VOLTAGE = 10;

/** Hardware emergency-stop button, active low with an internal pull-up.
 *  NOTE: this is a software-polled input, not a hardware interlock. A
 *  certifiable device needs a contactor that removes actuator power with no
 *  software in the path — see docs/safety.md, "Limits of this design". */
static const uint8_t PIN_ESTOP_BUTTON = 11;

/** Status LED: slow blink = idle, fast blink = active, solid = fault. */
static const uint8_t PIN_STATUS_LED = 48;

/** Uncomment when per-motor thermistors are fitted (not on revision A). */
// #define HAS_MOTOR_THERMISTORS
// static const uint8_t PIN_THERMISTOR[5] = {12, 13, 14, 16, 17};
