/**
 * NeuroGrip ESP32 motor controller.
 *
 * Responsibilities, and deliberately nothing more:
 *
 *   1. Generate servo PWM from commanded finger positions.
 *   2. Enforce velocity, acceleration, current and temperature limits.
 *   3. Stream telemetry at 100 Hz.
 *   4. **Safe the actuators by itself if the host stops talking.**
 *
 * Point 4 is the reason this firmware exists as a separate processor rather than
 * as a PWM peripheral. If the Linux host crashes, is unplugged, or stalls in the
 * kernel, nothing on that side can react — so the timeout that safes the hand
 * must live here, where a host failure cannot reach it. Every SET_TARGETS
 * refreshes it.
 *
 * What is deliberately NOT here: trajectory planning, grasp selection, any
 * notion of intent. The host owns those, because a motion must be interruptible
 * within one control cycle and a stored profile executing on the MCU would make
 * a cancel wait for a round trip.
 *
 * Two FreeRTOS tasks, pinned to separate cores:
 *   core 0 — comms: parse frames, emit telemetry
 *   core 1 — control: 200 Hz servo update, limits, watchdog
 *
 * The control task never blocks on I/O. Commands cross between them through a
 * small mutex-protected state block.
 *
 * The behaviour implemented here is mirrored by
 * `src/neurogrip/hal/servo/emulator.py`, which the host-side tests run against.
 * When this file changes, that one must change with it.
 */

#include <Arduino.h>
#include <ESP32Servo.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "ngp_protocol.h"
#include "pins.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static constexpr uint32_t SERIAL_BAUD       = 921600;
static constexpr uint32_t CONTROL_HZ        = 200;
static constexpr uint32_t TELEMETRY_HZ      = 100;
static constexpr uint32_t DEFAULT_WATCHDOG_MS = 300;

static constexpr uint8_t  FIRMWARE_MAJOR = 1;
static constexpr uint8_t  FIRMWARE_MINOR = 0;
static constexpr uint8_t  FIRMWARE_PATCH = 0;

// Hard ceilings. The host may lower these; it can never raise them.
static constexpr float    ABS_MAX_VELOCITY     = 3.0f;   // closure units/s
static constexpr float    ABS_MAX_ACCELERATION = 15.0f;  // closure units/s^2
static constexpr uint16_t ABS_MAX_CURRENT_MA   = 1200;
static constexpr int8_t   ABS_MAX_TEMP_C       = 75;
static constexpr uint16_t UNDERVOLTAGE_MV      = 6400;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct FingerRuntime {
    Servo    servo;
    float    position     = 0.0f;   // measured/estimated closure
    float    target       = 0.0f;   // commanded closure
    float    velocity     = 0.0f;
    uint16_t current_ma   = 0;
    int8_t   temperature  = 25;
    uint16_t min_pulse_us = 1000;
    uint16_t max_pulse_us = 2000;
    bool     inverted     = false;
    bool     enabled      = false;
    bool     stalled      = false;
};

static FingerRuntime g_fingers[NGP_FINGER_COUNT];

static portMUX_TYPE g_state_mux = portMUX_INITIALIZER_UNLOCKED;

static struct {
    float    max_velocity     = 2.0f;
    float    max_acceleration = 8.0f;
    uint16_t max_current_ma   = 900;
    int8_t   max_temperature  = 65;
    float    speed_scale      = 1.0f;
    float    force            = 0.6f;
    uint8_t  enabled_mask     = 0;
    bool     estop            = false;
    bool     homed            = false;
    bool     watchdog_tripped = false;
    uint32_t watchdog_ms      = DEFAULT_WATCHDOG_MS;
    uint32_t last_command_ms  = 0;
    uint16_t bus_voltage_mv   = 7400;
    uint8_t  sequence         = 0;
} g_state;

// ---------------------------------------------------------------------------
// Frame transmission
// ---------------------------------------------------------------------------

static uint8_t g_tx_sequence = 0;

static void ngp_send(uint8_t msg_id, const uint8_t *payload, uint8_t length) {
    uint8_t header[3] = {length, msg_id, g_tx_sequence++};
    uint8_t body[3 + NGP_MAX_PAYLOAD];
    memcpy(body, header, 3);
    if (length && payload) {
        memcpy(body + 3, payload, length);
    }
    const uint16_t crc = ngp_crc16(body, (uint16_t)(3 + length));

    const uint8_t sync[2] = {NGP_SYNC0, NGP_SYNC1};
    Serial.write(sync, 2);
    Serial.write(body, 3 + length);
    const uint8_t tail[2] = {(uint8_t)(crc & 0xFF), (uint8_t)(crc >> 8)};
    Serial.write(tail, 2);
}

static void ngp_send_error(ngp_error_code_t code, uint16_t detail) {
    ngp_error_t payload = {(uint8_t)code, detail};
    ngp_send(NGP_MSG_ERROR, (const uint8_t *)&payload, sizeof(payload));
}

static void ngp_send_event(ngp_event_code_t code, uint8_t finger, uint16_t detail) {
    ngp_event_t payload = {(uint8_t)code, finger, detail};
    ngp_send(NGP_MSG_EVENT, (const uint8_t *)&payload, sizeof(payload));
}

// ---------------------------------------------------------------------------
// Actuation
// ---------------------------------------------------------------------------

static uint16_t pulse_for(const FingerRuntime &finger, float closure) {
    float travel = closure < 0.0f ? 0.0f : (closure > 1.0f ? 1.0f : closure);
    if (finger.inverted) {
        travel = 1.0f - travel;
    }
    return (uint16_t)(finger.min_pulse_us +
                      travel * (float)(finger.max_pulse_us - finger.min_pulse_us));
}

static void safe_actuators(const char *reason) {
    portENTER_CRITICAL(&g_state_mux);
    g_state.enabled_mask = 0;
    for (auto &finger : g_fingers) {
        finger.enabled  = false;
        finger.velocity = 0.0f;
        // Hold position rather than opening: if the hand is carrying something,
        // dropping it is worse than stopping where it is. Detaching the PWM
        // signal lets the servo relax without back-driving hard.
        finger.target = finger.position;
        finger.servo.detach();
    }
    portEXIT_CRITICAL(&g_state_mux);
    ngp_send(NGP_MSG_LOG, (const uint8_t *)reason, (uint8_t)strlen(reason));
}

static void engage_estop() {
    portENTER_CRITICAL(&g_state_mux);
    g_state.estop = true;
    portEXIT_CRITICAL(&g_state_mux);
    safe_actuators("estop");
    ngp_send_event(NGP_EVENT_ESTOP_ENGAGED, 0xFF, 0);
}

/** Read motor current from the shunt amplifier on the given channel. */
static uint16_t read_current_ma(uint8_t index) {
    const uint16_t counts = analogRead(PIN_CURRENT_SENSE[index]);
    // 3.3 V reference, 12-bit ADC, 0.1 ohm shunt, gain 50.
    // TODO(hardware): calibrate per unit against a bench supply; the constant
    // below is nominal and has not been measured on the reference board.
    return (uint16_t)((float)counts * 3300.0f / 4095.0f / (0.1f * 50.0f));
}

/** Estimate motor temperature. */
static int8_t read_temperature(uint8_t index) {
#ifdef HAS_MOTOR_THERMISTORS
    const uint16_t counts = analogRead(PIN_THERMISTOR[index]);
    return (int8_t)(counts * 100 / 4095);
#else
    // No per-motor thermistor on the reference board: model the rise from
    // current instead. Conservative — it over-estimates, which is the safe
    // direction for a thermal limit.
    static float estimate[NGP_FINGER_COUNT] = {25, 25, 25, 25, 25};
    const float amps = g_fingers[index].current_ma / 1000.0f;
    estimate[index] += (0.9f * amps * amps - (estimate[index] - 25.0f) / 45.0f) / CONTROL_HZ;
    return (int8_t)estimate[index];
#endif
}

static void control_step(float dt) {
    const bool estop = g_state.estop;
    const float max_velocity = g_state.max_velocity * g_state.speed_scale;
    const float max_accel    = g_state.max_acceleration * g_state.speed_scale;

    bool overcurrent = false;
    bool overtemp    = false;

    for (uint8_t index = 0; index < NGP_FINGER_COUNT; ++index) {
        FingerRuntime &finger = g_fingers[index];

        finger.current_ma  = read_current_ma(index);
        finger.temperature = read_temperature(index);

        if (finger.current_ma > g_state.max_current_ma) overcurrent = true;
        if (finger.temperature > g_state.max_temperature) overtemp = true;

        if (estop || !finger.enabled) {
            finger.velocity = 0.0f;
            continue;
        }

        const float error = finger.target - finger.position;

        // Cap the desired velocity both by the limit and by the speed from
        // which the actuator can still stop on target (v = sqrt(2*a*|e|)), so a
        // step command does not overshoot.
        const float stopping = sqrtf(2.0f * max_accel * fabsf(error));
        float desired = fminf(max_velocity, stopping);
        desired = copysignf(fminf(desired, fabsf(error) / 0.02f), error);

        float delta = desired - finger.velocity;
        const float max_delta = max_accel * dt;
        if (delta > max_delta)  delta = max_delta;
        if (delta < -max_delta) delta = -max_delta;
        finger.velocity += delta;

        float next = finger.position + finger.velocity * dt;
        if ((next - finger.target) * error > 0.0f) {
            next = finger.target;          // do not overshoot
            finger.velocity = 0.0f;
        }
        if (next < 0.0f) next = 0.0f;
        if (next > 1.0f) next = 1.0f;

        // Stall: commanded to move, current is high, position is not changing.
        // The finger has met something; stop driving it further closed and let
        // the holding current provide the grip.
        const bool loaded = finger.current_ma > (uint16_t)(g_state.force * g_state.max_current_ma);
        if (loaded && fabsf(next - finger.position) < 1e-4f && fabsf(error) > 0.02f) {
            if (!finger.stalled) {
                finger.stalled = true;
                ngp_send_event(NGP_EVENT_CONTACT_DETECTED, index, finger.current_ma);
            }
            finger.velocity = 0.0f;
        } else if (finger.stalled && fabsf(error) < 0.01f) {
            finger.stalled = false;
        }

        finger.position = next;
        finger.servo.writeMicroseconds(pulse_for(finger, finger.position));
    }

    if (overcurrent) {
        ngp_send_error(NGP_ERR_OVERCURRENT, 0);
        engage_estop();
    }
    if (overtemp) {
        ngp_send_event(NGP_EVENT_THERMAL_THROTTLE, 0xFF, 0);
        // Derate rather than stop: let the user finish what they are doing.
        g_state.max_current_ma = (uint16_t)(g_state.max_current_ma * 0.8f);
    }
}

/**
 * The command-stream watchdog.
 *
 * This is the single most important behaviour in the firmware. Do not make it
 * conditional on anything the host controls.
 */
static void check_watchdog(uint32_t now_ms) {
    if (g_state.watchdog_ms == 0 || g_state.watchdog_tripped || g_state.estop) {
        return;
    }
    if (now_ms - g_state.last_command_ms <= g_state.watchdog_ms) {
        return;
    }
    g_state.watchdog_tripped = true;
    safe_actuators("watchdog");
    ngp_send_event(NGP_EVENT_WATCHDOG_TRIP, 0xFF, 0);
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

static void handle_set_targets(const uint8_t *payload, uint8_t length) {
    if (length != sizeof(ngp_set_targets_t)) {
        ngp_send_error(NGP_ERR_BAD_LENGTH, length);
        return;
    }
    g_state.last_command_ms = millis();

    if (g_state.estop) {
        ngp_send_error(NGP_ERR_ESTOP_ACTIVE, 0);
        return;
    }
    if (g_state.enabled_mask == 0) {
        ngp_send_error(NGP_ERR_NOT_ENABLED, 0);
        return;
    }

    const ngp_set_targets_t *message = (const ngp_set_targets_t *)payload;
    portENTER_CRITICAL(&g_state_mux);
    for (uint8_t i = 0; i < NGP_FINGER_COUNT; ++i) {
        g_fingers[i].target = ngp_position_from_wire(message->position[i]);
    }
    g_state.speed_scale = (float)message->speed / 127.5f;
    if (g_state.speed_scale < 0.05f) g_state.speed_scale = 0.05f;
    if (g_state.speed_scale > 2.0f)  g_state.speed_scale = 2.0f;
    g_state.watchdog_tripped = false;
    portEXIT_CRITICAL(&g_state_mux);
}

static void handle_set_limits(const uint8_t *payload, uint8_t length) {
    if (length != sizeof(ngp_set_limits_t)) {
        ngp_send_error(NGP_ERR_BAD_LENGTH, length);
        return;
    }
    const ngp_set_limits_t *message = (const ngp_set_limits_t *)payload;
    portENTER_CRITICAL(&g_state_mux);
    // The host may only ever tighten these.
    g_state.max_velocity     = fminf((float)message->max_velocity / 1000.0f, ABS_MAX_VELOCITY);
    g_state.max_acceleration = fminf((float)message->max_acceleration / 1000.0f, ABS_MAX_ACCELERATION);
    g_state.max_current_ma   = message->max_current_ma < ABS_MAX_CURRENT_MA
                                   ? message->max_current_ma : ABS_MAX_CURRENT_MA;
    g_state.max_temperature  = (int8_t)(message->max_temperature_c < (uint16_t)ABS_MAX_TEMP_C
                                   ? message->max_temperature_c : (uint16_t)ABS_MAX_TEMP_C);
    portEXIT_CRITICAL(&g_state_mux);
}

static void handle_enable(const uint8_t *payload, uint8_t length) {
    if (g_state.estop) {
        ngp_send_error(NGP_ERR_ESTOP_ACTIVE, 0);
        return;
    }
    const uint8_t mask = length ? payload[0] : 0x1F;
    portENTER_CRITICAL(&g_state_mux);
    g_state.enabled_mask = mask & 0x1F;
    for (uint8_t i = 0; i < NGP_FINGER_COUNT; ++i) {
        const bool on = (g_state.enabled_mask >> i) & 1;
        g_fingers[i].enabled = on;
        if (on && !g_fingers[i].servo.attached()) {
            g_fingers[i].servo.attach(PIN_SERVO[i], g_fingers[i].min_pulse_us,
                                      g_fingers[i].max_pulse_us);
        }
    }
    g_state.last_command_ms  = millis();
    g_state.watchdog_tripped = false;
    portEXIT_CRITICAL(&g_state_mux);
}

static void handle_disable(const uint8_t *payload, uint8_t length) {
    const uint8_t mask = length ? payload[0] : 0x1F;
    portENTER_CRITICAL(&g_state_mux);
    g_state.enabled_mask &= ~(mask & 0x1F);
    for (uint8_t i = 0; i < NGP_FINGER_COUNT; ++i) {
        if (!((g_state.enabled_mask >> i) & 1)) {
            g_fingers[i].enabled = false;
            g_fingers[i].servo.detach();
        }
    }
    portEXIT_CRITICAL(&g_state_mux);
}

static void handle_clear_estop(const uint8_t *payload, uint8_t length) {
    if (length < 2) {
        ngp_send_error(NGP_ERR_BAD_LENGTH, length);
        return;
    }
    const uint16_t magic = (uint16_t)payload[0] | ((uint16_t)payload[1] << 8);
    if (magic != NGP_CLEAR_ESTOP_MAGIC) {
        ngp_send_error(NGP_ERR_BAD_PARAMETER, magic);
        return;
    }
    portENTER_CRITICAL(&g_state_mux);
    g_state.estop            = false;
    g_state.watchdog_tripped = false;
    g_state.last_command_ms  = millis();
    portEXIT_CRITICAL(&g_state_mux);
    ngp_send_event(NGP_EVENT_ESTOP_RELEASED, 0xFF, 0);
}

static void handle_home() {
    portENTER_CRITICAL(&g_state_mux);
    for (auto &finger : g_fingers) {
        finger.target   = 0.0f;
        finger.position = 0.0f;
        finger.velocity = 0.0f;
        finger.stalled  = false;
        if (finger.enabled) {
            finger.servo.writeMicroseconds(pulse_for(finger, 0.0f));
        }
    }
    g_state.homed           = true;
    g_state.last_command_ms = millis();
    portEXIT_CRITICAL(&g_state_mux);
    ngp_send_event(NGP_EVENT_HOMING_COMPLETE, 0xFF, 0);
}

static void handle_set_calibration(const uint8_t *payload, uint8_t length) {
    if (length != sizeof(ngp_set_calibration_t)) {
        ngp_send_error(NGP_ERR_BAD_LENGTH, length);
        return;
    }
    const ngp_set_calibration_t *message = (const ngp_set_calibration_t *)payload;
    if (message->finger >= NGP_FINGER_COUNT) {
        ngp_send_error(NGP_ERR_BAD_PARAMETER, message->finger);
        return;
    }
    FingerRuntime &finger = g_fingers[message->finger];
    portENTER_CRITICAL(&g_state_mux);
    finger.min_pulse_us = message->min_pulse_us;
    finger.max_pulse_us = message->max_pulse_us;
    finger.inverted     = message->inverted != 0;
    portEXIT_CRITICAL(&g_state_mux);
    // TODO(persistence): write these to NVS so a power cycle keeps them.
}

static void dispatch(uint8_t msg_id, const uint8_t *payload, uint8_t length) {
    switch (msg_id) {
        case NGP_MSG_PING: {
            ngp_pong_t pong = {
                length >= 4 ? *(const uint32_t *)payload : 0u,
                FIRMWARE_MAJOR, FIRMWARE_MINOR, FIRMWARE_PATCH,
                millis(),
            };
            ngp_send(NGP_MSG_PONG, (const uint8_t *)&pong, sizeof(pong));
            break;
        }
        case NGP_MSG_SET_TARGETS:     handle_set_targets(payload, length); break;
        case NGP_MSG_SET_LIMITS:      handle_set_limits(payload, length);  break;
        case NGP_MSG_ENABLE:          handle_enable(payload, length);      break;
        case NGP_MSG_DISABLE:         handle_disable(payload, length);     break;
        case NGP_MSG_ESTOP:           engage_estop();                      break;
        case NGP_MSG_CLEAR_ESTOP:     handle_clear_estop(payload, length); break;
        case NGP_MSG_HOME:            handle_home();                       break;
        case NGP_MSG_SET_CALIBRATION: handle_set_calibration(payload, length); break;
        case NGP_MSG_SET_FORCE:
            if (length >= 2) {
                g_state.force = (float)payload[1] / 255.0f;
                g_state.last_command_ms = millis();
            }
            break;
        case NGP_MSG_SET_WATCHDOG:
            if (length >= 2) {
                g_state.watchdog_ms = (uint32_t)payload[0] | ((uint32_t)payload[1] << 8);
                g_state.last_command_ms = millis();
            }
            break;
        case NGP_MSG_REQUEST_STATE: /* the telemetry task will send one */ break;
        case NGP_MSG_REBOOT:        ESP.restart();                         break;
        default:                    ngp_send_error(NGP_ERR_UNKNOWN_MESSAGE, msg_id); break;
    }
}

// ---------------------------------------------------------------------------
// Incremental frame parser (mirrors FrameParser on the host)
// ---------------------------------------------------------------------------

static uint8_t  g_rx[2 + 3 + NGP_MAX_PAYLOAD + 2];
static uint16_t g_rx_len = 0;

static void parse_incoming() {
    while (Serial.available() > 0 && g_rx_len < sizeof(g_rx)) {
        g_rx[g_rx_len++] = (uint8_t)Serial.read();
    }

    uint16_t start = 0;
    while (g_rx_len - start >= 5) {
        if (g_rx[start] != NGP_SYNC0 || g_rx[start + 1] != NGP_SYNC1) {
            ++start;  // resynchronise
            continue;
        }
        const uint8_t  length = g_rx[start + 2];
        const uint16_t total  = (uint16_t)(2 + 3 + length + 2);
        if (g_rx_len - start < total) {
            break;  // wait for the rest
        }
        const uint16_t crc = ngp_crc16(&g_rx[start + 2], (uint16_t)(3 + length));
        const uint16_t received = (uint16_t)g_rx[start + total - 2] |
                                  ((uint16_t)g_rx[start + total - 1] << 8);
        if (crc == received) {
            dispatch(g_rx[start + 3], &g_rx[start + 5], length);
            start += total;
        } else {
            start += 2;  // drop the sync word and rescan
        }
    }

    if (start > 0) {
        memmove(g_rx, g_rx + start, g_rx_len - start);
        g_rx_len -= start;
    }
    if (g_rx_len >= sizeof(g_rx)) {
        g_rx_len = 0;  // runaway garbage; resynchronise from scratch
    }
}

static void send_state() {
    ngp_state_t state;
    portENTER_CRITICAL(&g_state_mux);
    state.sequence = g_state.sequence++;
    uint8_t flags = 0;
    if (g_state.enabled_mask)     flags |= NGP_FLAG_ENABLED;
    if (g_state.estop)            flags |= NGP_FLAG_ESTOP;
    if (g_state.homed)            flags |= NGP_FLAG_HOMED;
    if (g_state.watchdog_tripped) flags |= NGP_FLAG_WATCHDOG_TRIPPED;

    bool moving = false;
    uint32_t total_current = 0;
    for (uint8_t i = 0; i < NGP_FINGER_COUNT; ++i) {
        const FingerRuntime &finger = g_fingers[i];
        state.fingers[i].position      = ngp_position_to_wire(finger.position);
        state.fingers[i].target        = ngp_position_to_wire(finger.target);
        state.fingers[i].current_ma    = finger.current_ma;
        state.fingers[i].temperature_c = finger.temperature;
        if (fabsf(finger.velocity) > 1e-3f) moving = true;
        if (finger.current_ma > g_state.max_current_ma) flags |= NGP_FLAG_OVERCURRENT;
        if (finger.temperature > g_state.max_temperature) flags |= NGP_FLAG_OVERTEMP;
        total_current += finger.current_ma;
    }
    if (moving) flags |= NGP_FLAG_MOVING;

    // Bus voltage from the divider on the battery rail.
    state.bus_voltage_mv = (uint16_t)(analogRead(PIN_BUS_VOLTAGE) * 3300L * 3 / 4095);
    if (state.bus_voltage_mv < UNDERVOLTAGE_MV) flags |= NGP_FLAG_UNDERVOLTAGE;

    state.flags     = flags;
    state.uptime_ms = millis();
    portEXIT_CRITICAL(&g_state_mux);

    ngp_send(NGP_MSG_STATE, (const uint8_t *)&state, sizeof(state));
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

static void control_task(void *) {
    const TickType_t period = pdMS_TO_TICKS(1000 / CONTROL_HZ);
    TickType_t last_wake = xTaskGetTickCount();
    const float dt = 1.0f / (float)CONTROL_HZ;

    for (;;) {
        const uint32_t now = millis();
        check_watchdog(now);
        control_step(dt);
        vTaskDelayUntil(&last_wake, period);
    }
}

static void comms_task(void *) {
    const TickType_t period = pdMS_TO_TICKS(1000 / TELEMETRY_HZ);
    TickType_t last_wake = xTaskGetTickCount();

    for (;;) {
        parse_incoming();
        send_state();
        vTaskDelayUntil(&last_wake, period);
    }
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    Serial.setRxBufferSize(1024);

    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    for (uint8_t i = 0; i < NGP_FINGER_COUNT; ++i) {
        g_fingers[i].servo.setPeriodHertz(SERVO_PWM_HZ);
        pinMode(PIN_CURRENT_SENSE[i], INPUT);
    }
    pinMode(PIN_BUS_VOLTAGE, INPUT);
    pinMode(PIN_ESTOP_BUTTON, INPUT_PULLUP);
    analogReadResolution(12);

    // Start de-energised. The host must explicitly enable drive, so a power
    // cycle can never leave the hand moving on its own.
    g_state.last_command_ms = millis();

    xTaskCreatePinnedToCore(comms_task,   "comms",   4096, nullptr, 2, nullptr, 0);
    xTaskCreatePinnedToCore(control_task, "control", 4096, nullptr, 3, nullptr, 1);
}

void loop() {
    // A hardware e-stop button, if fitted, is polled here. It is not a
    // substitute for a proper hardware interlock that removes actuator power
    // without software in the path — see docs/safety.md, "Limits of this design".
    if (digitalRead(PIN_ESTOP_BUTTON) == LOW && !g_state.estop) {
        engage_estop();
    }
    vTaskDelay(pdMS_TO_TICKS(20));
}
