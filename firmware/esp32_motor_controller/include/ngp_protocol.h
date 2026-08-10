/**
 * NeuroGrip Protocol (NGP) v1 — wire format shared with the host.
 *
 * This header MUST stay byte-for-byte compatible with
 * `src/neurogrip/hal/protocol.py`. The Python side is the reference; when the
 * two disagree, the Python unit tests in `tests/unit/test_hal_and_protocol.py`
 * are the arbiter. See `docs/protocol.md`.
 *
 * Frame layout (little-endian throughout):
 *
 *   +------+------+--------+--------+-----+---------+--------+
 *   | 0xA5 | 0x5A | LENGTH | MSG_ID | SEQ | PAYLOAD | CRC16  |
 *   +------+------+--------+--------+-----+---------+--------+
 *     sync   sync    u8       u8      u8   LENGTH B    u16
 *
 * LENGTH counts payload bytes only. The CRC (CCITT-FALSE, init 0xFFFF) covers
 * LENGTH, MSG_ID, SEQ and PAYLOAD — everything except the sync pattern.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NGP_PROTOCOL_VERSION 1
#define NGP_SYNC0 0xA5
#define NGP_SYNC1 0x5A
#define NGP_MAX_PAYLOAD 255
#define NGP_FINGER_COUNT 5

/** Finger positions travel as uint16 in units of 1/NGP_POSITION_SCALE. */
#define NGP_POSITION_SCALE 10000

/** Magic word required to clear the emergency stop, so a corrupted frame
 *  cannot re-enable drive by accident. */
#define NGP_CLEAR_ESTOP_MAGIC 0x5EA1

/* -- Message identifiers. Host->MCU are < 0x80, MCU->host are >= 0x80. ----- */
typedef enum {
    NGP_MSG_PING            = 0x01,
    NGP_MSG_SET_TARGETS     = 0x02,
    NGP_MSG_SET_LIMITS      = 0x03,
    NGP_MSG_ENABLE          = 0x04,
    NGP_MSG_DISABLE         = 0x05,
    NGP_MSG_ESTOP           = 0x06,
    NGP_MSG_CLEAR_ESTOP     = 0x07,
    NGP_MSG_REQUEST_STATE   = 0x08,
    NGP_MSG_SET_CALIBRATION = 0x09,
    NGP_MSG_HOME            = 0x0A,
    NGP_MSG_SET_FORCE       = 0x0B,
    NGP_MSG_SET_WATCHDOG    = 0x0C,
    NGP_MSG_REBOOT          = 0x0D,

    NGP_MSG_PONG            = 0x81,
    NGP_MSG_STATE           = 0x82,
    NGP_MSG_EVENT           = 0x83,
    NGP_MSG_ERROR           = 0x84,
    NGP_MSG_LOG             = 0x85,
} ngp_message_id_t;

/* -- Status flags in the STATE message ------------------------------------ */
typedef enum {
    NGP_FLAG_NONE             = 0,
    NGP_FLAG_ENABLED          = 1 << 0,
    NGP_FLAG_MOVING           = 1 << 1,
    NGP_FLAG_ESTOP            = 1 << 2,
    NGP_FLAG_HOMED            = 1 << 3,
    NGP_FLAG_OVERCURRENT      = 1 << 4,
    NGP_FLAG_OVERTEMP         = 1 << 5,
    NGP_FLAG_WATCHDOG_TRIPPED = 1 << 6,
    NGP_FLAG_UNDERVOLTAGE     = 1 << 7,
} ngp_status_flag_t;

/* -- Asynchronous events -------------------------------------------------- */
typedef enum {
    NGP_EVENT_STALL_DETECTED   = 1,
    NGP_EVENT_TARGET_REACHED   = 2,
    NGP_EVENT_CONTACT_DETECTED = 3,
    NGP_EVENT_HOMING_COMPLETE  = 4,
    NGP_EVENT_ESTOP_ENGAGED    = 5,
    NGP_EVENT_ESTOP_RELEASED   = 6,
    NGP_EVENT_WATCHDOG_TRIP    = 7,
    NGP_EVENT_THERMAL_THROTTLE = 8,
} ngp_event_code_t;

/* -- Error codes ---------------------------------------------------------- */
typedef enum {
    NGP_ERR_NONE            = 0,
    NGP_ERR_UNKNOWN_MESSAGE = 1,
    NGP_ERR_BAD_LENGTH      = 2,
    NGP_ERR_BAD_PARAMETER   = 3,
    NGP_ERR_NOT_ENABLED     = 4,
    NGP_ERR_ESTOP_ACTIVE    = 5,
    NGP_ERR_NOT_HOMED       = 6,
    NGP_ERR_OVERCURRENT     = 7,
    NGP_ERR_OVERTEMP        = 8,
    NGP_ERR_UNDERVOLTAGE    = 9,
    NGP_ERR_HARDWARE_FAULT  = 10,
} ngp_error_code_t;

/* -- Payload layouts ------------------------------------------------------
 * Packed so sizeof() matches the wire size exactly. The ESP32 (Xtensa LX7) is
 * little-endian, as is the host, so no byte swapping is required.
 */

typedef struct __attribute__((packed)) {
    uint16_t position[NGP_FINGER_COUNT]; /**< 0..NGP_POSITION_SCALE           */
    uint8_t  speed;                      /**< speed_scale * 127.5             */
    uint8_t  flags;                      /**< reserved, must be zero          */
} ngp_set_targets_t;

typedef struct __attribute__((packed)) {
    uint16_t max_velocity;     /**< closure units/s   × 1000 */
    uint16_t max_acceleration; /**< closure units/s^2 × 1000 */
    uint16_t max_current_ma;
    uint16_t max_temperature_c;
} ngp_set_limits_t;

typedef struct __attribute__((packed)) {
    uint8_t  finger;
    uint16_t min_pulse_us;
    uint16_t max_pulse_us;
    uint8_t  inverted;
    uint8_t  slack; /**< tendon take-up, closure units × 255 */
} ngp_set_calibration_t;

typedef struct __attribute__((packed)) {
    uint16_t position;
    uint16_t target;
    uint16_t current_ma;
    int8_t   temperature_c;
} ngp_finger_state_t;

typedef struct __attribute__((packed)) {
    uint8_t            sequence;
    uint8_t            flags;
    ngp_finger_state_t fingers[NGP_FINGER_COUNT];
    uint16_t           bus_voltage_mv;
    uint32_t           uptime_ms;
} ngp_state_t;

typedef struct __attribute__((packed)) {
    uint32_t token;
    uint8_t  version_major;
    uint8_t  version_minor;
    uint8_t  version_patch;
    uint32_t uptime_ms;
} ngp_pong_t;

typedef struct __attribute__((packed)) {
    uint8_t  code;
    uint8_t  finger; /**< 0xFF means "all fingers" */
    uint16_t detail;
} ngp_event_t;

typedef struct __attribute__((packed)) {
    uint8_t  code;
    uint16_t detail;
} ngp_error_t;

/* -- CRC ------------------------------------------------------------------ */

/**
 * CRC-16/CCITT-FALSE. Bitwise rather than table-driven: frames are at most ~40
 * bytes at 200 Hz, so the cost is negligible, and the host uses the identical
 * loop, which makes the two implementations trivially comparable during
 * bring-up. CRC of "123456789" is 0x29B1.
 */
static inline uint16_t ngp_crc16(const uint8_t *data, uint16_t length) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < length; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

static inline uint16_t ngp_position_to_wire(float closure) {
    if (closure <= 0.0f) return 0;
    if (closure >= 1.0f) return NGP_POSITION_SCALE;
    return (uint16_t)(closure * NGP_POSITION_SCALE + 0.5f);
}

static inline float ngp_position_from_wire(uint16_t value) {
    if (value >= NGP_POSITION_SCALE) return 1.0f;
    return (float)value / (float)NGP_POSITION_SCALE;
}

#ifdef __cplusplus
}
#endif
