# Hardware Setup

Target device: Raspberry Pi 4.

## Servos (5x 180-degree, 1.5 kg-cm)
- Default BCM pins: 17, 27, 22, 23, 24
- Frequency: 50 Hz
- Update configs/servo.yaml for your wiring and servo range

## EMG (ADS1115)
- Connect ADS1115 to I2C (SDA/SCL)
- Enable I2C in raspi-config
- Use channel 0 by default

## Camera
- CSI or USB camera
- Configure device index in camera settings if needed

Safety: always test with mock mode first.
