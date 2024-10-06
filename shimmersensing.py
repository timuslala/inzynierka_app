import serial
import struct
import sys
import time
import math
import numpy as np

# Constants for conversion
ACCEL_RANGE_G = 2  # Adjust this based on your Shimmer accelerometer range (2, 4, 8, 16)
SCALE_FACTOR = ACCEL_RANGE_G / 32768.0  # Scaling factor for Shimmer accelerometer data
G_TO_MS2 = 9.81  # Conversion factor from g to m/s²
GYRO_SCALE_FACTOR = 250.0 / 32768.0  # Scaling for gyroscope (range ±250°/s)

# Low-pass filter for drift reduction
ALPHA = 0.8  # Low-pass filter constant
DRIFT_THRESHOLD = 0.05  # Drift threshold to correct integration drift
VELOCITY_THRESHOLD = 0.01  # Threshold to reset velocity to zero if too small

# Debugging variables
debug_print_interval = 100  # co ile próbek wyświetlać zmienne do debugowania
debug_counter = 0  # licznik próbek


def low_pass_filter(current_value, previous_value):
    """Low-pass filter to smooth the data."""
    return ALPHA * previous_value + (1 - ALPHA) * current_value


def calibrate_accelerometer():
    """Calibrates the accelerometer by calculating the gravity vector when the device is stationary."""
    calibration_samples = 100  # Number of samples for calibration
    sum_x, sum_y, sum_z = 0, 0, 0
    numbytes = 0
    ddata = bytes()

    for _ in range(calibration_samples):
        while numbytes < framesize:
            ddata += ser.read(framesize)
            numbytes = len(ddata)

        data = ddata[0:framesize]
        ddata = ddata[framesize:]
        numbytes = len(ddata)

        (analogaccelx, analogaccely, analogaccelz) = struct.unpack('HHH', data[4:framesize])

        sum_x += analogaccelx - 32768
        sum_y += analogaccely - 32768
        sum_z += analogaccelz - 32768

    avg_x = sum_x / calibration_samples
    avg_y = sum_y / calibration_samples
    avg_z = sum_z / calibration_samples

    # Determine gravity vector
    gravity_magnitude = math.sqrt(avg_x ** 2 + avg_y ** 2 + avg_z ** 2)
    gravity_x = (avg_x / gravity_magnitude) * G_TO_MS2
    gravity_y = (avg_y / gravity_magnitude) * G_TO_MS2
    gravity_z = (avg_z / gravity_magnitude) * G_TO_MS2

    return gravity_x, gravity_y, gravity_z


def read_gyroscope(data):
    """Read gyroscope data."""
    (gyrox, gyroy, gyroz) = struct.unpack('hhh', data[framesize:framesize + 6])
    return gyrox * GYRO_SCALE_FACTOR, gyroy * GYRO_SCALE_FACTOR, gyroz * GYRO_SCALE_FACTOR


def update_orientation(gyro_x, gyro_y, gyro_z, dt, orientation):
    """Update the device's orientation based on gyroscope data."""
    orientation[0] += gyro_x * dt  # Rotation around X-axis
    orientation[1] += gyro_y * dt  # Rotation around Y-axis
    orientation[2] += gyro_z * dt  # Rotation around Z-axis

    # Convert to radians for trigonometric functions
    roll = np.radians(orientation[0])
    pitch = np.radians(orientation[1])
    yaw = np.radians(orientation[2])

    return roll, pitch, yaw


def apply_gravity_compensation(accel_x, accel_y, accel_z, roll, pitch, yaw):
    """Compensate for the effect of gravity on acceleration."""
    # Rotation matrix from Euler angles (roll, pitch, yaw)
    R = np.array([
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
        [np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
         np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
         np.sin(roll) * np.cos(pitch)],
        [np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw),
         np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw),
         np.cos(roll) * np.cos(pitch)]
    ])

    # Local acceleration transformed to Earth reference frame
    accel_global = np.dot(R, np.array([accel_x, accel_y, accel_z]))

    # Compensate for gravity
    accel_x_comp = accel_global[0]
    accel_y_comp = accel_global[1]
    accel_z_comp = accel_global[2] - G_TO_MS2  # Subtract gravity

    return accel_x_comp, accel_y_comp, accel_z_comp


def integrate_velocity(accel_comp, prev_velocity, dt):
    """Integrate acceleration to get velocity, with drift correction."""
    if abs(accel_comp) < DRIFT_THRESHOLD:
        accel_comp = 0
    new_velocity = prev_velocity + accel_comp * dt

    # Apply a small velocity threshold to counteract drift
    if abs(new_velocity) < VELOCITY_THRESHOLD:
        new_velocity = 0

    return new_velocity


def wait_for_ack():
    ddata = bytes()
    ack = struct.pack('B', 0xff)
    while ddata != ack:
        ddata = ser.read(1)
        print("0x%02x" % ddata[0])
    return


if len(sys.argv) < 2:
    print("no device specified")
    print("You need to specify the serial port of the device you wish to connect to")
    print("example:")
    print("   aAccel5Hz.py Com12")
    print("or")
    print("   aAccel5Hz.py /dev/rfcomm0")
else:
    ser = serial.Serial(sys.argv[1], 115200)
    ser.flushInput()
    print("port opening, done.")

    # Send sensor configuration command
    ser.write(struct.pack('BBBB', 0x08, 0x80, 0x00, 0x00))  # analogaccel
    wait_for_ack()
    print("sensor setting, done.")

    # Set sampling rate
    ser.write(struct.pack('BBB', 0x05, 0x00, 0x19))  # 5.12Hz (6400 (0x1900)).
    wait_for_ack()
    print("sampling rate setting, done.")

    # Start streaming data
    ser.write(struct.pack('B', 0x07))
    wait_for_ack()
    print("start command sending, done.")

    # Read data
    ddata = bytes()
    numbytes = 0
    framesize = 10  # 1 byte packet type + 3 bytes timestamp + 3x2 bytes Analog Accel

    print(
        "Packet Type,Timestamp,Accel Norm (m/s²),Current Velocity (m/s),Accel X,Y,Z (m/s²),Gyro X,Y,Z (°/s),Orientation Roll,Pitch,Yaw")

    prev_time = None
    current_velocity_x = 0.0  # Velocity in X-axis
    current_velocity_y = 0.0  # Velocity in Y-axis
    current_velocity_z = 0.0  # Velocity in Z-axis

    # Calibration - determining gravity vector
    gravity_x, gravity_y, gravity_z = calibrate_accelerometer()

    # Initialize orientation (roll, pitch, yaw)
    orientation = [0.0, 0.0, 0.0]

    # Filtered values
    filtered_accel_norm = 0

    # Initialize acceleration compensation variables to avoid NameError
    accel_x_comp, accel_y_comp, accel_z_comp = 0, 0, 0
    roll, pitch, yaw = 0, 0, 0  # Initialize orientation variables

    try:
        while True:
            while numbytes < framesize + 6:  # Wait for full data packet (accelerometer + gyroscope)
                ddata += ser.read(framesize + 6)
                numbytes = len(ddata)

            data = ddata[0:framesize + 6]
            ddata = ddata[framesize + 6:]
            numbytes = len(ddata)

            (packettype) = struct.unpack('B', data[0:1])
            (timestamp0, timestamp1, timestamp2) = struct.unpack('BBB', data[1:4])
            (analogaccelx, analogaccely, analogaccelz) = struct.unpack('HHH', data[4:framesize])

            timestamp = timestamp0 + timestamp1 * 256 + timestamp2 * 65536

            # Read gyroscope data
            gyro_x, gyro_y, gyro_z = read_gyroscope(data)

            if prev_time is not None:
                dt = (timestamp - prev_time) / 5120.0  # Time in seconds (5.12 Hz)

                # Convert acceleration values to m/s²
                accel_x = (analogaccelx - 32768) * SCALE_FACTOR * G_TO_MS2
                accel_y = (analogaccely - 32768) * SCALE_FACTOR * G_TO_MS2
                accel_z = (analogaccelz - 32768) * SCALE_FACTOR * G_TO_MS2

                # Update orientation based on gyroscope data
                roll, pitch, yaw = update_orientation(gyro_x, gyro_y, gyro_z, dt, orientation)

                # Compensate for gravity
                accel_x_comp, accel_y_comp, accel_z_comp = apply_gravity_compensation(accel_x, accel_y, accel_z, roll,
                                                                                      pitch, yaw)

                # Calculate acceleration norm
                accel_norm = math.sqrt(accel_x_comp ** 2 + accel_y_comp ** 2 + accel_z_comp ** 2)

                # Low-pass filter for smoothing data
                filtered_accel_norm = low_pass_filter(accel_norm, filtered_accel_norm)

                # If acceleration is below threshold (noise), set to zero
                if filtered_accel_norm < DRIFT_THRESHOLD:
                    filtered_accel_norm = 0

                # Integrate to obtain velocity for each axis
                current_velocity_x = integrate_velocity(accel_x_comp, current_velocity_x, dt)
                current_velocity_y = integrate_velocity(accel_y_comp, current_velocity_y, dt)
                current_velocity_z = integrate_velocity(accel_z_comp, current_velocity_z, dt)

            prev_time = timestamp

            # Increment debug counter
            debug_counter += 1

            # Debug print every N samples
            if debug_counter % debug_print_interval == 0:
                print(
                    f"DEBUG: Raw Accel X,Y,Z: {accel_x:.2f},{accel_y:.2f},{accel_z:.2f} | Compensated Accel X,Y,Z: {accel_x_comp:.2f},{accel_y_comp:.2f},{accel_z_comp:.2f}")
                print(
                    f"DEBUG: Gyro X,Y,Z: {gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f} | Orientation Roll,Pitch,Yaw: {np.degrees(roll):.2f},{np.degrees(pitch):.2f},{np.degrees(yaw):.2f}")
                print(
                    f"DEBUG: Current Velocity X,Y,Z: {current_velocity_x:.2f},{current_velocity_y:.2f},{current_velocity_z:.2f} m/s | Accel Norm (filtered): {filtered_accel_norm:.2f}")

            # Display current data including debug values
            print("0x%02x,%5d,\t%.2f,\t%.2f,\t%.2f,%.2f,%.2f,\t%.2f,%.2f,%.2f,\t%.2f,%.2f,%.2f" % (
                packettype[0], timestamp, filtered_accel_norm, (current_velocity_x + current_velocity_y + current_velocity_z) / 3,
                accel_x_comp, accel_y_comp, accel_z_comp,
                gyro_x, gyro_y, gyro_z,
                np.degrees(roll), np.degrees(pitch), np.degrees(yaw)))

    except KeyboardInterrupt:
        # Stop streaming
        ser.write(struct.pack('B', 0x20))
        print("stop command sent, waiting for ACK_COMMAND")
        wait_for_ack()
        print("ACK_COMMAND received.")
        ser.close()
        print("All done")
