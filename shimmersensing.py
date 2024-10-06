import serial
import struct
import sys
import time

# Constants for conversion
ACCEL_RANGE_G = 2  # Adjust this based on your Shimmer accelerometer range (2, 4, 8, 16)
SCALE_FACTOR = ACCEL_RANGE_G / 32768.0  # Scaling factor for Shimmer accelerometer data
G_TO_MS2 = 9.81  # Conversion factor from g to m/s²

if len(sys.argv) < 2:
    print("No device specified")
    print("You need to specify the serial port of the device you wish to connect to")
    print("Example:")
    print("   python3 shimmersensing.py COM12")
    print("Or")
    print("   python3 shimmersensing.py /dev/rfcomm0")
else:
    ser = serial.Serial(sys.argv[1], 115200)
    ser.flushInput()
    print("Port opening, done.")

    # Send sensor configuration command
    ser.write(struct.pack('BBBB', 0x08, 0x80, 0x00, 0x00))  # analogaccel
    print("Sensor setting, done.")

    # Set sampling rate
    ser.write(struct.pack('BBB', 0x05, 0x00, 0x19))  # 5.12Hz (6400 (0x1900)).
    print("Sampling rate setting, done.")

    # Start streaming data
    ser.write(struct.pack('B', 0x07))
    print("Start command sending, done.")

    # Read data
    ddata = bytes()
    numbytes = 0
    framesize = 10  # 1 byte packet type + 3 bytes timestamp + 3x2 bytes Analog Accel

    try:
        while True:
            while numbytes < framesize:  # Wait for full data packet (accelerometer only)
                ddata += ser.read(framesize)
                numbytes = len(ddata)

            data = ddata[0:framesize]
            ddata = ddata[framesize:]
            numbytes = len(ddata)

            (analogaccelx, analogaccely, analogaccelz) = struct.unpack('HHH', data[4:framesize])

            # Convert raw accelerometer values to m/s²
            accel_x = (analogaccelx - 32768) * SCALE_FACTOR * G_TO_MS2
            accel_y = (analogaccely - 32768) * SCALE_FACTOR * G_TO_MS2
            accel_z = (analogaccelz - 32768) * SCALE_FACTOR * G_TO_MS2

            # Print accelerometer values
            print(f"Accel X: {accel_x:.2f} m/s², Accel Y: {accel_y:.2f} m/s², Accel Z: {accel_z:.2f} m/s²")

    except KeyboardInterrupt:
        # Stop streaming
        ser.write(struct.pack('B', 0x20))
        print("Stop command sent.")
        ser.close()
        print("All done.")
