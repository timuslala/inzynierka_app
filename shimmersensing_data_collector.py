
import time
from serial import Serial
import matplotlib.pyplot as plt
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from pyshimmer.dev.channels import ESensorGroup
import math
global values_to_plot, accels
import numpy as np
import scipy.signal as signal
import pandas as pd





def handler(pkt: DataPacket) -> None:
    global accels
    # Wyciągnij wartości z `pkt`
    timestamp = pkt[EChannelType.TIMESTAMP]
    accel_x = pkt[EChannelType.ACCEL_LSM303DLHC_X]
    accel_y = pkt[EChannelType.ACCEL_LSM303DLHC_Y]
    accel_z = pkt[EChannelType.ACCEL_LSM303DLHC_Z]


    # Przelicz przyspieszenie na m/s^2 jeśli jest w milig (1 g ≈ 9.81 m/s^2)
    accel_x = accel_x * 9.81 / 16000
    accel_y = accel_y * 9.81 / 16000
    accel_z = accel_z * 9.81 / 16000

    # Oblicz absolutne przyspieszenie (norma wektora przyspieszenia)
    absolute_acceleration = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

    accels.append((accel_x, accel_y, accel_z, absolute_acceleration))



if __name__ == '__main__':
    global accels
    accels = []

    serial = Serial('COM6', DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)

    shim_dev.initialize()

    dev_name = shim_dev.get_device_name()
    print(f'My name is: {dev_name}')
    shim_dev.set_sensors(sensors=[ESensorGroup.ACCEL_WR])
    shim_dev.add_stream_callback(handler)

    shim_dev.start_streaming()
    time.sleep(60)
    shim_dev.stop_streaming()
    accels = pd.DataFrame(accels, columns=['accel_x', 'accel_y', 'accel_z', 'accel_absolute'])
    accels.to_pickle("./accels4.pkl")
    shim_dev.shutdown()

