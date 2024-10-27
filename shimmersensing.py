import time

from serial import Serial

from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
global channelTypes


def handler(pkt: DataPacket) -> None:
    global channelTypes

    for channelType in channelTypes:
        cur_value = pkt[channelType]
        print(f'Received new data point for {channelType}: {cur_value}')


if __name__ == '__main__':
    global channelTypes
    serial = Serial('COM6', DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)

    shim_dev.initialize()

    dev_name = shim_dev.get_device_name()
    print(f'My name is: {dev_name}')

    shim_dev.add_stream_callback(handler)
    channelTypes = shim_dev.get_data_types()
    shim_dev.start_streaming()
    time.sleep(5.0)
    shim_dev.stop_streaming()
    shim_dev.shutdown()