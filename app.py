from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from queue import Queue
import threading
import numpy as np
import math
from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from pyshimmer.dev.channels import ESensorGroup
from scipy.signal import butter, filtfilt, find_peaks

# Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Parameters
SAMPLE_RATE = 48.72  # Hz
BUFFER_SIZE = 60 * SAMPLE_RATE  # One minute of data
LOW_CUT, HIGH_CUT = 0.1, 0.5  # Bandpass filter range (Hz)
MIN_PEAK_DISTANCE = int(1.5 * SAMPLE_RATE)  # Minimum peak distance (in samples)

# Thread-safe queue for data
data_queue = Queue(maxsize=int(BUFFER_SIZE))
ready_to_display = threading.Event()

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def detect_breaths(data, sample_rate):
    filtered_signal = bandpass_filter(data, LOW_CUT, HIGH_CUT, sample_rate)
    peaks, _ = find_peaks(filtered_signal, distance=MIN_PEAK_DISTANCE)
    valleys, _ = find_peaks(-filtered_signal, distance=MIN_PEAK_DISTANCE)
    breath_count = min(len(peaks), len(valleys))
    return breath_count, filtered_signal

def shimmer_handler(pkt: DataPacket):
    accel_x = pkt[EChannelType.ACCEL_LSM303DLHC_X] * 9.81 / 16000
    accel_y = pkt[EChannelType.ACCEL_LSM303DLHC_Y] * 9.81 / 16000
    accel_z = pkt[EChannelType.ACCEL_LSM303DLHC_Z] * 9.81 / 16000
    absolute_acceleration = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    if not data_queue.full():
        data_queue.put(absolute_acceleration)
    if data_queue.qsize() >= SAMPLE_RATE * 10:
        ready_to_display.set()

def shimmer_thread():
    serial = Serial("COM6", DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)
    shim_dev.initialize()
    shim_dev.set_sensors(sensors=[ESensorGroup.ACCEL_WR])
    shim_dev.add_stream_callback(shimmer_handler)
    shim_dev.start_streaming()
    try:
        threading.Event().wait()
    except:
        shim_dev.stop_streaming()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def handle_connect():
    socketio.emit("status", {"message": "Connected to server"})

def process_data():
    while True:
        ready_to_display.wait()
        data = []
        while not data_queue.empty():
            data.append(data_queue.get())
        breath_count, filtered_signal = detect_breaths(data, SAMPLE_RATE)
        socketio.emit(
            "update", {"breath_count": breath_count, "filtered_signal": filtered_signal.tolist()}
        )
        ready_to_display.clear()

if __name__ == "__main__":
    threading.Thread(target=shimmer_thread, daemon=True).start()
    threading.Thread(target=process_data, daemon=True).start()
    socketio.run(app, debug=True)
