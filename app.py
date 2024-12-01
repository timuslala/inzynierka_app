import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import threading
import time
from collections import deque
import csv
from datetime import datetime
import numpy as np
import math
from serial import Serial
from scipy.signal import butter, filtfilt, find_peaks
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from pyshimmer.dev.channels import ESensorGroup

alerts = []
last_breath_time = time.time()

# Parameters
SAMPLE_RATE = 48.72  # Hz
BUFFER_SIZE = int(120 * SAMPLE_RATE)  # Two minutes of data
LOW_CUT, HIGH_CUT = 0.1, 0.5  # Bandpass filter range (Hz)
MIN_PEAK_DISTANCE = int(1.5 * SAMPLE_RATE)  # Minimum peak distance (in samples)
MIN_AMPLITUDE_CHANGE = 0.008  # Default minimum amplitude change

# Thread-safe deques for sliding windows
data_buffer = deque(maxlen=BUFFER_SIZE)
x_data_buffer = deque(maxlen=BUFFER_SIZE)
y_data_buffer = deque(maxlen=BUFFER_SIZE)
z_data_buffer = deque(maxlen=BUFFER_SIZE)

# Dash app setup
app = dash.Dash(__name__)
app.title = "Licznik Oddechów na Żywo"

# Dark mode styles
dark_style = {
    "background-color": "#1e1e1e",
    "color": "#f5f5f5",
    "font-family": "Arial, sans-serif",
    "padding": "20px",
    "min-height": "100vh",
}

# Main layout
app.layout = html.Div(
    children=[
        html.H1("Licznik Oddechów na Żywo", style={"text-align": "center", "margin-bottom": "20px"}),
        html.Div(
            id="breath-count",
            style={
                "font-size": "24px",
                "margin-bottom": "20px",
                "text-align": "center",
                "padding": "10px",
                "border": "1px solid #444",
                "border-radius": "5px",
                "background-color": "#2e2e2e",
            },
        ),
        html.Div(
            id="breath-frequency",
            style={
                "font-size": "24px",
                "margin-bottom": "20px",
                "text-align": "center",
                "padding": "10px",
                "border": "1px solid #444",
                "border-radius": "5px",
                "background-color": "#2e2e2e",
            },
        ),
        dcc.Graph(
            id="respiratory-chart",
            style={"backgroundColor": "#2e2e2e", "padding": "10px", "border-radius": "5px"},
            config={"displayModeBar": False},
        ),
        html.Button("Ustawienia", id="ustawienia-button", n_clicks=0, style={"margin-top": "20px"}),
        html.Div(
            id="ustawienia-container",
            children=[
                html.Div(
                    [
                        html.Label("Rozmiar okna przesuwnego:"),
                        dcc.Slider(
                            id="sliding-window-slider",
                            min=5,
                            max=120,
                            step=1,
                            value=60,
                            marks={i: f"{i}s" for i in range(5, 121, 15)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Label("Minimalna zmiana amplitudy:", style={"margin-top": "20px"}),
                        dcc.Slider(
                            id="min-amplitude-slider",
                            min=0.001,
                            max=0.02,
                            step=0.001,
                            value=0.008,
                            marks={i / 1000: f"{i / 1000:.3f}" for i in range(1, 21, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Div(
                            dcc.Checklist(
                                id="debug-toggle",
                                options=[{"label": "Debug Dane", "value": "debug"}],
                                value=[],
                                style={
                                    "padding": "10px",
                                    "border": "1px solid #444",
                                    "border-radius": "5px",
                                    "background-color": "#2e2e2e",
                                    "margin-top": "20px",
                                },
                            )
                        ),
                    ],
                    style={"padding": "20px", "background-color": "#1e1e1e", "border": "1px solid #444",
                           "border-radius": "5px"},
                )
            ],
            style={"margin-top": "20px", "display": "none"},  # Initially hidden
        ),
        html.Div(
            id="alerts-container",
            children=[
                html.H2("Alerty", style={"margin-top": "20px", "color": "#f00"}),
                html.Div(id="alerts-list", style={"max-height": "200px", "overflow-y": "auto"})
            ],
            style={"padding": "20px", "background-color": "#2e2e2e", "border": "1px solid #444", "border-radius": "5px"}
        ),
        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
    ],
    style=dark_style,
)
@app.callback(
    Output("alerts-list", "children"),
    Input("update-interval", "n_intervals"),
)
def update_alerts(_):
    return [html.Div(f"[{alert['timestamp']}] {alert['message']}") for alert in alerts[-10:]]

# Callback to toggle ustawienia container visibility
@app.callback(
    Output("ustawienia-container", "style"),
    [Input("ustawienia-button", "n_clicks")],
    prevent_initial_call=True  # Optional: Prevents callback firing on initial page load
)
def toggle_settings_visibility(n_clicks):
    if n_clicks is not None and n_clicks % 2 == 1:
        # Show settings container when the button is clicked an odd number of times
        return {"margin-top": "20px", "display": "block"}
    # Hide settings container for even number of clicks or no clicks
    return {"margin-top": "20px", "display": "none"}

# Callback to update outputs
@app.callback(
    [
        Output("respiratory-chart", "figure"),
        Output("breath-count", "children"),
        Output("breath-frequency", "children"),
    ],
    [
        Input("update-interval", "n_intervals"),
        Input("sliding-window-slider", "value"),
        Input("min-amplitude-slider", "value"),
        Input("debug-toggle", "value"),
    ],
)

def update_all_outputs(n_intervals, window_size_seconds, min_amplitude_change, debug_toggle):
    buffer_size = int(window_size_seconds * SAMPLE_RATE)

    # Pobieranie danych z buforów
    data = list(data_buffer)[-buffer_size:]
    x_data = list(x_data_buffer)[-buffer_size:]
    y_data = list(y_data_buffer)[-buffer_size:]
    z_data = list(z_data_buffer)[-buffer_size:]

    # Walidacja rozmiaru buforów
    if len(data) < SAMPLE_RATE or len(x_data) < SAMPLE_RATE or len(y_data) < SAMPLE_RATE or len(z_data) < SAMPLE_RATE:
        return dash.no_update

    # Wykrywanie oddechów
    breath_count, filtered_signal = detect_breaths(data, SAMPLE_RATE, min_amplitude_change)
    breath_frequency = (breath_count / window_size_seconds) * 60

    # Oś czasu
    time_axis = [i / SAMPLE_RATE for i in range(len(filtered_signal))]

    # Tworzenie wykresu
    respiratory_figure = {
        "data": [
            {"x": time_axis, "y": filtered_signal, "type": "scattergl", "name": "Filtrowany Sygnał"},
        ],
        "layout": {
            "title": "Sygnał Oddechowy",
            "xaxis": {"title": "Czas (s)", "color": dark_style["color"], "gridcolor": "#444"},
            "yaxis": {"title": "Amplituda", "range": [-1, 1], "color": dark_style["color"], "gridcolor": "#444"},
            "paper_bgcolor": dark_style["background-color"],
            "plot_bgcolor": "#2e2e2e",
            "font": {"color": dark_style["color"]},
        },
    }

    # Dodanie sygnałów x, y, z do wykresu, jeśli włączony debug
    if "debug" in debug_toggle:
        respiratory_figure["data"].extend([
            {"x": time_axis, "y": x_data, "type": "scattergl", "name": "Sygnał X"},
            {"x": time_axis, "y": y_data, "type": "scattergl", "name": "Sygnał Y"},
            {"x": time_axis, "y": z_data, "type": "scattergl", "name": "Sygnał Z"},
        ])

    breath_text = f"Liczba Wykrytych Oddechów: {breath_count}"
    frequency_text = f"Częstotliwość Oddechów: {breath_frequency:.2f} oddechów na minutę"
    return respiratory_figure, breath_text, frequency_text

def log_alert(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_entry = {"timestamp": timestamp, "message": message}
    alerts.append(alert_entry)
    with open("alerts.csv", "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "message"])
        if file.tell() == 0:  # Add header only if file is empty
            writer.writeheader()
        writer.writerow(alert_entry)

# Signal processing functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

import numpy as np
import time
from scipy.signal import find_peaks

def detect_breaths(data, sample_rate, min_amplitude_change):
    # Apply bandpass filter
    filtered_signal = bandpass_filter(data, LOW_CUT, HIGH_CUT, sample_rate)

    # Smooth the signal with a moving average
    smoothed_signal = np.convolve(filtered_signal, np.ones(5) / 5, mode='same')

    # Find peaks
    peaks, _ = find_peaks(smoothed_signal, distance=MIN_PEAK_DISTANCE, height=min_amplitude_change)

    # Determine the time window for the last 10 seconds in samples
    samples_in_10_seconds = int(10 * sample_rate)
    recent_signal = smoothed_signal[-samples_in_10_seconds:]  # Extract the last 10 seconds of signal

    # Find peaks in the recent signal
    recent_peaks, _ = find_peaks(recent_signal, distance=MIN_PEAK_DISTANCE, height=min_amplitude_change)

    # Check for no-breath alert
    if len(recent_peaks) == 0:
        log_alert("Brak oddechu przez ostatnie 10 sekund!")

    return len(peaks), smoothed_signal


# Shimmer thread initialization
def shimmer_handler(pkt: DataPacket):
    accel_x = pkt[EChannelType.ACCEL_LSM303DLHC_X] * 9.81 / 16000
    accel_y = pkt[EChannelType.ACCEL_LSM303DLHC_Y] * 9.81 / 16000
    accel_z = pkt[EChannelType.ACCEL_LSM303DLHC_Z] * 9.81 / 16000
    absolute_acceleration = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

    data_buffer.append(absolute_acceleration)
    x_data_buffer.append(accel_x)
    y_data_buffer.append(accel_y)
    z_data_buffer.append(accel_z)

def shimmer_thread():
    serial = Serial("COM6", DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)
    shim_dev.initialize()
    shim_dev.set_sensors(sensors=[ESensorGroup.ACCEL_WR])
    shim_dev.add_stream_callback(shimmer_handler)
    shim_dev.start_streaming()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shim_dev.stop_streaming()

if __name__ == "__main__":
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=shimmer_thread, daemon=True).start()
    app.run_server(debug=True)
