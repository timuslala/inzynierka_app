import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import threading
import time
from collections import deque
import math
from serial import Serial
from scipy.signal import butter, filtfilt, find_peaks
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from pyshimmer.dev.channels import ESensorGroup

# Parameters
SAMPLE_RATE = 48.72  # Hz
BUFFER_SIZE = int(120 * SAMPLE_RATE)  # Two minutes of data
LOW_CUT, HIGH_CUT = 0.05, 0.8  # Bandpass filter range (Hz)
MIN_PEAK_DISTANCE = int(1.5 * SAMPLE_RATE)  # Minimum peak distance (in samples)

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

# Ustawienia layout
ustawienia_layout = html.Div(
    children=[
        html.H2("Ustawienia", style={"text-align": "center", "margin-bottom": "20px"}),
        html.Div(
            [
                html.Label("Rozmiar okna przesuwnego:"),
                dcc.Slider(
                    id="sliding-window-slider",
                    min=5,
                    max=120,
                    step=1,
                    value=60,  # Default sliding window size in seconds
                    marks={i: f"{i}s" for i in range(5, 121, 15)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Div(
            [
                html.Label("Minimalna zmiana amplitudy:"),
                dcc.Slider(
                    id="min-amplitude-slider",
                    min=0.01,
                    max=0.2,
                    step=0.01,
                    value=0.08,  # Default minimum amplitude change
                    marks={i / 100: f"{i/100:.2f}" for i in range(1, 21, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"margin-bottom": "20px"},
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
                },
            ),
            style={"margin-bottom": "20px"},
        ),
    ],
    style=dark_style,
)

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
        dcc.Interval(id="update-interval", interval=300, n_intervals=0),
        html.Div(id="popup-container", style={"display": "none"}),  # Popup container for Ustawienia
    ],
    style=dark_style,
)

# Popup logic
@app.callback(
    Output("popup-container", "style"),
    Input("ustawienia-button", "n_clicks"),
)
def toggle_settings_popup(n_clicks):
    if n_clicks % 2 == 1:
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output("popup-container", "children"),
    Input("popup-container", "style"),
)
def render_popup(style):
    if style["display"] == "block":
        return ustawienia_layout
    raise PreventUpdate

# Additional sliders
@app.callback(
    Output("sliding-window-slider", "value"),
    Output("min-amplitude-slider", "value"),
    Input("sliding-window-slider", "value"),
    Input("min-amplitude-slider", "value"),
)
def sync_sliders(sliding_window_value, min_amplitude_value):
    return sliding_window_value, min_amplitude_value

# Main update logic
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
    ],
)
def update_all_outputs(n_intervals, window_size_seconds, min_amplitude_change):
    buffer_size = int(window_size_seconds * SAMPLE_RATE)
    data = list(data_buffer)[-buffer_size:]
    if len(data) < SAMPLE_RATE:
        raise PreventUpdate
    breath_count, filtered_signal = detect_breaths(data, SAMPLE_RATE)
    breath_frequency = (breath_count / window_size_seconds) * 60
    time_axis = [i / SAMPLE_RATE for i in range(len(filtered_signal))]
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
    breath_text = f"Liczba Wykrytych Oddechów: {breath_count}"
    frequency_text = f"Częstotliwość Oddechów: {breath_frequency:.2f} oddechów na minutę"
    return respiratory_figure, breath_text, frequency_text

# Bandpass filter functions and detection
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
    peaks, _ = find_peaks(filtered_signal, distance=MIN_PEAK_DISTANCE, height=MIN_AMPLITUDE_CHANGE)
    valleys, _ = find_peaks(-filtered_signal, distance=MIN_PEAK_DISTANCE, height=MIN_AMPLITUDE_CHANGE)
    breath_count = 0
    for peak, valley in zip(peaks, valleys):
        if peak > valley and (filtered_signal[peak] - filtered_signal[valley]) >= MIN_AMPLITUDE_CHANGE:
            breath_count += 1
    return breath_count, filtered_signal

# Shimmer thread initialization
def shimmer_handler(pkt: DataPacket):
    accel_x = pkt[EChannelType.ACCEL_LSM303DLHC_X] * 9.81 / 16000
    accel_y = pkt[EChannelType.ACCEL_LSM303DLHC_Y] * 9.81 / 16000
    accel_z = pkt[EChannelType.ACCEL_LSM303DLHC_Z] * 9.81 / 16000
    absolute_acceleration = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    data_buffer.append(absolute_acceleration)

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
