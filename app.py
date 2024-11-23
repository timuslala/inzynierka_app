import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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
BUFFER_SIZE = int(60 * SAMPLE_RATE)  # One minute of data
LOW_CUT, HIGH_CUT = 0.1, 0.5  # Bandpass filter range (Hz)
MIN_PEAK_DISTANCE = int(1.5 * SAMPLE_RATE)  # Minimum peak distance (in samples)

# Thread-safe deques for sliding windows
data_buffer = deque(maxlen=BUFFER_SIZE)
x_data_buffer = deque(maxlen=BUFFER_SIZE)
y_data_buffer = deque(maxlen=BUFFER_SIZE)
z_data_buffer = deque(maxlen=BUFFER_SIZE)

# Dash app setup
app = dash.Dash(__name__)
app.title = "Live Breath Counter"

# Dark mode styles
dark_style = {
    "background-color": "#1e1e1e",
    "color": "#f5f5f5",
    "font-family": "Arial, sans-serif",
    "padding": "20px",
    "min-height": "100vh",
}

# Layout
app.layout = html.Div(
    children=[
        html.H1("Live Breath Counter", style={"text-align": "center", "margin-bottom": "20px"}),
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
        dcc.Graph(
            id="respiratory-chart",
            style={"backgroundColor": "#2e2e2e", "padding": "10px", "border-radius": "5px"},
            config={"displayModeBar": False},  # Remove toolbar for cleaner look
        ),
        dcc.Checklist(
            id="debug-toggle",
            options=[{"label": "Debug Data", "value": "debug"}],
            value=[],
            style={
                "margin-bottom": "20px",
                "padding": "10px",
                "border": "1px solid #444",
                "border-radius": "5px",
                "background-color": "#2e2e2e",
            },
        ),
        html.Div(
            id="debug-plots",
            style={
                "display": "none",
                "padding": "10px",
                "border": "1px solid #444",
                "border-radius": "5px",
                "background-color": "#2e2e2e",
            },
        ),
        dcc.Interval(id="update-interval", interval=300, n_intervals=0),  # Update every second
    ],
)

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Breath detection function
def detect_breaths(data, sample_rate):
    filtered_signal = bandpass_filter(data, LOW_CUT, HIGH_CUT, sample_rate)
    peaks, _ = find_peaks(filtered_signal, distance=MIN_PEAK_DISTANCE)
    valleys, _ = find_peaks(-filtered_signal, distance=MIN_PEAK_DISTANCE)
    breath_count = min(len(peaks), len(valleys))
    return breath_count, filtered_signal

# Shimmer data acquisition
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
    print("Starting shimmer thread")
    serial = Serial("COM6", DEFAULT_BAUDRATE)
    shim_dev = ShimmerBluetooth(serial)
    shim_dev.initialize()
    shim_dev.set_sensors(sensors=[ESensorGroup.ACCEL_WR])
    shim_dev.add_stream_callback(shimmer_handler)
    shim_dev.start_streaming()

    try:
        while True:
            time.sleep(1)  # Keep the thread alive
    except KeyboardInterrupt:
        shim_dev.stop_streaming()

# Callback to update the main chart and debug plots
@app.callback(
    [
        Output("respiratory-chart", "figure"),
        Output("breath-count", "children"),
        Output("debug-plots", "children"),
        Output("debug-plots", "style"),
    ],
    [
        Input("update-interval", "n_intervals"),
        Input("debug-toggle", "value"),
    ],
)
def update_charts(n_intervals, debug_toggle):
    if len(data_buffer) < SAMPLE_RATE:  # Wait until sufficient data is available
        return dash.no_update, dash.no_update, dash.no_update, {"display": "none"}

    data = list(data_buffer)
    breath_count, filtered_signal = detect_breaths(data, SAMPLE_RATE)

    # Time axis for the sliding window
    time_axis = [i / SAMPLE_RATE for i in range(len(filtered_signal))]

    # Main respiratory chart
    respiratory_figure = {
        "data": [
            {"x": time_axis, "y": filtered_signal, "type": "line", "name": "Filtered Signal"},
        ],
        "layout": {
            "title": "Respiratory Signal",
            "xaxis": {"title": "Time (s)", "color": dark_style["color"], "gridcolor": "#444"},
            "yaxis": {"title": "Amplitude", "range": [-1, 1], "color": dark_style["color"], "gridcolor": "#444"},
            "paper_bgcolor": dark_style["background-color"],
            "plot_bgcolor": "#2e2e2e",
            "font": {"color": dark_style["color"]},
        },
    }

    # Debug plots
    if "debug" in debug_toggle:
        x_data = list(x_data_buffer)
        y_data = list(y_data_buffer)
        z_data = list(z_data_buffer)

        debug_figures = [
            dcc.Graph(
                figure={
                    "data": [{"x": time_axis, "y": x_data, "type": "line", "name": "X Data"}],
                    "layout": {
                        "title": "X Data",
                        "xaxis": {"title": "Time (s)", "color": dark_style["color"], "gridcolor": "#444"},
                        "yaxis": {"title": "Amplitude", "color": dark_style["color"], "gridcolor": "#444"},
                        "paper_bgcolor": dark_style["background-color"],
                        "plot_bgcolor": "#2e2e2e",
                        "font": {"color": dark_style["color"]},
                    },
                }
            ),
            dcc.Graph(
                figure={
                    "data": [{"x": time_axis, "y": y_data, "type": "line", "name": "Y Data"}],
                    "layout": {
                        "title": "Y Data",
                        "xaxis": {"title": "Time (s)", "color": dark_style["color"], "gridcolor": "#444"},
                        "yaxis": {"title": "Amplitude", "color": dark_style["color"], "gridcolor": "#444"},
                        "paper_bgcolor": dark_style["background-color"],
                        "plot_bgcolor": "#2e2e2e",
                        "font": {"color": dark_style["color"]},
                    },
                }
            ),
            dcc.Graph(
                figure={
                    "data": [{"x": time_axis, "y": z_data, "type": "line", "name": "Z Data"}],
                    "layout": {
                        "title": "Z Data",
                        "xaxis": {"title": "Time (s)", "color": dark_style["color"], "gridcolor": "#444"},
                        "yaxis": {"title": "Amplitude", "color": dark_style["color"], "gridcolor": "#444"},
                        "paper_bgcolor": dark_style["background-color"],
                        "plot_bgcolor": "#2e2e2e",
                        "font": {"color": dark_style["color"]},
                    },
                }
            ),
        ]
        debug_style = {"display": "block"}
    else:
        debug_figures = []
        debug_style = {"display": "none"}

    # Update breath count
    breath_text = f"Total Breaths Detected: {breath_count}"

    return respiratory_figure, breath_text, debug_figures, debug_style

# Start the shimmer thread
if __name__ == "__main__":
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":  # Only run threads in the reloader process
        threading.Thread(target=shimmer_thread, daemon=True).start()
    app.run_server(debug=True)
