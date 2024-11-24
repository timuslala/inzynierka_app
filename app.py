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
app.title = "Live Breath Counter"

# Dark mode styles
dark_style = {
    "background-color": "#1e1e1e",
    "color": "#f5f5f5",
    "font-family": "Arial, sans-serif",
    "padding": "20px",
    "min-height": "100vh",
}

# Layout with Slider for Sliding Window Size
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
        html.Div(
            [
                dcc.Slider(
                    id="sliding-window-slider",
                    min=5,
                    max=120,
                    step=1,
                    value=60,  # Default sliding window size in seconds
                    marks={i: f"{i}s" for i in range(5, 121, 15)},
                    tooltip={"placement": "bottom", "always_visible": True},
                )
            ],
            style={"margin-bottom": "20px"},  # Style applied here
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
        dcc.Interval(id="update-interval", interval=300, n_intervals=0),  # Update every 300 ms
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

# Minimum amplitude change to count a breath
MIN_AMPLITUDE_CHANGE = 0.08  # Adjust based on your signal characteristics

# Breath detection function with amplitude threshold
def detect_breaths(data, sample_rate):
    # Apply the bandpass filter
    filtered_signal = bandpass_filter(data, LOW_CUT, HIGH_CUT, sample_rate)

    # Find peaks and valleys in the filtered signal
    peaks, peak_props = find_peaks(filtered_signal, distance=MIN_PEAK_DISTANCE, height=MIN_AMPLITUDE_CHANGE)
    valleys, valley_props = find_peaks(-filtered_signal, distance=MIN_PEAK_DISTANCE, height=MIN_AMPLITUDE_CHANGE)

    # Only count breaths where the peak-to-valley difference exceeds the threshold
    breath_count = 0
    for peak, valley in zip(peaks, valleys):
        if peak > valley and (filtered_signal[peak] - filtered_signal[valley]) >= MIN_AMPLITUDE_CHANGE:
            breath_count += 1

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
        Output("breath-frequency", "children"),
        Output("debug-plots", "children"),
        Output("debug-plots", "style"),
    ],
    [
        Input("update-interval", "n_intervals"),
        Input("debug-toggle", "value"),
        Input("sliding-window-slider", "value"),
    ],
)
def update_all_outputs(n_intervals, debug_toggle, window_size_seconds):
    # Determine buffer size based on the sliding window
    buffer_size = int(window_size_seconds * SAMPLE_RATE)

    # Extract the last `buffer_size` data points from the fixed-size deque
    data = list(data_buffer)[-buffer_size:]
    x_data = list(x_data_buffer)[-buffer_size:]
    y_data = list(y_data_buffer)[-buffer_size:]
    z_data = list(z_data_buffer)[-buffer_size:]

    # Ensure enough data is available
    if len(data) < SAMPLE_RATE:  # Require at least 1 second of data
        raise dash.exceptions.PreventUpdate

    # Detect breaths and filter signal
    breath_count, filtered_signal = detect_breaths(data, SAMPLE_RATE)
    breath_frequency = (breath_count / window_size_seconds) * 60

    # Time axis for the current sliding window
    time_axis = [i / SAMPLE_RATE for i in range(len(filtered_signal))]

    # Main respiratory chart
    respiratory_figure = {
        "data": [
            {"x": time_axis, "y": filtered_signal, "type": "scattergl", "name": "Filtered Signal"},
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
        debug_figures = [
            dcc.Graph(
                figure={
                    "data": [{"x": time_axis, "y": x_data, "type": "scattergl", "name": "X Data"}],
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
                    "data": [{"x": time_axis, "y": y_data, "type": "scattergl", "name": "Y Data"}],
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
                    "data": [{"x": time_axis, "y": z_data, "type": "scattergl", "name": "Z Data"}],
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

    # Update text outputs
    breath_text = f"Total Breaths Detected: {breath_count}"
    frequency_text = f"Breath Frequency: {breath_frequency:.2f} breaths per minute"

    return respiratory_figure, breath_text, frequency_text, debug_figures, debug_style



# Start the shimmer thread
if __name__ == "__main__":
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":  # Only run threads in the reloader process
        threading.Thread(target=shimmer_thread, daemon=True).start()
    app.run_server(debug=True)
