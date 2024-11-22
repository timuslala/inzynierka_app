document.getElementById("data-form").onsubmit = async function (e) {
    e.preventDefault();

    // Get input data
    const data = document.getElementById("data").value;
    const sampleRate = document.getElementById("sample_rate").value;

    // Send data to backend
    const response = await fetch("/process", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            data: JSON.parse(data),
            sample_rate: sampleRate,
        }),
    });

    const result = await response.json();

    if (response.ok) {
        // Update results
        document.getElementById("breath-count").innerText = result.breath_count;

        // Update chart
        updateChart(result.respiratory_signal, result.peaks, result.valleys);
    } else {
        alert(result.error || "Error processing data");
    }
};

function updateChart(signal, peaks, valleys) {
    const ctx = document.getElementById("respiratory-chart").getContext("2d");
    const time = Array.from({ length: signal.length }, (_, i) => i);

    // Create data for peaks and valleys
    const peakPoints = peaks.map((i) => ({ x: i, y: signal[i] }));
    const valleyPoints = valleys.map((i) => ({ x: i, y: signal[i] }));

    // Render chart
    new Chart(ctx, {
        type: "line",
        data: {
            datasets: [
                { label: "Respiratory Signal", data: signal, borderColor: "blue" },
                { label: "Peaks", data: peakPoints, type: "scatter", backgroundColor: "green" },
                { label: "Valleys", data: valleyPoints, type: "scatter", backgroundColor: "red" },
            ],
        },
        options: {
            scales: {
                x: { type: "linear" },
                y: { beginAtZero: false },
            },
        },
    });
}
