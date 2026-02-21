"""
ComputeFFT - Branch C Step 2: Heavy Path (600ms)

Simulates Fast Fourier Transform computation on the windowed signal data.
Produces frequency spectrum analysis including dominant frequencies,
harmonic peaks, and spectral energy.

Branch C total: Windowing (400ms) + ComputeFFT (600ms) + FailureModel (1000ms) = 2000ms
"""
import json
import time
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Heavy operation step 2: 600ms (FFT computation)
PROCESSING_DELAY_MS = 600


def lambda_handler(event, context):
    """
    Compute FFT frequency analysis from windowed signal.

    This is step 2 of the heavy processing path (Branch C).
    Receives Windowing output as input.
    """
    start_time = time.time()

    # Extract data from Windowing output
    machine_id = event.get("machine_id", "unknown")
    sensor_id = event.get("sensor_id", "unknown")
    windowed_signal = event.get("windowed_signal", [])
    window_size = event.get("window_size", 0)
    sample_rate_hz = event.get("sample_rate_hz", 1000)

    logger.info(json.dumps({
        "event": "compute_fft_start",
        "machine_id": machine_id,
        "branch": "heavy_path",
        "chain_step": 2,
        "window_size": window_size
    }))

    print(f"[ComputeFFT] Starting - Chain step 2/3 ({PROCESSING_DELAY_MS}ms)")
    print(f"[ComputeFFT] Machine: {machine_id}, Window: {window_size} samples")

    # Simulate heavy FFT computation
    time.sleep(PROCESSING_DELAY_MS / 1000.0)

    # Simulated FFT frequency analysis results
    dominant_frequency_hz = round(random.uniform(50.0, 500.0), 1)

    frequency_spectrum = {
        "dominant_frequency_hz": dominant_frequency_hz,
        "harmonic_peaks": [
            round(dominant_frequency_hz * i, 1) for i in range(1, 4)
        ],
        "spectral_energy": round(random.uniform(0.5, 5.0), 3),
        "noise_floor_db": round(random.uniform(-40.0, -20.0), 1),
        "signal_to_noise_ratio_db": round(random.uniform(10.0, 40.0), 1),
        "frequency_resolution_hz": round(sample_rate_hz / max(window_size, 1), 2),
        "nyquist_frequency_hz": sample_rate_hz / 2
    }

    total_time = (time.time() - start_time) * 1000

    result = {
        "function": "ComputeFFT",
        "branch": "heavy_path",
        "chain_step": 2,
        "machine_id": machine_id,
        "sensor_id": sensor_id,
        "status": "complete",
        "frequency_spectrum": frequency_spectrum,
        "processing_time_ms": int(total_time),
        "artificial_delay_ms": PROCESSING_DELAY_MS,
        "timestamp": time.time()
    }

    logger.info(json.dumps({
        "event": "compute_fft_complete",
        "machine_id": machine_id,
        "chain_step": 2,
        "processing_time_ms": int(total_time),
        "dominant_frequency_hz": dominant_frequency_hz
    }))

    print(f"[ComputeFFT] COMPLETE in {total_time:.2f}ms")
    print(f"[ComputeFFT] Dominant freq: {dominant_frequency_hz}Hz, Energy: {frequency_spectrum['spectral_energy']}")
    print(f"[ComputeFFT] Next: FailureModel (ML prediction)")

    return result


if __name__ == '__main__':
    test_event = {
        "machine_id": "MACH-TEST-001",
        "sensor_id": "SENS-TEST-01",
        "windowed_signal": [0.0, 1.2, 3.4, 5.6, 4.3, 2.1, 0.5, 0.0],
        "window_size": 8,
        "sample_rate_hz": 1000
    }

    result = lambda_handler(test_event, None)
    print(f"\nResult: {json.dumps(result, indent=2)}")
