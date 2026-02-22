import torch
import numpy as np
import timesfm
import matplotlib.pyplot as plt

def main():
    print("Initializing TimesFM 2.5 model...")
    # Set precision for better performance on supporting hardware
    torch.set_float32_matmul_precision("high")

    # Load the pretrained model from Hugging Face
    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Configure the forecast
    print("Configuring forecast parameters...")
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    # Generate synthetic data: A noisy sine wave
    print("Generating synthetic data...")
    t = np.linspace(0, 40, 400)
    series1 = np.sin(t) + 0.1 * np.random.randn(400)
    series2 = np.cos(t * 0.5) + 0.1 * np.random.randn(400)
    
    inputs = [series1, series2]
    horizon = 24

    # Perform forecast
    print(f"Generating forecast for horizon={horizon}...")
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=inputs,
    )

    print("\nForecast completed successfully.")
    print(f"Point forecast shape: {point_forecast.shape}") # Expect (2, 24)
    print(f"Quantile forecast shape: {quantile_forecast.shape}") # Expect (2, 24, 10)

    # Simple summary of results
    for i in range(len(inputs)):
        print(f"\nSeries {i+1} summary:")
        print(f"  Input length: {len(inputs[i])}")
        print(f"  Point forecast (first 5): {point_forecast[i, :5]}")
        print(f"  Quantile forecast (first 5, median): {quantile_forecast[i, :5, 5]}")

if __name__ == "__main__":
    main()
