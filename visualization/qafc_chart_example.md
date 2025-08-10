# QAFC Chart Visualization Examples

## Basic Usage

Display the QAFC chart with default parameters:
```bash
python visualization/qafc_chart.py
```

## Save Chart to File

Save the chart as a PNG image:
```bash
python visualization/qafc_chart.py --output qafc_analysis.png
```

## Specify Date Range

Display data for a specific date range:
```bash
python visualization/qafc_chart.py --start 2024-01-01 --end 2024-12-31
```

## Adjust QAFC Parameters

Fine-tune the QAFC indicator parameters:
```bash
python visualization/qafc_chart.py \
    --process-noise 0.005 \
    --measurement-noise 0.05 \
    --noise-window 30 \
    --prediction-lookback 15 \
    --base-multiplier 2.5
```

## Different Price Sources

Use different price sources (close, hlc3, hl2, etc.):
```bash
python visualization/qafc_chart.py --src-type close
```

## Hide Volume Panel

Display chart without volume panel:
```bash
python visualization/qafc_chart.py --no-volume
```

## Complete Example

A complete example with multiple options:
```bash
python visualization/qafc_chart.py \
    --config config.yaml \
    --start 2024-06-01 \
    --end 2024-12-31 \
    --src-type hlc3 \
    --process-noise 0.01 \
    --measurement-noise 0.1 \
    --noise-window 20 \
    --base-multiplier 2.0 \
    --output visualization/output/qafc_analysis_2024.png
```

## Key Features Displayed

The chart displays:
1. **Main Panel**: Candlestick chart with QAFC channels
   - Color-coded centerline (green for uptrend, red for downtrend, gray for neutral)
   - Dynamic upper and lower channels
   
2. **Volume Panel**: Trading volume (optional)

3. **Trend Strength Panel**: Shows the strength of the detected trend (0-1)

4. **Confidence Score Panel**: Displays the prediction confidence (0-1)

5. **Momentum Flow Panel**: Bar chart showing momentum direction and strength

## Understanding the Indicators

- **Centerline**: The ultra-low latency Kalman filtered price
- **Channels**: Dynamic width based on volatility, trend strength, and confidence
- **Trend Strength**: Higher values indicate stronger trends
- **Confidence Score**: Higher values indicate more reliable predictions
- **Momentum Flow**: Positive for bullish momentum, negative for bearish