import yaml
from visualization.chart import Chart
from indicators import Stochastic, StochasticRSI

# 設定ファイルを読み込む
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 設定からチャートを作成
chart = Chart.from_config(
    config,
    month='2024-05',
    style='yahoo',
    figsize=(12, 8)
)

# ストキャスティクス
stoch = Stochastic(k_period=14, d_period=3, smooth_k=3)
chart.add_indicator(
    stoch,
    panel=1,
    color='blue',
    width=1.0,
    linestyle='-'
)

# ストキャスティクスRSI
stoch_rsi = StochasticRSI(
    rsi_period=14,
    stoch_period=14,
    k_period=3,
    d_period=3
)
chart.add_indicator(
    stoch_rsi,
    panel=2,
    color='purple',
    width=1.0,
    linestyle='-'
)

# チャートを表示して保存
chart.plot(save_path='chart.png')