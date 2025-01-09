from visualization.chart import Chart
from main import Config

# 設定を読み込む
config = Config('config.yaml')

# チャートを作成
chart = Chart.from_config(config.config)

# チャートを表示
chart.show(title="BTCUSDT 1時間足チャート")