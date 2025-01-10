# from visualization.chart import Chart
# from main import Config

# # 設定を読み込む
# config = Config('config.yaml')

# # チャートを作成
# chart = Chart.from_config(config.config)

# # チャートを表示
# chart.show(title="BTCUSDT 1時間足チャート")

from signals.filter_signal import ChopFilterSignal

# デフォルトパラメータでの使用
filter_signal = ChopFilterSignal()

# カスタムパラメータでの使用
params = {
    'chop_solid': 50.0  # カスタムしきい値
}
filter_signal = ChopFilterSignal(period=20, solid=params)

print(filter_signal.generate(data))

# 結果の解釈
# direction[i] == 1: 買いシグナル
# direction[i] == -1: 売りシグナル