# アルファトレンドフィルター（Alpha Trend Filter）

アルファトレンドフィルターは、アルファトレンドインデックス(Alpha Trend Index)と効率比(Efficiency Ratio)を組み合わせた二乗平均平方根(RMS)計算による高度なトレンド/レンジ検出指標です。この指標は市場の状態をより正確に把握し、適切な取引戦略を選択するための強力なツールを提供します。

## 主な特徴

- **二乗平均平方根（RMS）による組み合わせ**: トレンドインデックスと効率比を二乗平均平方根で組み合わせることで、両指標の強みを活かす
- **アルファトレンドインデックスの基盤**: 信頼性の高いアルファトレンドインデックスを基本要素として使用
- **効率比（ER）の活用**: 価格の方向性の効率性を測定する効率比を組み合わせて精度向上
- **動的パラメータ調整**: すべてのパラメータは効率比(ER)に基づいて動的に調整され、現在の市場状態に最適化
- **0～1の範囲**: 出力値は0から1の間に正規化され、直感的な解釈が可能

## 主要パラメータ

| パラメータ | デフォルト値 | 説明 |
|------------|--------------|------|
| `er_period` | 21 | 効率比(ER)の計算期間 |
| `max_chop_period` | 21 | チョピネス計算の最大期間 |
| `min_chop_period` | 8 | チョピネス計算の最小期間 |
| `max_atr_period` | 21 | ATR計算の最大期間 |
| `min_atr_period` | 10 | ATR計算の最小期間 |
| `max_stddev_period` | 21 | 標準偏差計算の最大期間 |
| `min_stddev_period` | 14 | 標準偏差計算の最小期間 |
| `max_lookback_period` | 14 | 最小標準偏差を探す最大ルックバック期間 |
| `min_lookback_period` | 7 | 最小標準偏差を探す最小ルックバック期間 |
| `max_rms_window` | 14 | RMS計算の最大ウィンドウサイズ |
| `min_rms_window` | 5 | RMS計算の最小ウィンドウサイズ |

## 二乗平均平方根（Root Mean Square）による組み合わせについて

アルファトレンドフィルターの中核となる計算方法は、トレンドインデックスと効率比(ER)を二乗平均平方根(RMS)で組み合わせる手法です。この計算方法には以下の利点があります：

1. **相互強化**: トレンドインデックスの強みと効率比の強みを同時に活かすことができます
2. **ノイズ低減**: 単純な平均と比較して、二乗平均平方根はより大きな値に重みを置き、ノイズに強くなります
3. **動的適応**: ウィンドウサイズを効率比に基づいて調整することで、市場状態に応じて感度を変えられます
4. **指標の相補性**: トレンドインデックスと効率比を組み合わせることで、一方の弱点を他方で補完します

計算方法：
```
combined_rms[i] = √[(trend_index[i]² + |er[i]|²) / 2]
```

これにより、両方の指標が高い値を示す場合は高い値に、両方が低い値を示す場合は低い値になります。

## 使用例

```python
import numpy as np
from indicators import AlphaTrendFilter

# インジケーターのインスタンス化
alpha_trend_filter = AlphaTrendFilter(
    er_period=21,
    max_chop_period=21,
    min_chop_period=8,
    max_rms_window=14,
    min_rms_window=5
)

# OHLC価格データを用意
open_prices = np.array([...])
high_prices = np.array([...])
low_prices = np.array([...])
close_prices = np.array([...])

# フィルターを計算
result = alpha_trend_filter.calculate(open_prices, high_prices, low_prices, close_prices)

# 結果の取得
filter_values = result.values               # フィルター値（メイン出力）
trend_index = result.trend_index           # アルファトレンドインデックス
er = result.er                             # 効率比
combined_rms = result.combined_rms         # 組み合わせRMS
rms_window = result.rms_window             # RMS計算のウィンドウサイズ

# または、以下のメソッドでも個別に取得可能
trend_index = alpha_trend_filter.get_trend_index()
er = alpha_trend_filter.get_efficiency_ratio()
combined_rms = alpha_trend_filter.get_combined_rms()
rms_window = alpha_trend_filter.get_rms_window()
```

## 市場状態の解釈

アルファトレンドフィルターの値は0から1の間に正規化されており、以下のように解釈できます：

- **0.7～1.0**: 強いトレンド相場 - トレンドフォロー戦略に最適
- **0.3～0.7**: 中間状態 - 市場の方向性に注意しながらの取引が必要
- **0.0～0.3**: 強いレンジ相場 - レンジ戦略や逆張り戦略に適している

## 他の指標との比較

| 指標 | 長所 | 短所 |
|------|------|------|
| Alpha Trend Filter | 複数の要素を二乗平均平方根で組み合わせた高精度な市場状態検出、動的最適化 | 計算がやや複雑 |
| Alpha Trend Index | シンプルで効果的なトレンド/レンジ検出 | 単一指標に依存する限界がある |
| Alpha Filter | 複数の指標を使用した市場フィルタリング | 二乗平均平方根による組み合わせ効果がない |
| Choppiness Index | シンプルなレンジ検出 | トレンドの強さを測定できない |

## 活用のヒント

1. **RMSウィンドウの調整**: より短期的な変化に反応するには`min_rms_window`を小さく設定
2. **複数時間軸の活用**: 長期・中期・短期の時間軸でアルファトレンドフィルターを計算し、時間軸の一致を確認
3. **閾値の調整**: デフォルトの閾値（0.3, 0.7）を市場や銘柄に合わせて微調整
4. **他の指標との組み合わせ**: 方向性を示す指標と組み合わせて最も効果的に活用
5. **トレンド/レンジの確認**: 高値/安値を更新しているかとフィルター値を比較して確認 