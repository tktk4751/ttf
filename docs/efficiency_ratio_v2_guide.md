# EfficiencyRatio V2 インジケーター使用ガイド

## 概要

EfficiencyRatio V2 (ER_V2) は、既存のEfficiencyRatioインジケーターを大幅にアップグレードした高性能版です。

### 主な特徴

- **UKF_HLC3プライスソース**: カルマンフィルターで適応処理された高品質な価格データを使用
- **UltimateSmoother**: John Ehlersの最適化された平滑化フィルター
- **動的期間対応**: ドミナントサイクルに基づく適応的な期間調整
- **高精度トレンド判定**: 統計的閾値を用いた精密なレンジ相場検出
- **高速計算**: Numba JITによる最適化された計算処理

## 効率比（Efficiency Ratio）とは

効率比は価格変動の効率性を測定する指標で、以下のように計算されます：

```
ER = |価格変化| / ボラティリティ
```

- **1に近い**: 効率的な価格変動（強いトレンド）
- **0に近い**: 非効率な価格変動（レンジ・ノイズ）

### 判定基準

- **0.618以上**: 効率的な価格変動（強いトレンド）
- **0.382以下**: 非効率な価格変動（レンジ・ノイズ）
- **0.382-0.618**: 中程度の効率性

## 基本的な使用方法

### 1. 基本設定

```python
from indicators.efficiency_ratio_v2 import ER_V2

# 基本設定
er_v2 = ER_V2(
    period=5,                      # 基本期間
    src_type='ukf_hlc3',          # UKF_HLC3プライスソース
    use_ultimate_smoother=True,    # UltimateSmoother使用
    smoother_period=10.0,          # 平滑化期間
    use_dynamic_period=False,      # 固定期間モード
    slope_index=3,                 # トレンド判定期間
    range_threshold=0.005          # レンジ判定閾値
)

# 計算の実行
result = er_v2.calculate(data)
```

### 2. 動的期間設定

```python
# 動的期間設定
er_v2_dynamic = ER_V2(
    period=5,                      # 基本期間
    src_type='ukf_hlc3',          # UKF_HLC3プライスソース
    use_ultimate_smoother=True,    # UltimateSmoother使用
    smoother_period=10.0,          # 平滑化期間
    use_dynamic_period=True,       # 動的期間モード
    detector_type='absolute_ultimate',  # ドミナントサイクル検出方法
    max_cycle=120,                 # 最大サイクル
    min_cycle=5,                   # 最小サイクル
    slope_index=3,                 # トレンド判定期間
    range_threshold=0.005          # レンジ判定閾値
)

result = er_v2_dynamic.calculate(data)
```

## パラメータ説明

### 基本パラメータ

- `period`: 期間（固定期間モード時）
- `src_type`: 価格ソース（デフォルト: 'ukf_hlc3'）
- `ukf_params`: UKFパラメータ（オプション）

### 平滑化パラメータ

- `use_ultimate_smoother`: UltimateSmootherを使用するか（デフォルト: True）
- `smoother_period`: UltimateSmoother期間（デフォルト: 10.0）

### 動的期間パラメータ

- `use_dynamic_period`: 動的期間を使用するか（デフォルト: True）
- `detector_type`: 検出器タイプ（デフォルト: 'absolute_ultimate'）
- `max_cycle`: 最大サイクル期間（デフォルト: 120）
- `min_cycle`: 最小サイクル期間（デフォルト: 5）
- `max_output`: 最大出力値（デフォルト: 120）
- `min_output`: 最小出力値（デフォルト: 5）

### トレンド判定パラメータ

- `slope_index`: トレンド判定期間（デフォルト: 3）
- `range_threshold`: レンジ判定の基本閾値（デフォルト: 0.005）

## 結果の取得

```python
# 計算結果の取得
result = er_v2.calculate(data)

# 各値の取得
er_values = result.values                    # 原のER値
smoothed_values = result.smoothed_values     # 平滑化されたER値
trend_signals = result.trend_signals         # トレンド信号
current_trend = result.current_trend         # 現在のトレンド状態
current_trend_value = result.current_trend_value  # 現在のトレンド値
dynamic_periods = result.dynamic_periods     # 動的期間（動的期間モード時）

# または個別のgetterメソッド
er_values = er_v2.get_values()
smoothed_values = er_v2.get_smoothed_values()
trend_signals = er_v2.get_trend_signals()
current_trend = er_v2.get_current_trend()
current_trend_value = er_v2.get_current_trend_value()
dynamic_periods = er_v2.get_dynamic_periods()
```

## トレンド信号の解釈

トレンド信号は以下の値を取ります：

- **1**: 上昇トレンド
- **-1**: 下降トレンド
- **0**: レンジ相場

現在のトレンド状態は文字列で表現されます：

- **'up'**: 上昇トレンド
- **'down'**: 下降トレンド
- **'range'**: レンジ相場

## 実例

### 基本的な使用例

```python
import pandas as pd
import numpy as np
from indicators.efficiency_ratio_v2 import ER_V2

# サンプルデータの作成
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [101, 102, 103, 104, 105],
    'low': [99, 100, 101, 102, 103],
    'close': [100.5, 101.5, 102.5, 103.5, 104.5],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# ER_V2インジケーターの作成
er_v2 = ER_V2(
    period=5,
    src_type='ukf_hlc3',
    use_ultimate_smoother=True,
    smoother_period=10.0
)

# 計算の実行
result = er_v2.calculate(data)

# 結果の表示
print(f"ER値: {result.values}")
print(f"平滑化値: {result.smoothed_values}")
print(f"現在のトレンド: {result.current_trend}")
```

### 戦略での活用例

```python
def trading_strategy(data):
    """ER_V2を使用した簡単な戦略例"""
    
    # ER_V2の計算
    er_v2 = ER_V2(
        period=5,
        src_type='ukf_hlc3',
        use_ultimate_smoother=True,
        smoother_period=10.0,
        use_dynamic_period=True
    )
    
    result = er_v2.calculate(data)
    
    # 最新の値を取得
    latest_er = result.smoothed_values[-1]
    latest_trend = result.current_trend
    
    # 取引判定
    if latest_er > 0.618 and latest_trend == 'up':
        return "BUY"  # 効率的な上昇トレンド
    elif latest_er > 0.618 and latest_trend == 'down':
        return "SELL"  # 効率的な下降トレンド
    elif latest_er < 0.382:
        return "HOLD"  # 非効率な価格変動（レンジ）
    else:
        return "WAIT"  # 中程度の効率性
```

## パフォーマンス

ER_V2は以下の最適化により高速な計算を実現しています：

- **Numba JIT**: 計算集約的な部分をJITコンパイル
- **キャッシュ機能**: 同じデータでの再計算を回避
- **効率的なメモリ管理**: 最小限のメモリ使用量

## 注意事項

1. **データ品質**: 高品質なOHLCデータが必要です
2. **期間設定**: 短すぎる期間は信頼性が低下します
3. **動的期間**: 十分なデータ量が必要です（最低100-200点推奨）
4. **UKFパラメータ**: 必要に応じてUKFパラメータを調整してください

## 関連ファイル

- `indicators/efficiency_ratio_v2.py`: メインのインジケーター実装
- `examples/efficiency_ratio_v2_example.py`: 使用例
- `tests/test_efficiency_ratio_v2.py`: テストファイル

## 関連インジケーター

- `indicators/efficiency_ratio.py`: 元のEfficiencyRatioインジケーター
- `indicators/ultimate_smoother.py`: UltimateSmootherインジケーター
- `indicators/ehlers_unified_dc.py`: EhlersUnifiedDCインジケーター（動的期間用）

## 更新履歴

- **V2.0**: 初回リリース
  - UKF_HLC3プライスソース対応
  - UltimateSmoother統合
  - 動的期間機能
  - 高精度トレンド判定
  - 高速化最適化 