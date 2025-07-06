# Ultimate Smoother 使用ガイド

John Ehlersの論文「The Ultimate Smoother」に基づくアルティメットスムーザーの実装と使用方法を説明します。

## 概要

アルティメットスムーザーは、John Ehlersが開発した高性能なスムージングフィルターです。従来のフィルターと比較して、パスバンドでのラグが最小化され、高周波ノイズの除去に優れています。

### 主な特徴

- **ゼロラグ**: パスバンドにおいてラグが最小化される
- **第2次IIRフィルター**: 無限インパルス応答フィルターの実装
- **高速計算**: Numba最適化による高速処理
- **UKF統合**: Unscented Kalman Filterとの統合サポート
- **完全統合**: プロジェクトアーキテクチャとの完全統合

## 数学的基礎

アルティメットスムーザーは以下の数学的定義に基づいて計算されます：

### 係数計算

```
a1 = exp(-1.414 * π / period)
c2 = 2 * a1 * cos(1.414 * π / period)
c3 = -a1 * a1
c1 = (1 + c2 - c3) / 4
```

### Ultimate Smoother式

```
US[i] = (1 - c1) * Price[i] + 
        (2 * c1 - c2) * Price[i-1] - 
        (c1 + c3) * Price[i-2] + 
        c2 * US[i-1] + 
        c3 * US[i-2]
```

## 基本的な使用方法

### 1. 基本的な使用例

```python
from indicators.ultimate_smoother import UltimateSmoother
import pandas as pd

# データフレームの準備
data = pd.DataFrame({
    'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
})

# UltimateSmoother の作成
smoother = UltimateSmoother(period=20, src_type='close')

# 計算実行
result = smoother.calculate(data)

# 結果の取得
ultimate_values = result.values
coefficients = result.coefficients

print(f"Ultimate Smoother値: {ultimate_values}")
print(f"係数: {coefficients}")
```

### 2. 異なるソースタイプでの使用

```python
# 異なる価格ソースでの使用
smoothers = {
    'close': UltimateSmoother(period=20, src_type='close'),
    'hlc3': UltimateSmoother(period=20, src_type='hlc3'),
    'hl2': UltimateSmoother(period=20, src_type='hl2'),
    'ohlc4': UltimateSmoother(period=20, src_type='ohlc4')
}

for name, smoother in smoothers.items():
    result = smoother.calculate(data)
    print(f"{name}: {result.values[-1]:.4f}")
```

### 3. UKFとの統合

```python
# UKFパラメータの設定
ukf_params = {
    'alpha': 0.001,
    'process_noise_scale': 0.0005,
    'volatility_window': 15
}

# UKFソースでの使用
smoother_ukf = UltimateSmoother(
    period=20, 
    src_type='ukf_close', 
    ukf_params=ukf_params
)

result_ukf = smoother_ukf.calculate(data)
print(f"UKF版: {result_ukf.values[-1]:.4f}")
```

## パラメータの詳細

### period (float)
- **デフォルト**: 20.0
- **説明**: フィルターの臨界期間
- **推奨範囲**: 10 - 50
- **効果**: 
  - 小さい値: より敏感、より小さなラグ
  - 大きい値: よりスムース、より大きなラグ

### src_type (str)
- **デフォルト**: 'close'
- **基本ソース**: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
- **UKFソース**: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'

### ukf_params (dict, optional)
UKFソース使用時のパラメータ：
- **alpha**: UKFのalpha値（デフォルト: 0.001）
- **process_noise_scale**: プロセスノイズスケール（デフォルト: 0.001）
- **volatility_window**: ボラティリティ計算ウィンドウ（デフォルト: 10）

## 実践的な戦略例

### 1. トレンドフォロー戦略

```python
# トレンドフォロー戦略の例
def trend_follow_strategy(data, period=20):
    smoother = UltimateSmoother(period=period, src_type='close')
    result = smoother.calculate(data)
    
    # 現在の価格とスムーザー値を比較
    current_price = data['close'].iloc[-1]
    current_smoother = result.values[-1]
    
    if current_price > current_smoother:
        return "買いシグナル"
    elif current_price < current_smoother:
        return "売りシグナル"
    else:
        return "ホールド"
```

### 2. 複数時間軸分析

```python
# 複数時間軸でのアルティメットスムーザー
def multi_timeframe_analysis(data):
    periods = [10, 20, 50]
    signals = {}
    
    for period in periods:
        smoother = UltimateSmoother(period=period, src_type='close')
        result = smoother.calculate(data)
        
        # 傾きの計算
        slope = result.values[-1] - result.values[-2]
        signals[f'period_{period}'] = 'up' if slope > 0 else 'down'
    
    return signals
```

### 3. 他のフィルターとの比較

```python
# パフォーマンス比較
def compare_smoothers(data, period=20):
    # Ultimate Smoother
    us_smoother = UltimateSmoother(period=period, src_type='close')
    us_result = us_smoother.calculate(data)
    
    # EMA（比較用）
    alpha = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data['close'].iloc[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data['close'].iloc[i] + (1 - alpha) * ema[i-1]
    
    # 相関係数の計算
    us_corr = np.corrcoef(us_result.values, data['close'])[0, 1]
    ema_corr = np.corrcoef(ema, data['close'])[0, 1]
    
    return {
        'ultimate_smoother_correlation': us_corr,
        'ema_correlation': ema_corr,
        'ultimate_smoother_advantage': us_corr - ema_corr
    }
```

## 結果の解釈

### UltimateSmootherResult オブジェクト

```python
# 結果オブジェクトの内容
result = smoother.calculate(data)

# 利用可能な属性
print(f"Ultimate Smoother値: {result.values}")
print(f"係数: {result.coefficients}")
```

### 値の解釈

- **Ultimate Smoother値**: フィルタリングされた価格値
- **係数**: 使用された係数（デバッグ用）

## 他のスムーザーとの比較

| 特徴 | Ultimate Smoother | EMA | SuperSmoother |
|------|------------------|-----|---------------|
| ラグ | 最小 | 中程度 | 小 |
| スムージング | 優秀 | 良好 | 優秀 |
| ノイズ除去 | 最優秀 | 良好 | 優秀 |
| 応答性 | 最優秀 | 良好 | 優秀 |
| 計算コスト | 中程度 | 最小 | 中程度 |

## 注意事項

1. **最小データポイント**: 最低4点以上のデータが必要
2. **期間設定**: 期間が短すぎると不安定になる可能性
3. **UKFパラメータ**: UKFソースを使用する場合、適切なパラメータ調整が必要
4. **キャッシュ**: 同じデータで複数回計算する場合、結果がキャッシュされる

## トラブルシューティング

### よくある問題

1. **データが短すぎる**: 最低4点以上のデータを提供
2. **期間が不適切**: 期間は2以上に設定
3. **UKFエラー**: UKFパラメータの調整または基本ソースの使用

### ログ出力

```python
import logging

# ログレベルの設定
logging.basicConfig(level=logging.INFO)

# 計算実行
result = smoother.calculate(data)
```

これでアルティメットスムーザーを効果的に使用できます。独立したアルゴリズムとして、優れたスムージング性能を提供します。 