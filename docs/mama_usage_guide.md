# MAMA/FAMA インジケーター使用ガイド

## 概要

MAMA (Mother of Adaptive Moving Average) と FAMA (Following Adaptive Moving Average) は、John Ehlers によって開発された適応型移動平均インジケーターです。市場のサイクルに応じて自動的に期間を調整し、トレンドの変化を効率的に捉えます。

## 特徴

- **適応型**: 市場サイクルの変化に応じて自動的に応答速度を調整
- **ノイズフィルタリング**: 高品質なトレンド信号を提供
- **MESA アルゴリズム**: Maximum Entropy Spectrum Analysis に基づく
- **高速計算**: Numba による最適化で高速処理

## インストールと使用

### 基本的な使用方法

```python
import pandas as pd
from indicators.mama import MAMA

# 価格データの準備
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...]
})

# MAMAインジケーターの作成
mama = MAMA(fast_limit=0.5, slow_limit=0.05, src_type='hl2')

# 計算実行
result = mama.calculate(data)

# 結果の取得
mama_values = result.mama_values  # MAMA値
fama_values = result.fama_values  # FAMA値
period_values = result.period_values  # 適応期間
alpha_values = result.alpha_values  # スムージング係数
```

### パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `fast_limit` | 0.5 | 高速制限値（0-1の範囲） |
| `slow_limit` | 0.05 | 低速制限値（0-1の範囲） |
| `src_type` | 'hl2' | 価格ソースタイプ |

### ソースタイプ

- **'hl2'**: (高値 + 安値) / 2 （推奨、EasyLanguage準拠）
- **'close'**: 終値
- **'hlc3'**: (高値 + 安値 + 終値) / 3
- **'ohlc4'**: (始値 + 高値 + 安値 + 終値) / 4

## 結果の解釈

### MAMA と FAMA の関係

- **MAMA > FAMA**: 上昇トレンド
- **MAMA < FAMA**: 下降トレンド
- **クロスオーバー**: トレンド転換の可能性

### 適応期間 (Period)

- **短い期間**: 市場の変動が大きい
- **長い期間**: 市場が安定している
- **範囲**: 6-50 の間で自動調整

### Alpha 値

- **高い値**: 素早い応答（ノイズに敏感）
- **低い値**: 遅い応答（より安定）
- **範囲**: slow_limit から fast_limit の間で自動調整

## 高度な使用方法

### 複数期間での比較

```python
# 異なるパラメータでの比較
mama_fast = MAMA(fast_limit=0.7, slow_limit=0.02)
mama_slow = MAMA(fast_limit=0.3, slow_limit=0.1)

result_fast = mama_fast.calculate(data)
result_slow = mama_slow.calculate(data)
```

### InPhase/Quadrature 成分の取得

```python
# 内部計算値の取得
i1_values, q1_values = mama.get_inphase_quadrature()
phase_values = mama.get_phase_values()
```

### シグナル生成例

```python
import numpy as np

def generate_mama_signals(mama_values, fama_values):
    """MAMA/FAMAクロスオーバーシグナルを生成"""
    signals = np.zeros(len(mama_values))
    
    for i in range(1, len(mama_values)):
        # ゴールデンクロス (買いシグナル)
        if (mama_values[i] > fama_values[i] and 
            mama_values[i-1] <= fama_values[i-1]):
            signals[i] = 1
        
        # デッドクロス (売りシグナル)
        elif (mama_values[i] < fama_values[i] and 
              mama_values[i-1] >= fama_values[i-1]):
            signals[i] = -1
    
    return signals
```

## 可視化

```python
import matplotlib.pyplot as plt

def plot_mama_fama(data, result):
    """MAMA/FAMAチャートを作成"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 価格とMAMA/FAMA
    ax1.plot(data['close'], label='Close Price', alpha=0.7)
    ax1.plot(result.mama_values, label='MAMA', linewidth=2)
    ax1.plot(result.fama_values, label='FAMA', linewidth=2)
    ax1.set_title('MAMA/FAMA Adaptive Moving Averages')
    ax1.legend()
    ax1.grid(True)
    
    # 適応期間
    ax2.plot(result.period_values, label='Adaptive Period', color='green')
    ax2.set_title('Adaptive Period')
    ax2.set_ylabel('Period')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## トレーディング戦略例

### 基本的なクロスオーバー戦略

```python
def mama_crossover_strategy(data, fast_limit=0.5, slow_limit=0.05):
    """MAMAクロスオーバー戦略"""
    mama = MAMA(fast_limit=fast_limit, slow_limit=slow_limit)
    result = mama.calculate(data)
    
    # シグナル生成
    signals = generate_mama_signals(result.mama_values, result.fama_values)
    
    # パフォーマンス計算
    positions = np.zeros(len(data))
    current_position = 0
    
    for i in range(len(signals)):
        if signals[i] == 1:  # 買いシグナル
            current_position = 1
        elif signals[i] == -1:  # 売りシグナル
            current_position = 0
        
        positions[i] = current_position
    
    return positions, signals, result
```

### アダプティブ期間フィルター戦略

```python
def adaptive_period_filter_strategy(data, period_threshold=15):
    """適応期間に基づくフィルター戦略"""
    mama = MAMA()
    result = mama.calculate(data)
    
    # 短期間の時のみトレード（高ボラティリティ時）
    valid_trades = result.period_values < period_threshold
    
    signals = generate_mama_signals(result.mama_values, result.fama_values)
    filtered_signals = signals * valid_trades
    
    return filtered_signals, result
```

## パフォーマンス最適化

### キャッシュの活用

```python
# 同じデータでの複数計算時はキャッシュが自動的に使用される
mama = MAMA()
result1 = mama.calculate(data)  # 初回計算
result2 = mama.calculate(data)  # キャッシュから高速取得
```

### バッチ処理

```python
def process_multiple_symbols(symbol_data_dict):
    """複数銘柄の一括処理"""
    mama = MAMA()
    results = {}
    
    for symbol, data in symbol_data_dict.items():
        results[symbol] = mama.calculate(data)
        mama.reset()  # 次の銘柄のためにリセット
    
    return results
```

## 注意事項

1. **データ長**: 最低10点以上のデータを推奨（理想的には20点以上）
2. **初期値**: 最初の数値は安定するまで時間がかかる
3. **パラメータ調整**: `fast_limit` > `slow_limit` の関係を維持
4. **市場特性**: 異なる市場・時間枠で最適なパラメータは異なる

## トラブルシューティング

### よくある問題

**Q: 結果がすべてNaNになる**
A: データが不足している可能性があります。最低10点以上のデータを用意してください。

**Q: MAMAとFAMAの差が小さすぎる**
A: `fast_limit`を大きくするか、`slow_limit`を小さくしてください。

**Q: シグナルが多すぎる/少なすぎる**
A: 適応期間フィルターを使用するか、パラメータを調整してください。

## 参考文献

- John Ehlers, "Rocket Science for Traders"
- John Ehlers, "Cybernetic Analysis for Stocks and Futures"
- MESA (Maximum Entropy Spectrum Analysis) アルゴリズム

## サンプルコード

完全なサンプルコードは `examples/mama_example.py` を参照してください。 