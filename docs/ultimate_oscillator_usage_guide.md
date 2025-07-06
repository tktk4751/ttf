# Ultimate Oscillator 使用ガイド

John Ehlersの論文「Ultimate Oscillator」（Traders Tips 4/2025）に基づくアルティメットオシレーターの実装と使用方法を説明します。

## 概要

アルティメットオシレーターは、John Ehlersが開発した高精度なオシレーター指標です。2つの異なる期間のハイパスフィルターの差分を取り、RMSで正規化することで、市場サイクルの変化を検出します。

### 主な特徴

- **第3次ハイパスフィルター**: 高精度なノイズ除去
- **差分信号検出**: 異なる時間軸の信号差分による精密分析
- **RMS正規化**: 安定した振動範囲での信号提供
- **サイクル検出**: 市場サイクルの変化を高感度で検出
- **完全統合**: プロジェクトアーキテクチャとの完全統合

## 数学的基礎

アルティメットオシレーターは以下の数学的定義に基づいて計算されます：

### 1. ハイパスフィルター3次（HighPass3）

```
a1 = exp(-1.414 * π / Period)
c2 = 2 * a1 * cos(1.414 * π / 2 / Period)  
c3 = -a1 * a1
c1 = (1 + c2 - c3) / 4
```

**ハイパスフィルター式**:
```
HP[i] = c1 * (Data[i] - 2*Data[i-1] + Data[i-2]) + c2 * HP[i-1] + c3 * HP[i-2]
```

### 2. 信号差分計算

```
Signals[i] = HighPass3(Data, Width*Edge)[i] - HighPass3(Data, Edge)[i]
```

### 3. RMS正規化

```
RMS[i] = sqrt(sum(Signals[j]^2 for j in [i-period+1, i]) / period)
UltimateOsc[i] = Signals[i] / RMS[i]
```

## 基本的な使用方法

### 1. 基本的な使用例

```python
from indicators.ultimate_oscillator import UltimateOscillator
import pandas as pd

# データフレームの準備
data = pd.DataFrame({
    'open': [100, 102, 101, 103, 105],
    'high': [102, 104, 103, 105, 107],
    'low': [99, 101, 100, 102, 104],
    'close': [101, 103, 102, 104, 106]
})

# Ultimate Oscillator の作成
oscillator = UltimateOscillator(
    edge=30,           # エッジ期間
    width=2,           # 幅倍数
    rms_period=100,    # RMS計算期間
    src_type='close'   # ソースタイプ
)

# 計算実行
result = oscillator.calculate(data)

# 結果の取得
ultimate_values = result.values
signals = result.signals
rms_values = result.rms_values

print(f"Ultimate Oscillator値: {ultimate_values}")
print(f"信号値: {signals}")
print(f"RMS値: {rms_values}")
```

### 2. 異なるソースタイプでの使用

```python
# 異なる価格ソースでの使用
oscillators = {
    'close': UltimateOscillator(edge=30, width=2, src_type='close'),
    'hlc3': UltimateOscillator(edge=30, width=2, src_type='hlc3'),
    'hl2': UltimateOscillator(edge=30, width=2, src_type='hl2'),
    'ohlc4': UltimateOscillator(edge=30, width=2, src_type='ohlc4')
}

for name, oscillator in oscillators.items():
    result = oscillator.calculate(data)
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
oscillator_ukf = UltimateOscillator(
    edge=30, 
    width=2, 
    rms_period=100,
    src_type='ukf_close', 
    ukf_params=ukf_params
)

result_ukf = oscillator_ukf.calculate(data)
print(f"UKF版: {result_ukf.values[-1]:.4f}")
```

## パラメータの詳細

### edge (int)
- **デフォルト**: 30
- **説明**: 短期ハイパスフィルターの期間
- **推奨範囲**: 15 - 50
- **効果**:
  - 小さい値: より敏感、高頻度の信号
  - 大きい値: より滑らか、低頻度の信号

### width (int)
- **デフォルト**: 2
- **説明**: 長期フィルター期間の倍数（長期期間 = edge * width）
- **推奨範囲**: 1.5 - 4
- **効果**:
  - 小さい値: より細かい差分検出
  - 大きい値: より大きなサイクル差分検出

### rms_period (int)
- **デフォルト**: 100
- **説明**: RMS正規化の計算期間
- **推奨範囲**: 50 - 200
- **効果**:
  - 小さい値: より敏感な正規化
  - 大きい値: より安定した正規化

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

### 1. ゼロクロス戦略

```python
def zero_cross_strategy(data, edge=30, width=2):
    """ゼロクロス戦略の例"""
    oscillator = UltimateOscillator(edge=edge, width=width, src_type='close')
    result = oscillator.calculate(data)
    
    values = result.values
    signals = []
    
    for i in range(1, len(values)):
        if not (np.isnan(values[i]) or np.isnan(values[i-1])):
            # 買いシグナル（負から正へのクロス）
            if values[i-1] < 0 and values[i] > 0:
                signals.append(('BUY', i))
            # 売りシグナル（正から負へのクロス）
            elif values[i-1] > 0 and values[i] < 0:
                signals.append(('SELL', i))
    
    return signals
```

### 2. 極値検出戦略

```python
def extremes_strategy(data, edge=25, width=2, threshold=2.0):
    """極値検出戦略の例"""
    oscillator = UltimateOscillator(edge=edge, width=width, src_type='close')
    result = oscillator.calculate(data)
    
    values = result.values
    signals = []
    
    for i in range(len(values)):
        if not np.isnan(values[i]):
            # 買いシグナル（oversold から回復）
            if values[i] < -threshold:
                # 次の期間で上昇したら買い
                if i < len(values) - 1 and values[i+1] > values[i]:
                    signals.append(('BUY', i+1))
            
            # 売りシグナル（overbought から下降）
            elif values[i] > threshold:
                # 次の期間で下降したら売り
                if i < len(values) - 1 and values[i+1] < values[i]:
                    signals.append(('SELL', i+1))
    
    return signals
```

### 3. トレンド確認戦略

```python
def trend_confirmation_strategy(data, edge=30, width=2):
    """トレンド確認戦略の例"""
    oscillator = UltimateOscillator(edge=edge, width=width, src_type='close')
    result = oscillator.calculate(data)
    
    values = result.values
    prices = data['close'].values
    signals = []
    
    # 5期間の移動平均でトレンド判定
    ma_period = 5
    price_ma = np.convolve(prices, np.ones(ma_period)/ma_period, mode='valid')
    
    # オシレーターとトレンドの組み合わせ
    for i in range(ma_period, len(values)):
        if not np.isnan(values[i]):
            trend_up = prices[i] > price_ma[i - ma_period]
            trend_down = prices[i] < price_ma[i - ma_period]
            
            # 上昇トレンド中のoversold買い
            if trend_up and values[i] < -1.0 and i > 0 and values[i] > values[i-1]:
                signals.append(('BUY', i))
            
            # 下降トレンド中のoverbought売り
            elif trend_down and values[i] > 1.0 and i > 0 and values[i] < values[i-1]:
                signals.append(('SELL', i))
    
    return signals
```

### 4. 複数時間軸分析

```python
def multi_timeframe_analysis(data):
    """複数時間軸でのアルティメットオシレーター分析"""
    timeframes = [
        {'edge': 15, 'width': 2, 'name': 'short'},
        {'edge': 30, 'width': 2, 'name': 'medium'},
        {'edge': 50, 'width': 3, 'name': 'long'}
    ]
    
    results = {}
    
    for tf in timeframes:
        oscillator = UltimateOscillator(
            edge=tf['edge'], 
            width=tf['width'], 
            src_type='close'
        )
        result = oscillator.calculate(data)
        results[tf['name']] = result.values[-1]
    
    # 統合分析
    if all(val > 0 for val in results.values()):
        return "強気（全時間軸で正）"
    elif all(val < 0 for val in results.values()):
        return "弱気（全時間軸で負）"
    else:
        return "混合（時間軸で分散）"
```

## 結果の解釈

### UltimateOscillatorResult オブジェクト

```python
# 結果オブジェクトの内容
result = oscillator.calculate(data)

# 利用可能な属性
print(f"Ultimate Oscillator値: {result.values}")        # メインの指標値
print(f"信号値: {result.signals}")                      # 差分信号
print(f"RMS値: {result.rms_values}")                    # 正規化係数
print(f"短期HP: {result.highpass_short}")               # 短期ハイパスフィルター
print(f"長期HP: {result.highpass_long}")                # 長期ハイパスフィルター
```

### 値の解釈

- **Ultimate Oscillator値**: -3から+3程度の範囲で振動
  - **正の値**: 長期フィルターが短期フィルターより強い（上昇圧力）
  - **負の値**: 短期フィルターが長期フィルターより強い（下降圧力）
  - **ゼロ近辺**: バランス状態
  
- **閾値の目安**:
  - **+2以上**: Overbought（売られすぎ）
  - **-2以下**: Oversold（買われすぎ）
  - **ゼロクロス**: トレンド転換の可能性

## 他のオシレーターとの比較

| 特徴 | Ultimate Oscillator | RSI | MACD | Stochastic |
|------|-------------------|-----|------|------------|
| ノイズ除去 | 最優秀 | 良好 | 中程度 | 中程度 |
| サイクル検出 | 最優秀 | 中程度 | 良好 | 中程度 |
| 感度調整 | 高度 | 基本 | 中程度 | 基本 |
| 正規化 | RMS | 0-100 | 価格ベース | 0-100 |
| 計算複雑度 | 高 | 低 | 中 | 低 |

## 注意事項

1. **最小データポイント**: 最低 max(edge*width, rms_period) + 10点以上のデータが必要
2. **パラメータ調整**: 市場特性に応じたパラメータ調整が重要
3. **UKFパラメータ**: UKFソースを使用する場合、適切なパラメータ調整が必要
4. **キャッシュ**: 同じデータで複数回計算する場合、結果がキャッシュされる

## トラブルシューティング

### よくある問題

1. **データが短すぎる**: 十分なデータポイントを提供
2. **パラメータが不適切**: edge > 0, width > 0, rms_period > 0
3. **UKFエラー**: UKFパラメータの調整または基本ソースの使用

### ログ出力

```python
import logging

# ログレベルの設定
logging.basicConfig(level=logging.INFO)

# 計算実行
result = oscillator.calculate(data)
```

## 実用例とパフォーマンス

### パラメータ感度分析

```python
def parameter_sensitivity_analysis(data):
    """パラメータ感度分析"""
    edge_values = [15, 20, 25, 30, 35, 40]
    results = {}
    
    for edge in edge_values:
        oscillator = UltimateOscillator(edge=edge, width=2, rms_period=100)
        result = oscillator.calculate(data)
        
        valid_values = result.values[~np.isnan(result.values)]
        results[edge] = {
            'std': np.std(valid_values),
            'range': np.max(valid_values) - np.min(valid_values),
            'zero_crosses': np.sum(np.abs(np.diff(np.sign(valid_values))) > 0)
        }
    
    return results
```

### 最適なパラメータの選択

- **短期取引**: edge=15-25, width=2, rms_period=50-100
- **中期取引**: edge=25-35, width=2-3, rms_period=100-150
- **長期取引**: edge=35-50, width=3-4, rms_period=150-200

これでアルティメットオシレーターを効果的に使用できます。高精度なサイクル検出と信号生成を活用して、優れた取引戦略を構築してください！ 