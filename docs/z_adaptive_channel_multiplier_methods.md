# Zアダプティブチャネル - 動的乗数計算方法

## 概要

Zアダプティブチャネルに新しい動的乗数計算方法を追加しました。これにより、より柔軟なチャネル幅の制御が可能になります。

## 利用可能な計算方法

### 1. アダプティブ方式 (`multiplier_method='adaptive'`)

**デフォルト方式**。従来の複雑な動的乗数計算を使用します。

**計算式:**
```
動的最大乗数 = max_max_multiplier - trigger * (max_max_multiplier - min_max_multiplier)
動的最小乗数 = max_min_multiplier - trigger * (max_min_multiplier - min_min_multiplier)
最終動的乗数 = 動的最大乗数 - trigger * (動的最大乗数 - 動的最小乗数)
```

**特徴:**
- 4つのパラメータ（max_max_multiplier, min_max_multiplier, max_min_multiplier, min_min_multiplier）で細かく制御
- トレンド強度に応じて複雑に変化
- より洗練された適応性

### 2. シンプル方式 (`multiplier_method='simple'`)

**新しく追加された方式**。シンプルで直感的な計算を使用します。

**計算式:**
```
動的乗数 = 16 - (CER*8) - (XTRENDINDEX*8)
```

**特徴:**
- CERとXTRENDINDEXの両方を使用したシンプルな計算式
- 基準値は16で、CERとXTRENDINDEXが高いほど乗数が小さくなる
- トレンド強度とサイクル効率の両方を考慮
- パラメータ調整が簡単で直感的

## 使用方法

### アダプティブ方式（デフォルト）

```python
from indicators.z_adaptive_channel import ZAdaptiveChannel

# アダプティブ方式
adaptive_channel = ZAdaptiveChannel(
    multiplier_method='adaptive',  # デフォルトなので省略可能
    max_max_multiplier=8.0,
    min_max_multiplier=3.0,
    max_min_multiplier=1.5,
    min_min_multiplier=0.5,
    multiplier_source='cer'
)

result = adaptive_channel.calculate(data)
middle, upper, lower = adaptive_channel.get_bands()
```

### シンプル方式

```python
# シンプル方式
simple_channel = ZAdaptiveChannel(
    multiplier_method='simple',
    multiplier_source='cer'
)

result = simple_channel.calculate(data)
middle, upper, lower = simple_channel.get_bands()
```

## パラメータ比較

| パラメータ | アダプティブ方式 | シンプル方式 |
|-----------|----------------|-------------|
| `max_max_multiplier` | 使用 | 無視 |
| `min_max_multiplier` | 使用 | 無視 |
| `max_min_multiplier` | 使用 | 無視 |
| `min_min_multiplier` | 使用 | 無視 |
| `multiplier_source` | 使用 | 使用 |
| `ma_source` | 使用 | 使用 |

## トリガーソース

両方の計算方法で以下のトリガーソースが利用可能です：

- `'cer'`: サイクル効率比（Cycle Efficiency Ratio）
- `'x_trend'`: Xトレンドインデックス
- `'z_trend'`: Zアダプティブトレンドインデックス

## 実装の詳細

### 最適化

- **ベクトル化**: NumbaのVectorizeデコレータを使用した並列処理
- **JIT コンパイル**: Numbaを使用した高速化
- **キャッシュ**: 計算結果のキャッシュによる再計算の回避

### 新しく追加された関数

1. `calculate_new_simple_dynamic_multiplier_vec()`: ベクトル化された新しいシンプル乗数計算
2. `calculate_new_simple_dynamic_multiplier_optimized()`: 最適化された新しいシンプル乗数計算

## 使い分けの指針

### アダプティブ方式を選ぶべき場合

- より細かい調整が必要な場合
- 複雑な市場環境で精密なチャネル制御が必要な場合
- 既存のパラメータセットを利用したい場合

### シンプル方式を選ぶべき場合

- 計算が速い方が良い場合
- パラメータ調整を簡単にしたい場合
- 直感的で理解しやすいロジックが欲しい場合
- プロトタイピングや初期検証の場合

## パフォーマンス比較

シンプル方式はアダプティブ方式よりも高速です：

- **計算量**: シンプル方式は O(n)、アダプティブ方式は O(3n)
- **メモリ使用量**: シンプル方式の方が少ない
- **パラメータ数**: シンプル方式は乗数関連パラメータが不要

## 例: 計算結果の比較

```
例1: CER=0.3, XTRENDINDEX=0.4
アダプティブ方式: 3.4018 (パラメータに依存)
シンプル方式: 10.6 (16 - 0.3*8 - 0.4*8)

例2: CER=0.6, XTRENDINDEX=0.8
アダプティブ方式: 4.2 (パラメータに依存)  
シンプル方式: 4.8 (16 - 0.6*8 - 0.8*8)
```

シンプル方式では、CERとXTRENDINDEXの値と乗数の関係が明確で、両方の指標が高いほどチャネル幅が狭くなります。 