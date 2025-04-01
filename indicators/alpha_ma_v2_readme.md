# AlphaMAV2 インジケーター

## 概要

AlphaMAV2は、RSXの優れた平滑化アルゴリズムとAlphaMAの適応型フレームワークを組み合わせた新世代の移動平均線です。効率比（Efficiency Ratio）に基づいて平滑化期間を動的に調整し、RSXで使用されている3段階の平滑化関数を適用することで、非常にスムーズかつ正確な価格トレンドの追跡を実現します。

## 特徴

- **RSXの3段階平滑化アルゴリズム**：
  - 高度なノイズ除去能力
  - よりスムーズな価格トレンドの表現
  - 遅延を最小限に抑えた反応性

- **適応的な期間設定**：
  - 市場の効率性に基づいて平滑化期間を自動調整
  - トレンドが強い時：短い期間（より反応的）
  - レンジ相場時：長い期間（よりスムーズ）

- **使いやすさ**：
  - シンプルなAPIと包括的なドキュメント
  - 効率的なNumba最適化による高速な計算
  - 柔軟なパラメータ設定

## 技術的な仕組み

AlphaMAV2は以下の技術を組み合わせています：

1. **効率比（ER）の計算**：
   - 価格変動の効率性を測定
   - 0～1の範囲（1に近いほど効率的なトレンド、0に近いほど非効率なレンジ）

2. **動的期間の計算**：
   - ERの値に基づいて平滑化期間を調整
   - ERが高い → 短い期間（最小5）
   - ERが低い → 長い期間（最大34）

3. **RSXの3段階平滑化**：
   ```
   Stage 1: f28 = f20 * f28_prev + f18 * data
            f30 = f18 * f28 + f20 * f30_prev
            smoothed_1 = f28 * 1.5 - f30 * 0.5
   
   Stage 2: f38 = f20 * f38_prev + f18 * smoothed_1
            f40 = f18 * f38 + f20 * f40_prev
            smoothed_2 = f38 * 1.5 - f40 * 0.5
   
   Stage 3: f48 = f20 * f48_prev + f18 * smoothed_2
            f50 = f18 * f48 + f20 * f50_prev
            final = f48 * 1.5 - f50 * 0.5
   ```
   ここで、`f18 = 3.0 / (length + 2.0)`、`f20 = 1.0 - f18`

## 使用方法

### 基本的な使用例

```python
from indicators import AlphaMAV2
import pandas as pd

# データの読み込み
data = pd.read_csv('your_data.csv')

# AlphaMAV2の作成（デフォルト設定）
alpha_ma_v2 = AlphaMAV2()

# 計算
values = alpha_ma_v2.calculate(data)

# 効率比の取得
er_values = alpha_ma_v2.get_efficiency_ratio()

# 動的期間の取得
dynamic_periods = alpha_ma_v2.get_dynamic_period()

# シグナルラインの取得（オフセット1）
signal = alpha_ma_v2.get_signal_line(offset=1)

# クロスオーバー/クロスアンダーシグナルの取得
crossover, crossunder = alpha_ma_v2.get_crossover_signals()
```

### カスタム設定

```python
# カスタムパラメータ設定
alpha_ma_v2 = AlphaMAV2(
    er_period=21,    # 効率比の計算期間
    max_period=55,   # 平滑化期間の最大値
    min_period=3     # 平滑化期間の最小値
)
```

## トレード戦略の例

### トレンドフォロー戦略

```python
# シグナルラインとのクロスオーバー/クロスアンダーを使用したトレンドフォロー戦略
crossover, crossunder = alpha_ma_v2.get_crossover_signals(offset=3)

# クロスオーバー（上昇トレンドの開始）時に買い
buy_signals = crossover

# クロスアンダー（下降トレンドの開始）時に売り
sell_signals = crossunder
```

### マルチタイムフレーム戦略

```python
# 異なる期間設定でAlphaMAV2を作成
fast_ma = AlphaMAV2(er_period=8, max_period=21, min_period=3)
slow_ma = AlphaMAV2(er_period=21, max_period=55, min_period=13)

# 両方を計算
fast_values = fast_ma.calculate(data)
slow_values = slow_ma.calculate(data)

# FastがSlowを上回る（上昇トレンド）
buy_signals = np.where(fast_values > slow_values, 1, 0)

# FastがSlowを下回る（下降トレンド）
sell_signals = np.where(fast_values < slow_values, 1, 0)
```

## AlphaMA（旧バージョン）との比較

AlphaMAV2は、AlphaMAに比べて以下の点が改善されています：

| 特徴 | AlphaMAV2 | AlphaMA |
|------|-----------|---------|
| 平滑化アルゴリズム | RSXの3段階平滑化 | KAMAの単一平滑化 |
| パラメータ | シンプル（3つのみ） | 複雑（7つ） |
| ノイズ除去 | 優れている | 良好 |
| 反応速度 | 速い | 中程度 |
| 計算効率 | 高い | 中程度 |

## 推奨パラメータ設定

| 用途 | er_period | max_period | min_period |
|------|-----------|------------|------------|
| デイトレード | 5-8 | 21-34 | 3-5 |
| スイングトレード | 10-13 | 34-55 | 5-8 |
| 長期投資 | 21 | 55-89 | 8-13 |
| ノイズの多い市場 | 13-21 | 55-89 | 5-8 |
| トレンドの強い市場 | 5-10 | 34-55 | 3-5 | 