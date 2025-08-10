# 🎯 Hyper Adaptive Supertrend - 最強のスーパートレンドインジケーター

## 概要

Hyper Adaptive Supertrendは、既存のスーパートレンドインジケーターを大幅に進化させた最強版です。従来のHL2（高値・安値平均）をミッドラインとする手法を改革し、unified_smootherによる高度なスムージング手法、unscented_kalman_filterによるノイズ除去、x_atrによる精密なボラティリティ測定を統合しました。

## 🌟 主要改良点

### 1. **高度なミッドライン計算**
- 従来のHL2の代わりに`unified_smoother`を使用
- FRAMA、Super Smoother、Ultimate Smoother、Zero Lag EMA等の多様なスムージング手法
- 動的期間調整によるmarket cycleへの適応

### 2. **カルマンフィルタリング（オプション）**
- `unscented_kalman_filter`によるソース価格のノイズ除去
- シグマポイント法による非線形フィルタリング
- 適応的ノイズ推定機能

### 3. **精密なボラティリティ測定**
- `x_atr`によるAdvanced True Range計算
- ATRとSTR両方の計算方式をサポート
- 統合スムーシングによる高精度測定

### 4. **既存のスーパートレンドロジック**
- 実績のあるスーパートレンドのトレンド判定ロジックを維持
- バンド計算とトレンド方向判定は従来通り

## 📊 処理フロー

```
ソース価格抽出
    ↓
カルマンフィルター（オプション）
    ↓
統合スムーサー（ミッドライン計算）
    ↓
X_ATR計算（ボラティリティ）
    ↓
ミッドライン ± (X_ATR × 乗数) でバンド計算
    ↓
スーパートレンドロジック適用
```

## 🔧 主要パラメータ

### ATR/ボラティリティパラメータ
- `atr_period`: X_ATR期間（デフォルト: 14.0）
- `multiplier`: ATR乗数（デフォルト: 2.0）
- `atr_method`: X_ATRの計算方法（'atr' または 'str'）
- `atr_smoother_type`: X_ATRのスムーサー（デフォルト: 'ultimate_smoother'）

### ミッドライン（統合スムーサー）パラメータ
- `midline_smoother_type`: ミッドラインスムーサータイプ（デフォルト: 'frama'）
- `midline_period`: ミッドライン期間（デフォルト: 21.0）

### ソース価格関連パラメータ
- `src_type`: ソースタイプ（デフォルト: 'hlc3'）
- `enable_kalman`: カルマンフィルター使用フラグ（デフォルト: False）
- `kalman_alpha`: UKFアルファパラメータ（デフォルト: 0.1）
- `kalman_beta`: UKFベータパラメータ（デフォルト: 2.0）
- `kalman_kappa`: UKFカッパパラメータ（デフォルト: 0.0）
- `kalman_process_noise`: UKFプロセスノイズ（デフォルト: 0.01）

### 動的期間調整パラメータ
- `use_dynamic_period`: 動的期間を使用するか（デフォルト: False）
- `cycle_part`: サイクル部分の倍率（デフォルト: 1.0）
- `detector_type`: 検出器タイプ（デフォルト: 'absolute_ultimate'）
- `max_cycle`: 最大サイクル期間（デフォルト: 233）
- `min_cycle`: 最小サイクル期間（デフォルト: 13）

## 💻 使用例

### 基本的な使用法

```python
from indicators.hyper_adaptive_supertrend import HyperAdaptiveSupertrend

# 基本設定
indicator = HyperAdaptiveSupertrend(
    atr_period=14.0,
    multiplier=2.0,
    midline_smoother_type='frama',
    atr_method='atr'
)

result = indicator.calculate(data)
print(f"スーパートレンド値: {result.values}")
print(f"トレンド方向: {result.trend}")  # 1=上昇, -1=下降
```

### アドバンス設定（カルマンフィルター付き）

```python
# カルマンフィルター付きの高精度設定
indicator = HyperAdaptiveSupertrend(
    atr_period=14.0,
    multiplier=2.0,
    midline_smoother_type='ultimate_smoother',
    atr_method='str',
    enable_kalman=True,
    kalman_alpha=0.1,
    src_type='hlc3'
)

result = indicator.calculate(data)
print(f"フィルター効果: 元={result.raw_source}, 後={result.filtered_source}")
```

### 動的期間調整付きフル機能

```python
# 全機能を使用した最強設定
indicator = HyperAdaptiveSupertrend(
    atr_period=14.0,
    multiplier=2.0,
    midline_smoother_type='ultimate_smoother',
    atr_method='str',
    enable_kalman=True,
    use_dynamic_period=True,
    detector_type='absolute_ultimate',
    src_type='hlc3'
)

result = indicator.calculate(data)
midline_periods, atr_periods = indicator.get_dynamic_periods()
print(f"動的期間: ミッドライン={midline_periods}, ATR={atr_periods}")
```

### 便利関数での使用

```python
from indicators.hyper_adaptive_supertrend import calculate_hyper_adaptive_supertrend

# シンプルな関数呼び出し
supertrend_values = calculate_hyper_adaptive_supertrend(
    data,
    atr_period=14.0,
    multiplier=2.0,
    midline_smoother_type='frama',
    enable_kalman=True
)
```

## 📈 結果オブジェクト

`HyperAdaptiveSupertrendResult`には以下の情報が含まれます：

```python
@dataclass
class HyperAdaptiveSupertrendResult:
    values: np.ndarray           # スーパートレンドライン値
    upper_band: np.ndarray       # 上側のバンド価格
    lower_band: np.ndarray       # 下側のバンド価格
    trend: np.ndarray           # トレンド方向（1=上昇、-1=下降）
    midline: np.ndarray         # ミッドライン（統合スムーサー結果）
    atr_values: np.ndarray      # 使用されたX_ATR値
    raw_source: np.ndarray      # 元のソース価格
    filtered_source: np.ndarray # カルマンフィルター後のソース価格
    smoother_type: str          # 使用されたスムーサータイプ
    atr_method: str            # 使用されたATR計算方法
    kalman_enabled: bool       # カルマンフィルター使用フラグ
    parameters: Dict           # 使用されたパラメータ
```

## 🎛️ 利用可能なメソッド

### 基本メソッド
- `get_values()`: スーパートレンドライン値を取得
- `get_supertrend_direction()`: トレンド方向を取得
- `get_upper_band()` / `get_lower_band()`: バンド値を取得
- `get_midline()`: ミッドライン値を取得
- `get_atr_values()`: ATR値を取得

### 高度なメソッド
- `get_raw_source()` / `get_filtered_source()`: ソース価格取得
- `get_dynamic_periods()`: 動的期間情報取得
- `get_metadata()`: インジケーターメタデータ取得
- `reset()`: 状態リセット

## 📋 設定組み合わせ例

### 1. ベーシック設定
```python
HyperAdaptiveSupertrend(
    midline_smoother_type='frama',
    atr_method='atr',
    enable_kalman=False,
    use_dynamic_period=False
)
```

### 2. アドバンス設定
```python
HyperAdaptiveSupertrend(
    midline_smoother_type='ultimate_smoother',
    atr_method='str',
    enable_kalman=False,
    use_dynamic_period=False
)
```

### 3. カルマンフィルター付き
```python
HyperAdaptiveSupertrend(
    midline_smoother_type='frama',
    atr_method='atr',
    enable_kalman=True,
    use_dynamic_period=False
)
```

### 4. フル機能
```python
HyperAdaptiveSupertrend(
    midline_smoother_type='ultimate_smoother',
    atr_method='str',
    enable_kalman=True,
    use_dynamic_period=True
)
```

## ⚡ パフォーマンス特徴

- **Numbaによる高速化**: コア計算部分がJITコンパイルされ高速動作
- **キャッシュ機能**: 同一データの再計算を避ける効率的なキャッシュ
- **メモリ効率**: 必要なデータのみを保持する最適化された設計
- **エラー処理**: 堅牢なエラーハンドリングとフォールバック機能

## 🔍 技術的詳細

### 依存関係
- `indicators.smoother.unified_smoother`: ミッドライン計算
- `indicators.kalman.unscented_kalman_filter`: ノイズ除去
- `indicators.volatility.x_atr`: ボラティリティ測定
- `indicators.price_source`: ソース価格計算

### アルゴリズム
1. **ソース価格抽出**: PriceSourceによる多様な価格計算
2. **カルマンフィルタリング**: UKFによる非線形ノイズ除去
3. **ミッドライン計算**: 統合スムーサーによる適応的スムージング
4. **ボラティリティ測定**: X_ATRによる精密ATR計算
5. **バンド計算**: ミッドライン ± (ATR × 乗数)
6. **トレンド判定**: 従来のスーパートレンドロジック

## 📊 従来のスーパートレンドとの比較

| 項目 | 従来のスーパートレンド | Hyper Adaptive Supertrend |
|------|----------------------|---------------------------|
| ミッドライン | HL2固定 | 統合スムーサー（多種選択可能） |
| ボラティリティ | 基本ATR | X_ATR（ATR/STR両対応） |
| ノイズ除去 | なし | UKFによるカルマンフィルタリング |
| 期間調整 | 固定期間のみ | 動的期間調整対応 |
| スムージング | 単一手法 | 複数のスムージング手法 |
| 適応性 | 限定的 | 高い適応性 |

## 🎯 まとめ

Hyper Adaptive Supertrendは、従来のスーパートレンドの概念を保持しながら、最新の技術を統合した次世代のトレンドフォローイングインジケーターです。多様なスムージング手法、カルマンフィルター、高度なATR計算を組み合わせることで、より正確で適応的なトレンド分析を実現します。

各コンポーネントは独立して設定可能なため、トレーダーの戦略や市場状況に応じて最適な組み合わせを選択できます。