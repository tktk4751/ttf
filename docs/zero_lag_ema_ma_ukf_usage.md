# 🎯 Zero Lag EMA with Market-Adaptive UKF 使用ガイド

## 概要

Zero Lag EMA with Market-Adaptive UKF (MA-UKF) は、HLC3価格にMA-UKFフィルタリングを適用し、その後ゼロラグEMAを計算する革新的なインジケーターです。

### 特徴

- **2段階フィルタリング**: MA-UKF → Zero Lag EMA
- **市場適応性**: 動的な市場レジーム検出とパラメータ調整
- **低遅延**: Zero Lag EMAによる遅延の最小化
- **高精度**: Numba最適化による高速計算
- **包括的出力**: トレンド信号、市場レジーム、信頼度スコア

## クイックスタート

### 1. 基本テスト実行

```bash
# ダミーデータを使用した基本テスト
python examples/zero_lag_ema_ma_ukf_test.py --test-basic
```

### 2. 全テスト実行

```bash
# 全テストスイートを実行
python examples/zero_lag_ema_ma_ukf_test.py
```

### 3. 設定ファイル作成

```bash
# サンプル設定ファイルを作成
python examples/zero_lag_ema_ma_ukf_test.py --create-config
```

### 4. 実データでテスト

```bash
# 設定ファイルを使用して実データでテスト
python examples/zero_lag_ema_ma_ukf_test.py --config sample_config.yaml
```

## チャート描画

### 基本的な使用方法

```python
from visualization.zero_lag_ema_ma_ukf_chart import ZeroLagEMAMAUKFChart

# チャートオブジェクト作成
chart = ZeroLagEMAMAUKFChart()

# データ読み込み（設定ファイルから）
chart.load_data_from_config('config.yaml')

# インジケーター計算
chart.calculate_indicators(
    ema_period=14,
    lag_adjustment=2.0,
    slope_period=1,
    range_threshold=0.003
)

# チャート描画
chart.plot(
    title="Zero Lag EMA with MA-UKF",
    show_volume=True,
    savefig='output/chart.png'
)
```

### ダミーデータでの使用

```python
# ダミーデータ生成
dummy_data = chart.generate_dummy_data(n_periods=200)
chart.data = dummy_data

# 以降は同じ
chart.calculate_indicators()
chart.plot()
```

## コマンドライン使用例

### パラメータ調整テスト

```bash
# EMA期間と遅延調整を変更
python examples/zero_lag_ema_ma_ukf_test.py \
    --ema-period 21 \
    --lag-adjustment 1.5 \
    --test-basic
```

### 期間指定チャート

```bash
# 特定期間のチャートを生成
python examples/zero_lag_ema_ma_ukf_test.py \
    --config config.yaml \
    --start 2023-06-01 \
    --end 2023-12-31 \
    --output output/period_chart.png
```

### 個別テスト実行

```bash
# パラメータ比較テストのみ
python examples/zero_lag_ema_ma_ukf_test.py --test-params

# パフォーマンステストのみ
python examples/zero_lag_ema_ma_ukf_test.py --test-performance

# 堅牢性テストのみ
python examples/zero_lag_ema_ma_ukf_test.py --test-robustness
```

## プログラマティック使用

### インジケーター単体使用

```python
from indicators.zero_lag_ema_ma_ukf import ZeroLagEMAWithMAUKF
import pandas as pd

# インジケーター作成
zero_lag_ema = ZeroLagEMAWithMAUKF(
    ema_period=14,
    lag_adjustment=2.0,
    slope_period=1,
    range_threshold=0.003,
    # MA-UKFパラメータ
    ukf_alpha=0.001,
    ukf_beta=2.0,
    ukf_kappa=0.0
)

# データ準備（OHLCV形式のDataFrame）
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 計算実行
result = zero_lag_ema.calculate(data)

# 結果取得
zero_lag_values = result.values                    # Zero Lag EMA値
regular_ema_values = result.ema_values             # 通常のEMA値
filtered_hlc3 = result.filtered_source             # MA-UKFフィルタリング済みHLC3
raw_hlc3 = result.raw_source                       # 元のHLC3
trend_signals = result.trend_signals               # トレンド信号 (1, 0, -1)
market_regimes = result.market_regimes             # 市場レジーム状態
confidence_scores = result.confidence_scores       # MA-UKF信頼度
```

## パラメータガイド

### Zero Lag EMAパラメータ

- **ema_period** (デフォルト: 14)
  - EMAの計算期間
  - 小さい値: より敏感、多くのシグナル
  - 大きい値: より滑らか、少ないシグナル

- **lag_adjustment** (デフォルト: 2.0)
  - 遅延調整の強度
  - 小さい値: 保守的、遅延残存
  - 大きい値: 積極的、オーバーシュートリスク

### トレンド判定パラメータ

- **slope_period** (デフォルト: 1)
  - トレンド判定の期間
  - 通常は1で良好

- **range_threshold** (デフォルト: 0.003)
  - レンジ/トレンド判定の閾値
  - 小さい値: より敏感なトレンド検出
  - 大きい値: よりレンジ寄りの判定

### MA-UKFパラメータ

- **ukf_alpha** (デフォルト: 0.001)
  - シグマポイントの広がり
  - 通常は小さい値（0.0001-0.01）

- **ukf_beta** (デフォルト: 2.0)
  - 分布の尖度パラメータ
  - ガウス分布では2.0が最適

- **ukf_kappa** (デフォルト: 0.0)
  - 追加のスケーリングパラメータ
  - 通常は0で問題なし

## 出力の解釈

### チャート要素

1. **メインチャート**
   - ローソク足: 元の価格データ
   - 水色線: MA-UKFフィルタリング済みHLC3
   - 青線: 通常のEMA
   - 緑線: Zero Lag EMA（上昇トレンド）
   - 赤線: Zero Lag EMA（下降トレンド）
   - 灰線: Zero Lag EMA（レンジ相場）

2. **トレンド信号パネル**
   - +1: 上昇トレンド
   - 0: レンジ相場
   - -1: 下降トレンド

3. **市場レジームパネル**
   - +0.5以上: トレンド市場
   - -0.3～+0.3: レンジ市場
   - -0.5以下: 下降トレンド市場

4. **信頼度パネル**
   - 0.8以上: 高信頼度
   - 0.6-0.8: 中信頼度
   - 0.4-0.6: 低信頼度
   - 0.4未満: 非常に低い信頼度

### 統計情報

テスト実行時に表示される統計情報：

- **データ統計**: 有効データ数、価格範囲
- **トレンド分析**: 各トレンド状態の期間と比率
- **市場レジーム分析**: レジーム値の統計、トレンド/レンジ市場の比率
- **MA-UKF信頼度**: 平均信頼度、高/低信頼度の比率
- **フィルタリング効果**: ノイズ除去率、変動性の比較

## トラブルシューティング

### よくあるエラー

1. **インポートエラー**
   ```bash
   # プロジェクトルートから実行
   cd /path/to/ttf
   python examples/zero_lag_ema_ma_ukf_test.py
   ```

2. **データ読み込みエラー**
   ```bash
   # ダミーデータでテスト
   python examples/zero_lag_ema_ma_ukf_test.py --test-basic
   ```

3. **メモリエラー**
   ```bash
   # 小さいデータサイズでテスト
   python examples/zero_lag_ema_ma_ukf_test.py --test-performance
   ```

### デバッグモード

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# より詳細なログが出力されます
```

## 応用例

### カスタムパラメータでの最適化

```python
# 短期取引向け設定
short_term = ZeroLagEMAWithMAUKF(
    ema_period=8,
    lag_adjustment=3.0,
    range_threshold=0.001
)

# 長期取引向け設定
long_term = ZeroLagEMAWithMAUKF(
    ema_period=21,
    lag_adjustment=1.5,
    range_threshold=0.005
)
```

### 複数時間軸での分析

```python
# 1時間足と4時間足での比較分析
timeframes = ['1h', '4h']
for tf in timeframes:
    # データ読み込み（時間軸別）
    data = load_data_for_timeframe(tf)
    
    # 計算実行
    result = zero_lag_ema.calculate(data)
    
    # 結果の保存・分析
    analyze_results(result, tf)
```

## 参考情報

- **元の実装**: `indicators/zero_lag_ema_ma_ukf.py`
- **チャート実装**: `visualization/zero_lag_ema_ma_ukf_chart.py`
- **テストコード**: `examples/zero_lag_ema_ma_ukf_test.py`
- **MA-UKF統合**: `indicators/kalman_filter_unified.py`

## 今後の拡張

- **マルチアセット対応**: 複数銘柄での同時分析
- **アラート機能**: トレンド変化時の通知
- **バックテスト統合**: 戦略パフォーマンスの検証
- **ウェブダッシュボード**: リアルタイム監視 