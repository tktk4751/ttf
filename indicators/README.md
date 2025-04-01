# KernelMA 高速化ガイド

## 概要

KernelMA（カーネル移動平均線）は非常に強力なインジケーターですが、計算コストが高いという欠点があります。このドキュメントでは、KernelMAの計算を高速化するための方法を説明します。

## 高速化の手法

KernelMAの計算速度を向上させるために、以下の最適化が実装されています：

1. **Numba JIT コンパイル**: 主要な計算関数をJITコンパイルして実行速度を向上
2. **並列処理**: マルチコアCPUを活用するための並列計算
3. **GPU加速**: CUDAとCuPyを使用したGPU計算（対応ハードウェアがある場合）
4. **Cython最適化**: 計算集約的な部分をCで実装
5. **メモリ最適化**: メモリ使用量の削減とキャッシュの効率的な利用
6. **計算結果のキャッシュ**: 同じデータに対する再計算を回避

## 使用方法

### 基本的な使用法

```python
from indicators.kernel_ma import KernelMA

# KernelMAインスタンスの作成
kernel_ma = KernelMA(
    er_period=21,
    max_bandwidth=10.0,
    min_bandwidth=2.0,
    kernel_type='gaussian',
    slope_period=5
)

# 計算の実行
kernel_ma_values = kernel_ma.calculate(price_data)
```

### GPU加速の有効化

```python
# GPU加速を有効化（GPUが利用可能な場合）
kernel_ma = KernelMA(
    er_period=21,
    max_bandwidth=10.0,
    min_bandwidth=2.0,
    kernel_type='gaussian',
    slope_period=5,
    use_gpu='auto'  # 'auto', 'force', 'disable'のいずれか
)
```

### 環境変数による制御

```bash
# 環境変数でGPU使用を制御
export USE_GPU=true  # または 'false', 'auto'
```

## Cythonモジュールのビルド

Cython最適化を利用するには、まずモジュールをビルドする必要があります：

```bash
# indicatorsディレクトリに移動
cd indicators

# Cythonモジュールのビルド
python setup.py build_ext --inplace
```

## パフォーマンス比較

以下は、異なる最適化手法を使用した場合のパフォーマンス比較です：

| 最適化手法 | 相対速度 | メモリ使用量 |
|------------|----------|------------|
| 基本実装 | 1x | 高い |
| Numba JIT | 10-20x | 中程度 |
| Numba + 並列処理 | 20-40x | 中程度 |
| Cython | 30-50x | 低い |
| GPU (CUDA) | 50-100x | 低い |

※ パフォーマンスはハードウェアとデータサイズによって異なります。

## 注意事項

- GPU加速を使用するには、CUDA対応のNVIDIA GPUとCuPyライブラリが必要です。
- Cython最適化を使用するには、C/C++コンパイラとCythonライブラリが必要です。
- 大量のデータを処理する場合は、メモリ使用量に注意してください。
- 最適化手法によって、結果に微小な数値的な違いが生じる可能性があります。 

# AlphaXMA - ハイパースムーサーを使用した強化型アダプティブ移動平均線

## 概要

AlphaXMAは、AlphaMAをベースにハイパースムーサーによる平滑化を追加した高度な移動平均線です。
動的なパラメータ調整と3段階フィルタリングを組み合わせることで、ノイズに強く、トレンド追従性能の高い移動平均線を実現しています。

## 特徴

- 効率比（ER）に基づいて動的にパラメータを調整
- ハイパースムーサーによる3段階フィルタリング
- トレンド相場では素早く反応、レンジ相場ではノイズを除去
- Numba JITコンパイルによる高速化

## 使用方法

```python
from indicators import AlphaXMA

# AlphaXMAインスタンスの作成
alpha_xma = AlphaXMA(
    er_period=21,              # 効率比の計算期間
    max_kama_period=144,       # KAMAピリオドの最大値
    min_kama_period=10,        # KAMAピリオドの最小値
    max_slow_period=89,        # 遅い移動平均の最大期間
    min_slow_period=30,        # 遅い移動平均の最小期間
    max_fast_period=13,        # 速い移動平均の最大期間
    min_fast_period=2,         # 速い移動平均の最小期間
    hyper_smooth_period=10     # ハイパースムーサーの期間
)

# 計算の実行
alpha_xma_values = alpha_xma.calculate(price_data)

# 効率比の取得
er_values = alpha_xma.get_efficiency_ratio()

# 動的期間の取得
kama_period, fast_period, slow_period = alpha_xma.get_dynamic_periods()
```

## 推奨パラメータ

| 市場環境 | er_period | max_kama_period | min_kama_period | hyper_smooth_period |
|---------|-----------|-----------------|-----------------|---------------------|
| 株式市場 | 21        | 144             | 10              | 10                  |
| 仮想通貨 | 14        | 89              | 5               | 8                   |
| FX市場   | 34        | 233             | 13              | 13                  | 

# エーラーズのドミナントサイクル検出アルゴリズム

このパッケージは、ジョン・エーラーズ (John Ehlers) によって開発された様々なドミナントサイクル検出アルゴリズムの実装を提供します。これらのアルゴリズムは、価格時系列から主要な周期性を特定するために使用されます。

## 概要

トレーディングにおいて、市場の周期性を把握することは重要です。ジョン・エーラーズは、デジタル信号処理の原則に基づいたいくつかの手法を開発し、これらは市場の周期性を検出するための強力なツールとなっています。

このパッケージには次のアルゴリズムが含まれています：

1. **ホモダイン判別器 (Homodyne Discriminator, HoDyDC)**  
   価格データから直接ドミナントサイクルを検出するための基本的な手法

2. **位相累積法 (Phase Accumulation, PhAcDC)**  
   価格の位相変化を追跡して周期を検出する手法

3. **二重微分法 (Dual Differentiator, DuDiDC)**  
   価格変化の加速度を用いて周期を検出する手法

4. **拡張版アルゴリズム (E-シリーズ)**  
   上記の各手法に高域通過フィルターと低域通過フィルターを組み合わせた拡張版

5. **離散フーリエ変換 (Discrete Fourier Transform, DFTDC)**  
   周波数領域での解析を使用してドミナントサイクルを検出する手法

## インストール

このパッケージは、プロジェクトのルートディレクトリから以下のように使用できます：

```python
from indicators import (
    EhlersHoDyDC, EhlersPhAcDC, EhlersDuDiDC,
    EhlersHoDyDCE, EhlersPhAcDCE, EhlersDuDiDCE,
    EhlersDFTDC
)
```

## 使用方法

各アルゴリズムの使用方法は以下の通りです：

```python
import pandas as pd
import numpy as np
from indicators import EhlersHoDyDC

# サンプルデータの作成
data = pd.DataFrame({
    'close': np.random.randn(1000).cumsum()
})

# アルゴリズムのインスタンス化
cycle_detector = EhlersHoDyDC(cycle_part=0.5)

# ドミナントサイクルの計算
dominant_cycle = cycle_detector.calculate(data)

print(f"ドミナントサイクル: {dominant_cycle[-1]}")
```

## パラメータの説明

### 共通パラメータ

- **cycle_part** (float, デフォルト=0.5)：  
  検出したサイクルの一部を使用するための係数。1.0に近いほど長いサイクル、0.0に近いほど短いサイクルを検出します。

### 拡張版アルゴリズム (E-シリーズ) の追加パラメータ

- **lp_period** (int, デフォルト=10)：  
  低域通過フィルターの期間。短いノイズを除去するためのパラメータ。

- **hp_period** (int, デフォルト=48)：  
  高域通過フィルターの期間。長期トレンドを除去するためのパラメータ。

### DFTDCの追加パラメータ

- **window** (int, デフォルト=50)：  
  フーリエ変換を適用するウィンドウサイズ。

## アルゴリズムの選び方

各アルゴリズムには長所と短所があります：

1. **ホモダイン判別器 (HoDyDC)**
   - 長所：比較的安定した周期検出
   - 短所：ノイズの多い環境では誤検出の可能性

2. **位相累積法 (PhAcDC)**
   - 長所：ノイズに対してより堅牢
   - 短所：サイクル変化への反応がやや遅い

3. **二重微分法 (DuDiDC)**
   - 長所：サイクル変化への迅速な反応
   - 短所：ノイズに敏感で変動が大きい場合がある

4. **拡張版アルゴリズム (E-シリーズ)**
   - 長所：ノイズとトレンドの影響を軽減
   - 短所：パラメータ調整が必要

5. **離散フーリエ変換 (DFTDC)**
   - 長所：周波数領域での詳細な解析が可能
   - 短所：計算負荷が高く、リアルタイム使用には向かない場合がある

## 例

より詳細な使用例については、`examples`ディレクトリ内のサンプルスクリプトを参照してください：

- `ehlers_cycle_example.py` - 基本的な使用例
- `ehlers_cycle_benchmark.py` - 様々な市場条件下での各アルゴリズムのベンチマーク

## 参考文献

1. Ehlers, J. F. (2002). *Cycle Analytics for Traders*
2. Ehlers, J. F. (2013). *Cycle Analytics for Traders: Advanced Technical Trading Concepts*
3. Ehlers, J. F. (2020). *Rocket Science for Traders: Digital Signal Processing Applications* 

## 適応型移動平均指標

### CMA (Cycle Moving Average)
CMAはZ_MAをシンプル化したバージョンの適応型移動平均指標です。主な特徴:
- KAMAの期間をドミナントサイクル検出器で直接決定
- fastとslow期間に固定値を使用
- サイクル効率比(CER)に基づいて適応的に計算

使用例:
```python
from indicators import CMA, CycleEfficiencyRatio

# サイクル効率比（CER）を計算
cer = CycleEfficiencyRatio(detector_type='hody')
cer_values = cer.calculate(data)

# CMAを計算
cma = CMA(detector_type='hody', fast_period=2, slow_period=30)
cma_values = cma.calculate(data, cer_values)
```

### CATR (Cycle Average True Range)
CATRはZATRをシンプル化したバージョンのボラティリティ指標です。主な特徴:
- ATR期間をドミナントサイクル検出器で直接決定
- サイクル効率比(CER)を活用
- 金額ベースと%ベースの両方の値を提供

使用例:
```python
from indicators import CATR, CycleEfficiencyRatio

# サイクル効率比（CER）を計算
cer = CycleEfficiencyRatio(detector_type='hody')
cer_values = cer.calculate(data)

# CATRを計算
catr = CATR(detector_type='hody', smoother_type='alma')
catr_values = catr.calculate(data, cer_values)

# ATRチャネルを計算（例: 2 ATR）
absolute_atr = catr.get_absolute_atr()
upper_band = close + absolute_atr * 2
lower_band = close - absolute_atr * 2
```

### Z_MA (Z Moving Average) 