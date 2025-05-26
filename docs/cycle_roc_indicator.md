# サイクルROC（Cycle Rate of Change）インジケーター

## 概要

サイクルROCインジケーターは、エーラーズのドミナントサイクル検出技術を利用して、動的な期間でのROC（Rate of Change: 変化率）を計算する革新的なテクニカル指標です。従来の固定期間ROCとは異なり、市場のサイクル性を考慮して最適な計算期間を自動調整します。

## 特徴

### 主な機能
- **動的期間調整**: エーラーズのサイクル検出器を使用して最適な計算期間を自動決定
- **複数の価格ソース対応**: close、hlc3、hl2などの価格ソースを選択可能
- **ALMAスムージング**: ノイズを減らすためのオプションのスムージング機能
- **シグナル生成**: ROC値に基づく上昇/下降/中立シグナルを自動生成
- **Numba最適化**: 高速計算のためのNumba JITコンパイル
- **カルマンフィルター対応**: 価格データの前処理オプション

### サイクル検出器の種類
- `dudi_e`: 拡張二重微分（推奨、デフォルト）
- `hody_e`: 拡張ホモダイン判別機
- `phac_e`: 拡張位相累積
- `dudi`: 二重微分
- `hody`: ホモダイン判別機
- `phac`: 位相累積

## 使用方法

### 基本的な使用例

```python
from indicators.cycle_roc import CycleROC

# 基本的なサイクルROCの作成
cycle_roc = CycleROC(
    detector_type='dudi_e',    # サイクル検出器タイプ
    src_type='close',          # 価格ソース
    smooth_roc=True,           # ALMAスムージング有効
    signal_threshold=1.0       # シグナル判定しきい値（±1%）
)

# 計算実行
roc_values = cycle_roc.calculate(ohlc_data)

# 結果の取得
result = cycle_roc.get_result()
cycle_periods = cycle_roc.get_cycle_periods()
signals = cycle_roc.get_roc_signals()
```

### 詳細パラメータ設定

```python
cycle_roc = CycleROC(
    # サイクル検出器設定
    detector_type='dudi_e',
    lp_period=5,               # ローパスフィルター期間
    hp_period=144,             # ハイパスフィルター期間
    cycle_part=0.5,            # サイクル部分係数
    max_cycle=144,             # 最大サイクル期間
    min_cycle=5,               # 最小サイクル期間
    max_output=55,             # 最大出力値
    min_output=5,              # 最小出力値
    
    # 価格ソース設定
    src_type='hlc3',           # HLC3を使用
    
    # スムージング設定
    smooth_roc=True,
    roc_alma_period=5,         # ALMA期間
    roc_alma_offset=0.85,      # ALMAオフセット
    roc_alma_sigma=6,          # ALMAシグマ
    
    # シグナル設定
    signal_threshold=1.5,      # シグナルしきい値
    
    # カルマンフィルター設定（オプション）
    use_kalman_filter=True,
    kalman_measurement_noise=1.0,
    kalman_process_noise=0.01
)
```

## 計算原理

### 基本的な計算フロー

1. **サイクル検出**: 指定されたサイクル検出器でドミナントサイクル期間を計算
2. **期間制限**: 検出されたサイクル期間をmin_cycle〜max_cycleの範囲に制限
3. **ROC計算**: 各時点で動的期間を使用してROCを計算
4. **スムージング**: オプションでALMAスムージングを適用
5. **シグナル生成**: しきい値に基づいてシグナルを生成

### ROC計算式

```
ROC[i] = ((Price[i] - Price[i - Period[i]]) / Price[i - Period[i]]) × 100
```

ここで：
- `Price[i]`: i時点の価格
- `Period[i]`: i時点のドミナントサイクル期間
- `ROC[i]`: i時点のROC値（パーセンテージ）

### シグナル判定

- **上昇シグナル** (`1`): ROC > signal_threshold
- **下降シグナル** (`-1`): ROC < -signal_threshold  
- **中立シグナル** (`0`): -signal_threshold ≤ ROC ≤ signal_threshold

## 戻り値と結果取得

### CycleROCResult データクラス

```python
@dataclass
class CycleROCResult:
    values: np.ndarray          # 最終ROC値（スムージング適用後）
    raw_values: np.ndarray      # 生のROC値（スムージング前）
    cycle_periods: np.ndarray   # 使用されたサイクル期間
    roc_signals: np.ndarray     # ROCシグナル
```

### 結果の取得方法

```python
# 基本的な結果取得
roc_values = cycle_roc.calculate(data)
result = cycle_roc.get_result()

# 個別要素の取得
cycle_periods = cycle_roc.get_cycle_periods()
signals = cycle_roc.get_roc_signals()

# 詳細分析
print(f"ROC平均: {np.nanmean(result.values):.4f}")
print(f"平均サイクル期間: {np.nanmean(result.cycle_periods):.1f}")
print(f"上昇シグナル数: {np.sum(result.roc_signals == 1)}")
```

## 活用例

### トレンド分析

```python
# 長期トレンド分析用の設定
trend_roc = CycleROC(
    detector_type='dudi_e',
    max_cycle=100,             # 長期サイクルを検出
    min_cycle=20,
    smooth_roc=True,
    roc_alma_period=10,        # 長期スムージング
    signal_threshold=2.0       # 高いしきい値でノイズを除去
)
```

### 短期スキャルピング

```python
# 短期取引用の設定
scalp_roc = CycleROC(
    detector_type='phac_e',    # 高感度検出器
    max_cycle=20,              # 短期サイクル
    min_cycle=3,
    smooth_roc=False,          # スムージングなし
    signal_threshold=0.5       # 低いしきい値で敏感に反応
)
```

### マルチタイムフレーム分析

```python
# 異なる時間軸での分析
timeframes = {
    'short': CycleROC(max_cycle=20, min_cycle=3),
    'medium': CycleROC(max_cycle=50, min_cycle=10),
    'long': CycleROC(max_cycle=100, min_cycle=20)
}

for name, indicator in timeframes.items():
    values = indicator.calculate(data)
    signals = indicator.get_roc_signals()
    print(f"{name}: {np.sum(signals == 1)}上昇, {np.sum(signals == -1)}下降")
```

## パフォーマンス最適化

### Numba最適化

サイクルROCは以下の関数でNumba JITコンパイルを使用：

- `calculate_roc_for_period()`: 単一期間のROC計算
- `calculate_roc_array()`: 動的期間配列でのROC計算
- `calculate_roc_signals()`: シグナル生成

### キャッシング

- データハッシュベースの結果キャッシング
- 同一データでの再計算を自動スキップ
- パラメータ変更時の自動無効化

## 比較：従来ROC vs サイクルROC

| 特徴 | 従来ROC | サイクルROC |
|------|---------|-------------|
| 期間設定 | 固定 | 動的（サイクル検出） |
| 市場適応性 | 低 | 高 |
| ノイズ耐性 | 中 | 高（ALMAスムージング） |
| 計算コスト | 低 | 中（最適化済み） |
| 設定の複雑さ | 低 | 中 |
| 精度 | 中 | 高 |

## 注意事項と制限

1. **初期値**: 最初の数ポイントはNaN値となります
2. **計算負荷**: サイクル検出のため従来ROCより計算コストが高い
3. **パラメータ調整**: 最適なパラメータは市場や時間軸によって異なる
4. **データ品質**: 高品質なOHLCデータが必要

## 実装の詳細

### アーキテクチャ

```
CycleROC
├── EhlersUnifiedDC (サイクル検出)
├── PriceSource (価格抽出)
├── ALMA (スムージング、オプション)
└── Numba最適化関数
    ├── calculate_roc_for_period
    ├── calculate_roc_array
    └── calculate_roc_signals
```

### エラーハンドリング

- 不正なデータに対する堅牢な処理
- NaN値の適切な処理
- 計算エラー時の自動復旧

## テストとベンチマーク

### テストカバレージ

- 基本的な計算テスト
- 従来ROCとの比較テスト
- 異なるサイクル検出器のテスト
- エラーケースのテスト

### ベンチマーク結果（例）

```
データサイズ: 1000ポイント
計算時間: ~15ms（初回）、~1ms（キャッシュ）
メモリ使用量: ~2MB
```

## まとめ

サイクルROCインジケーターは、市場のサイクル性を考慮した高度なROC計算を提供し、従来の固定期間ROCよりも市場の変化に適応的で精度の高い分析を可能にします。Numba最適化とキャッシング機能により、高性能な実装を実現しています。 