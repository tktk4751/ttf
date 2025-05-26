# ROC継続性（ROC Persistence）インジケーター

## 概要

ROC継続性インジケーターは、ROC（Rate of Change）が正または負の領域にどれだけ長く滞在しているかを測定し、-1から1の範囲で継続性を表現する革新的なテクニカル指標です。サイクルROCをベースにしており、市場のモメンタムの持続性を定量化します。

## 特徴

### 主な機能
- **継続性の定量化**: ROCの正負の継続期間を-1から1の範囲で表現
- **飽和機能**: 144期間（設定可能）で自動的に±1に収束
- **動的期間**: サイクルROCベースで市場に適応した期間を使用
- **Numba最適化**: 高速計算のためのNumba JITコンパイル
- **平滑化オプション**: ノイズを減らすための継続性値平滑化
- **包括的な結果**: 継続期間、方向、元ROC値を含む詳細な結果

### 計算原理

1. **ROC計算**: サイクルROCで動的期間でのROCを計算
2. **方向判定**: ROC値が正(>0)、負(<0)、ゼロ(=0)かを判定
3. **継続期間追跡**: 同じ方向が続く期間をカウント
4. **正規化**: 継続期間を最大期間で割って-1から1の範囲に変換
5. **平滑化**: オプションで移動平均による平滑化

### 値の意味

- **+1.0**: ROCが正の領域に最大期間（144期間）滞在
- **+0.5**: ROCが正の領域に半分の期間（72期間）滞在
- **0.0**: ROCがゼロまたは継続期間なし
- **-0.5**: ROCが負の領域に半分の期間（72期間）滞在
- **-1.0**: ROCが負の領域に最大期間（144期間）滞在

## 使用方法

### 基本的な使用例

```python
from indicators.roc_persistence import ROCPersistence

# 基本的なROC継続性の作成
roc_persistence = ROCPersistence(
    detector_type='dudi_e',         # サイクル検出器タイプ
    max_persistence_periods=144,    # 最大継続期間
    smooth_persistence=True,        # 継続性値の平滑化
    persistence_smooth_period=3,    # 平滑化期間
    src_type='close'               # 価格ソース
)

# 計算実行
persistence_values = roc_persistence.calculate(ohlc_data)

# 結果の取得
result = roc_persistence.get_result()
persistence_periods = roc_persistence.get_persistence_periods()
roc_directions = roc_persistence.get_roc_directions()
```

### 詳細パラメータ設定

```python
roc_persistence = ROCPersistence(
    # サイクルROCパラメータ
    detector_type='dudi_e',
    lp_period=5,
    hp_period=144,
    cycle_part=0.7,
    max_cycle=144,
    min_cycle=5,
    src_type='hlc3',
    smooth_roc=True,
    roc_alma_period=5,
    
    # ROC継続性パラメータ
    max_persistence_periods=100,   # 短期間での飽和
    smooth_persistence=True,
    persistence_smooth_period=5,   # より強い平滑化
)
```

## 計算フロー

### 1. ROC計算
```
ROC[i] = ((Price[i] - Price[i - CyclePeriod[i]]) / Price[i - CyclePeriod[i]]) × 100
```

### 2. 方向判定
```
Direction[i] = {
    1   if ROC[i] > 0    # 正
   -1   if ROC[i] < 0    # 負
    0   if ROC[i] = 0    # ゼロ
}
```

### 3. 継続期間計算
```
if Direction[i] == Direction[i-1]:
    CurrentPeriods = CurrentPeriods + 1
else:
    CurrentPeriods = 1
```

### 4. 継続性値計算
```
if Direction[i] == 1:
    Persistence[i] = min(CurrentPeriods / MaxPeriods, 1.0)
elif Direction[i] == -1:
    Persistence[i] = -min(CurrentPeriods / MaxPeriods, 1.0)
else:
    Persistence[i] = 0.0
```

## パラメータ詳細

### ROC継続性パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_persistence_periods` | 144 | 最大継続期間（飽和点） |
| `smooth_persistence` | True | 継続性値の平滑化を行うか |
| `persistence_smooth_period` | 3 | 継続性値の平滑化期間 |

### サイクルROCパラメータ（継承）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `detector_type` | 'dudi_e' | サイクル検出器タイプ |
| `max_cycle` | 144 | 最大サイクル期間 |
| `min_cycle` | 5 | 最小サイクル期間 |
| `src_type` | 'hlc3' | 価格ソース |
| `smooth_roc` | True | ROC値のスムージング |

## 戻り値と結果取得

### ROCPersistenceResult データクラス

```python
@dataclass
class ROCPersistenceResult:
    values: np.ndarray                  # ROC継続性の値（-1から1）
    persistence_periods: np.ndarray     # 現在の継続期間
    roc_directions: np.ndarray         # ROCの方向（1=正、-1=負、0=ゼロ）
    roc_values: np.ndarray             # 元のROC値
    cycle_periods: np.ndarray          # サイクル期間
```

### 結果の取得方法

```python
# 基本的な結果取得
persistence_values = roc_persistence.calculate(data)
result = roc_persistence.get_result()

# 個別要素の取得
persistence_periods = roc_persistence.get_persistence_periods()
roc_directions = roc_persistence.get_roc_directions()
roc_values = roc_persistence.get_roc_values()

# 詳細分析
print(f"継続性平均: {np.nanmean(result.values):.4f}")
print(f"最大正継続: {np.nanmax(result.persistence_periods[result.roc_directions == 1])}")
print(f"最大負継続: {np.nanmax(result.persistence_periods[result.roc_directions == -1])}")
```

## 活用例

### トレンド強度の測定

```python
# 強いトレンド検出用の設定
trend_persistence = ROCPersistence(
    max_persistence_periods=200,      # 長期継続を検出
    smooth_persistence=True,
    persistence_smooth_period=5,      # ノイズ除去
    detector_type='dudi_e'
)

# 強いトレンドの判定
values = trend_persistence.calculate(data)
strong_uptrend = values > 0.8    # 強い上昇トレンド
strong_downtrend = values < -0.8  # 強い下降トレンド
```

### 短期反転シグナル

```python
# 短期反転検出用の設定
reversal_persistence = ROCPersistence(
    max_persistence_periods=50,       # 短期継続
    smooth_persistence=False,         # 即応性重視
    detector_type='phac_e'           # 高感度検出器
)

# 反転シグナルの検出
values = reversal_persistence.calculate(data)
# 極値からの反転を検出
reversal_signals = (values > 0.9) | (values < -0.9)
```

### マルチタイムフレーム分析

```python
# 異なる時間軸での継続性分析
timeframes = {
    'short': ROCPersistence(max_persistence_periods=30),   # 短期
    'medium': ROCPersistence(max_persistence_periods=100), # 中期
    'long': ROCPersistence(max_persistence_periods=200)    # 長期
}

for name, indicator in timeframes.items():
    values = indicator.calculate(data)
    current_persistence = values[-1]
    print(f"{name}: {current_persistence:.3f}")
```

## Numba最適化

### 最適化された関数

- `calculate_roc_persistence_numba()`: ROC継続性の計算
- `calculate_smoothed_persistence()`: 継続性値の平滑化

### パフォーマンス特性

```
データサイズ: 1000ポイント
計算時間: ~20ms（初回コンパイル込み）、~2ms（2回目以降）
メモリ使用量: ~3MB
```

## 使用シナリオ

### 1. トレンド継続性の評価

```python
# 現在のトレンドの強さを評価
current_persistence = persistence_values[-1]

if current_persistence > 0.7:
    print("強い上昇トレンド継続中")
elif current_persistence < -0.7:
    print("強い下降トレンド継続中")
else:
    print("トレンドが弱いまたは転換期")
```

### 2. エントリータイミング

```python
# 継続性が高まった時点でエントリー
persistence_change = persistence_values[-1] - persistence_values[-5]

if persistence_change > 0.3 and persistence_values[-1] > 0.5:
    print("上昇モメンタム強化：ロングエントリー検討")
elif persistence_change < -0.3 and persistence_values[-1] < -0.5:
    print("下降モメンタム強化：ショートエントリー検討")
```

### 3. 利確・損切りタイミング

```python
# 継続性の減衰を利確・損切りシグナルとして使用
if persistence_values[-1] > 0.8 and persistence_values[-1] < persistence_values[-5]:
    print("上昇継続性の減衰：利確検討")
elif persistence_values[-1] < -0.8 and persistence_values[-1] > persistence_values[-5]:
    print("下降継続性の減衰：損切り検討")
```

## 従来指標との比較

| 特徴 | ROC継続性 | 従来のROC | RSI |
|------|-----------|-----------|-----|
| 継続性測定 | ○ | × | × |
| トレンド強度 | ○ | △ | △ |
| 飽和機能 | ○ | × | ○ |
| 動的期間 | ○ | × | × |
| ノイズ耐性 | ○ | △ | ○ |

## 注意事項と制限

1. **初期値**: 最初の数ポイントはNaN値となります
2. **継続性の遅延**: 継続性は本質的に遅行指標です
3. **パラメータ調整**: 市場特性に応じた最適化が必要
4. **極値の意味**: ±1に近い値は転換の可能性も示唆

## 実装の詳細

### アーキテクチャ

```
ROCPersistence
├── CycleROC (ROC計算)
│   ├── EhlersUnifiedDC (サイクル検出)
│   ├── PriceSource (価格抽出)
│   └── ALMA (スムージング)
└── Numba最適化関数
    ├── calculate_roc_persistence_numba
    └── calculate_smoothed_persistence
```

### エラーハンドリング

- ROC計算エラーの自動復旧
- NaN値の適切な処理
- サイクル検出失敗時のフォールバック

## まとめ

ROC継続性インジケーターは、市場のモメンタムの持続性を定量化する強力なツールです。従来のモメンタム指標では捉えられない「継続性」の概念を導入し、トレンドの強さと持続可能性を同時に評価できます。サイクルROCをベースとした動的期間の使用により、市場の変化に適応的で、Numba最適化により高速な計算を実現しています。

主な用途：
- トレンドの強度と持続性の評価
- エントリー・エグジットタイミングの判定
- マルチタイムフレーム分析
- リスク管理のための継続性監視 