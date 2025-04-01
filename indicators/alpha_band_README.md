# アルファバンド (Alpha Band)

アルファバンドは、サイクル効率比（CER）を活用した高度に適応的なボラティリティバンドインジケーターです。AlphaKeltnerChannelをベースにして作られており、通常の効率比（ER）の代わりにサイクル効率比（CER）を使用しています。

## 特徴

- **中心線**: AlphaMA（動的適応型移動平均）を使用
- **バンド幅**: AlphaATR（動的適応型ATR）を使用
- **動的適応性**: サイクル効率比（CER）に基づいて全てのパラメータが自動調整
- **平滑化**: RSXの3段階平滑化アルゴリズムをAlphaATRに適用

## 市場状態に応じた挙動

- **トレンドが強い（CER高い）**:
  - 狭いバンド幅（小さい乗数）
  - トレンドをタイトに追従
  - 素早く反応するため早期のシグナル検出が可能

- **トレンドが弱い（CER低い）**:
  - 広いバンド幅（大きい乗数）
  - レンジ相場の振れ幅を捉える
  - ノイズに強く、偽シグナルを減少

## サイクル効率比（CER）の利点

通常の効率比（ER）と比較して、サイクル効率比（CER）には以下の利点があります：

- **動的周期検出**: 現在の相場環境から自動的に適切な周期を検出
- **市場サイクルへの同調**: 市場のリズムに合わせた調整が可能
- **多様なサイクル検出アルゴリズム**: 複数のエーラーズのドミナントサイクル検出法を選択可能

## パラメータ

| パラメータ | 説明 | デフォルト値 |
|------------|------|-------------|
| `cycle_detector_type` | サイクル検出器の種類 | 'hody_dc' |
| `lp_period` | ローパスフィルターの期間 | 5 |
| `hp_period` | ハイパスフィルターの期間 | 144 |
| `cycle_part` | サイクル部分の倍率 | 0.5 |
| `max_kama_period` | AlphaMAのKAMA最大期間 | 55 |
| `min_kama_period` | AlphaMAのKAMA最小期間 | 8 |
| `max_atr_period` | AlphaATRの最大期間 | 55 |
| `min_atr_period` | AlphaATRの最小期間 | 8 |
| `max_multiplier` | ATR乗数の最大値 | 3.0 |
| `min_multiplier` | ATR乗数の最小値 | 1.5 |

### サイクル検出器の種類

- `'dudi_dc'` - 二重微分法
- `'hody_dc'` - ホモダイン判別機（デフォルト）
- `'phac_dc'` - 位相累積法
- `'dudi_dce'` - 拡張二重微分法
- `'hody_dce'` - 拡張ホモダイン判別機
- `'phac_dce'` - 拡張位相累積法

## 使用例

```python
import pandas as pd
from indicators import AlphaBand

# データ読み込み
data = pd.read_csv('price_data.csv')

# AlphaBandインスタンス作成
alpha_band = AlphaBand(
    cycle_detector_type='hody_dc',
    lp_period=5,
    hp_period=144,
    cycle_part=0.5,
    max_multiplier=3.0,
    min_multiplier=1.5
)

# 計算実行
middle = alpha_band.calculate(data)

# バンド値の取得
middle, upper, lower = alpha_band.get_bands()

# その他の値を取得
cer = alpha_band.get_cycle_er()
dynamic_multiplier = alpha_band.get_dynamic_multiplier()
alpha_atr = alpha_band.get_alpha_atr()
```

## トレード戦略の例

### トレンドフォロー戦略

- **エントリー**:
  - ロング: 価格がバンドの下限から中心線を上抜けたとき
  - ショート: 価格がバンドの上限から中心線を下抜けたとき

- **利確**:
  - ロング: 価格がバンドの上限に達したとき
  - ショート: 価格がバンドの下限に達したとき

- **損切り**:
  - ロング: 価格がバンドの下限を下回ったとき
  - ショート: 価格がバンドの上限を上回ったとき

### レンジ相場戦略

- **エントリー**:
  - ロング: 価格がバンドの下限に達し、CER < 0.3のとき
  - ショート: 価格がバンドの上限に達し、CER < 0.3のとき

- **利確**:
  - ロング: 価格が中心線を上回ったとき
  - ショート: 価格が中心線を下回ったとき

- **損切り**:
  - 反対側のバンドを超えたとき

## 注意点

- CERの値が低い（0.3以下）場合、レンジ相場の可能性が高い
- CERの値が高い（0.7以上）場合、トレンド相場の可能性が高い
- 相場状態（トレンド・レンジ）に適した戦略を選択することが重要 