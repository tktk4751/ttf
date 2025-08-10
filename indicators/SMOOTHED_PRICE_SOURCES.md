# Smoothed Price Sources

## 概要
PriceSourceクラスに、Ultimate SmootherとSuper Smootherによる平滑化された価格ソースが統合されました。これらのソースは10期間のスムージングを提供し、ノイズを削減しながら価格トレンドを維持します。

## 利用可能なスムーズ化ソース

### Ultimate Smoother (US) ソース
- `us_close`: Ultimate Smoother (10期間) 終値
- `us_high`: Ultimate Smoother (10期間) 高値
- `us_low`: Ultimate Smoother (10期間) 安値
- `us_hlc3`: Ultimate Smoother (10期間) HLC3
- `us_hl2`: Ultimate Smoother (10期間) HL2

特徴:
- ノイズ削減率: 約60-62%
- より適度なスムージングで元の価格動向を保持
- パスバンドでゼロラグ特性
- 価格追従性重視の用途に最適

### Super Smoother (SS) ソース
- `ss_close`: Super Smoother (10期間) 終値
- `ss_high`: Super Smoother (10期間) 高値
- `ss_low`: Super Smoother (10期間) 安値
- `ss_hlc3`: Super Smoother (10期間) HLC3
- `ss_hl2`: Super Smoother (10期間) HL2

特徴:
- ノイズ削減率: 約79-80%
- より高いノイズ削減効果
- 2極デジタルフィルター
- 高ノイズ削減が必要な用途に最適

## 使用例

```python
from indicators.price_source import PriceSource
import pandas as pd

# OHLCデータを準備
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
})

# Ultimate Smootherで平滑化されたHLC3を取得
us_hlc3 = PriceSource.calculate_source(data, 'us_hlc3')

# Super Smootherで平滑化された終値を取得
ss_close = PriceSource.calculate_source(data, 'ss_close')

# 利用可能なすべてのソースを確認
sources = PriceSource.get_available_sources()
for src_type, description in sources.items():
    print(f"{src_type}: {description}")
```

## インジケーターでの使用

```python
class MyIndicator(Indicator):
    def __init__(self, src_type='us_hlc3'):
        # Ultimate Smoother HLC3を使用
        self.src_type = src_type
    
    def calculate(self, data):
        # スムーズ化された価格ソースを取得
        price = PriceSource.calculate_source(data, self.src_type)
        # インジケーター計算を実行
        ...
```

## 選択ガイドライン

### Ultimate Smootherを選ぶべき場合:
- 価格の細かい動きも捉えたい
- エントリー/エグジットシグナルの遅延を最小限にしたい
- トレンドフォロー戦略で使用
- より応答性の高いインジケーターが必要

### Super Smootherを選ぶべき場合:
- ノイズの多い市場データを扱う
- 長期的なトレンドを重視
- 偽シグナルを減らしたい
- より滑らかな出力が必要

## パフォーマンス比較

| ソースタイプ | Ultimate Smoother | Super Smoother |
|------------|------------------|----------------|
| CLOSE      | 62.0%           | 79.8%         |
| HIGH       | 61.7%           | 79.6%         |
| LOW        | 61.0%           | 79.5%         |
| HLC3       | 61.5%           | 79.7%         |
| HL2        | 61.2%           | 79.5%         |

*ノイズ削減率 = 1 - (スムーズ化後の標準偏差 / 元データの標準偏差)

## 技術詳細

### Ultimate Smoother
- John Ehlersによる設計
- 第2次IIRフィルター（無限インパルス応答）
- パスバンドでゼロラグ特性
- 固定10期間設定

### Super Smoother
- John Ehlersによる設計
- 2極デジタルフィルター
- 優れたノイズ削減特性
- 固定10期間設定

## 注意事項

1. スムーズ化されたソースは元データより遅延が発生します
2. 10期間は固定値で、現在変更できません
3. 計算には少なくとも10期間のデータが必要です
4. スムーザーモジュールは遅延ロードされるため、初回使用時に若干の遅延があります