# Price Source 更新情報

## 2024年 Ultimate Smoother / Super Smoother 統合

### 追加された機能

PriceSourceクラスに、10期間のUltimate SmootherとSuper Smootherによる平滑化価格ソースが追加されました。

### 新しいソースタイプ

#### Ultimate Smoother (US)
- `us_close` - Ultimate Smoother (10期間) 終値
- `us_high` - Ultimate Smoother (10期間) 高値  
- `us_low` - Ultimate Smoother (10期間) 安値
- `us_hlc3` - Ultimate Smoother (10期間) HLC3
- `us_hl2` - Ultimate Smoother (10期間) HL2

#### Super Smoother (SS)
- `ss_close` - Super Smoother (10期間) 終値
- `ss_high` - Super Smoother (10期間) 高値
- `ss_low` - Super Smoother (10期間) 安値  
- `ss_hlc3` - Super Smoother (10期間) HLC3
- `ss_hl2` - Super Smoother (10期間) HL2

### 使用方法

```python
from indicators.price_source import PriceSource

# Ultimate Smootherで平滑化されたHLC3価格を取得
us_hlc3 = PriceSource.calculate_source(data, 'us_hlc3')

# Super Smootherで平滑化された終値を取得  
ss_close = PriceSource.calculate_source(data, 'ss_close')
```

### 技術的実装詳細

1. **循環インポート回避**: スムーザーモジュールは遅延ロードで実装
2. **キャッシング**: 計算結果は内部キャッシュに保存され、同一データに対する再計算を防止
3. **固定パラメータ**: 
   - Ultimate Smoother: 10期間、固定モード
   - Super Smoother: 10期間、2極フィルター

### パフォーマンス特性

| スムーザー | ノイズ削減率 | 特徴 |
|-----------|------------|------|
| Ultimate Smoother | 約60-62% | 価格追従性重視、低遅延 |
| Super Smoother | 約79-80% | 高ノイズ削減、滑らかな出力 |

### 新しいユーティリティメソッド

```python
# スムーザーソースかどうかを判定
PriceSource.is_smoother_source('us_hlc3')  # True
PriceSource.is_smoother_source('close')    # False

# スムーザーのデフォルトパラメータを取得
params = PriceSource.get_smoother_params()
# {'period': 10, 'us_period_mode': 'fixed', 'ss_num_poles': 2}
```

### 実装ファイル

- `/indicators/price_source.py` - メイン実装
- `/indicators/smoother/ultimate_smoother.py` - Ultimate Smoother
- `/indicators/smoother/super_smoother.py` - Super Smoother  
- `/indicators/smoother/source_calculator.py` - 循環インポート回避用ユーティリティ