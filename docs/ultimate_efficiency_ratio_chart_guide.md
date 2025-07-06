# 🚀 Ultimate Efficiency Ratio Chart 使用ガイド

## 概要

Ultimate Efficiency Ratio Chart は、次世代のテクニカル分析インジケーター「Ultimate Efficiency Ratio V3.0」を視覚化するための高度なチャートシステムです。

### 🌟 主な特徴

- **🔬 量子強化ヒルベルト変換**: 瞬時振幅・位相・トレンド強度を超低遅延で検出
- **🎯 量子適応カルマンフィルター**: 動的ノイズモデリング + 量子コヒーレンス調整
- **🚀 5次元ハイパー効率率**: 従来ERを遥かに超える多次元効率測定
- **💡 量子もつれ効果解析**: 価格間相関の量子力学的解釈
- **⚡ 超低遅延・超高精度**: 実用性重視のシンプル設計

## 📦 インストールと設定

### 必要な依存関係

```bash
pip install numpy pandas matplotlib mplfinance pyyaml numba
```

### 設定ファイル (config.yaml)

```yaml
binance_data:
  data_dir: "data/binance"
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
  timeframes:
    - "1h"
    - "4h"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
```

## 🎯 基本的な使用方法

### 1. 直接実行（最も簡単）

```bash
# 基本実行
python visualization/ultimate_efficiency_ratio_chart.py

# 設定ファイル指定
python visualization/ultimate_efficiency_ratio_chart.py --config config.yaml

# 特定期間の表示
python visualization/ultimate_efficiency_ratio_chart.py --start 2024-01-01 --end 2024-12-31

# チャート保存
python visualization/ultimate_efficiency_ratio_chart.py --output ultimate_er_chart.png
```

### 2. Pythonスクリプトでの使用

```python
from visualization.ultimate_efficiency_ratio_chart import UltimateEfficiencyRatioChart

# チャートオブジェクトの作成
chart = UltimateEfficiencyRatioChart()

# データ読み込み
chart.load_data_from_config('config.yaml')

# インジケーター計算
chart.calculate_indicators(
    period=14,
    src_type='hlc3',
    hilbert_window=12,
    her_window=16,
    slope_index=3,
    range_threshold=0.003
)

# チャート描画
chart.plot(
    title="Ultimate Efficiency Ratio V3.0",
    show_volume=True,
    show_quantum=True,
    show_hilbert=True
)
```

## ⚙️ パラメータ設定

### インジケーターパラメータ

| パラメータ | デフォルト | 説明 |
|------------|------------|------|
| `period` | 14 | 基本期間（従来ERとの互換性用） |
| `src_type` | 'hlc3' | 価格ソース ('close', 'hlc3', 'ohlc4') |
| `hilbert_window` | 12 | ヒルベルト変換ウィンドウ |
| `her_window` | 16 | ハイパー効率率ウィンドウ |
| `slope_index` | 3 | トレンド判定期間 |
| `range_threshold` | 0.003 | レンジ判定しきい値 |

### チャート表示オプション

| オプション | デフォルト | 説明 |
|------------|------------|------|
| `show_volume` | True | 出来高パネルを表示 |
| `show_quantum` | True | 量子解析パネルを表示 |
| `show_hilbert` | True | ヒルベルト変換パネルを表示 |
| `figsize` | (16, 14) | 図のサイズ |
| `style` | 'yahoo' | mplfinanceスタイル |

## 📊 チャートの読み方

### メインパネル（ローソク足）

- **緑色の三角マーカー（↑）**: 上昇トレンドシグナル
- **赤色の逆三角マーカー（↓）**: 下降トレンドシグナル

### 効率率パネル

- **緑色の線**: 上昇トレンド時のUltimate ER
- **赤色の線**: 下降トレンド時のUltimate ER
- **グレーの線**: レンジ相場時のUltimate ER
- **点線**: ハイパー効率率（Hyper ER）

**効率率の解釈:**
- **0.7以上**: 強いトレンド（高効率）
- **0.3-0.7**: 中程度のトレンド
- **0.3以下**: レンジ相場（低効率）

### トレンド強度パネル

- **0.7以上**: 強いトレンド
- **0.3-0.7**: 中程度のトレンド
- **0.3以下**: 弱いトレンド

### 量子解析パネル（オプション）

- **紫色の線**: 量子コヒーレンス（市場の量子的秩序）
- **マゼンタの点線**: 量子もつれ効果（価格間相関）

**量子効果の解釈:**
- **高コヒーレンス（0.7以上）**: 市場が高度に組織化、予測可能性が高い
- **高もつれ効果（0.7以上）**: 価格間の強い相関、システマティックな動き

### ヒルベルト変換パネル（オプション）

- **オレンジ色の線**: 瞬時振幅（正規化済み）
- **茶色の点線**: 瞬時位相（正規化済み）

## 🎯 実践的な使用例

### 例1: 短期トレーディング設定

```python
chart.calculate_indicators(
    period=7,           # 短期期間
    hilbert_window=6,   # 高感度
    her_window=10,      # 短期効率率
    slope_index=2,      # 敏感なトレンド判定
    range_threshold=0.005  # 厳しいレンジ判定
)
```

### 例2: 長期投資設定

```python
chart.calculate_indicators(
    period=30,          # 長期期間
    hilbert_window=20,  # 安定性重視
    her_window=25,      # 長期効率率
    slope_index=7,      # 慎重なトレンド判定
    range_threshold=0.001  # 緩いレンジ判定
)
```

### 例3: 量子効果重視設定

```python
chart.calculate_indicators(
    period=14,
    hilbert_window=12,  # ヒルベルト変換に重点
    her_window=16,
    slope_index=3,
    range_threshold=0.003
)

chart.plot(
    show_quantum=True,  # 量子解析表示
    show_hilbert=True,  # ヒルベルト変換表示
    figsize=(20, 18)    # 大きなサイズで詳細表示
)
```

## 🚀 高度な使用方法

### 複数時間軸分析

```python
# 短期分析
chart_1h = UltimateEfficiencyRatioChart()
chart_1h.load_data_from_config('config_1h.yaml')
chart_1h.calculate_indicators(period=7, hilbert_window=6)
chart_1h.plot(title="1時間足 - Ultimate ER", savefig="uer_1h.png")

# 長期分析
chart_4h = UltimateEfficiencyRatioChart()
chart_4h.load_data_from_config('config_4h.yaml')
chart_4h.calculate_indicators(period=21, hilbert_window=16)
chart_4h.plot(title="4時間足 - Ultimate ER", savefig="uer_4h.png")
```

### バックテスト用データ出力

```python
# インジケーター計算後
er_data = {
    'ultimate_er': chart.er_result.values,
    'trend_signals': chart.er_result.trend_signals,
    'trend_strength': chart.er_result.trend_strength,
    'quantum_coherence': chart.er_result.quantum_coherence
}

# DataFrameに変換
import pandas as pd
df_signals = pd.DataFrame(er_data, index=chart.data.index)

# CSV出力
df_signals.to_csv('ultimate_er_signals.csv')
```

## 📈 トレーディング戦略例

### 戦略1: トレンドフォロー

```python
# シグナル生成条件
def generate_signals(er_result):
    signals = []
    
    for i in range(len(er_result.values)):
        ultimate_er = er_result.values[i]
        trend_signal = er_result.trend_signals[i]
        quantum_coherence = er_result.quantum_coherence[i]
        
        # エントリー条件
        if (ultimate_er > 0.7 and 
            trend_signal == 1 and 
            quantum_coherence > 0.6):
            signals.append('BUY')
        elif (ultimate_er > 0.7 and 
              trend_signal == -1 and 
              quantum_coherence > 0.6):
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals
```

### 戦略2: 量子効果トレーディング

```python
# 量子もつれ効果を活用
def quantum_strategy(er_result):
    signals = []
    
    for i in range(len(er_result.values)):
        quantum_entanglement = er_result.quantum_entanglement[i]
        trend_strength = er_result.trend_strength[i]
        
        # 量子もつれが強い時の戦略
        if quantum_entanglement > 0.8:
            if trend_strength > 0.7:
                signals.append('STRONG_TREND')
            elif trend_strength < 0.3:
                signals.append('REVERSAL_EXPECTED')
            else:
                signals.append('WAIT')
        else:
            signals.append('NO_SIGNAL')
    
    return signals
```

## 🛠️ トラブルシューティング

### よくある問題と解決方法

1. **設定ファイルが見つからない**
   ```bash
   # 設定ファイルパスを確認
   python visualization/ultimate_efficiency_ratio_chart.py --config /path/to/config.yaml
   ```

2. **データが読み込めない**
   - Binanceデータディレクトリが正しく設定されているか確認
   - データファイルが存在するか確認

3. **計算が遅い**
   - データ量を減らす（期間を短縮）
   - パラメータを調整（ウィンドウサイズを小さく）

4. **メモリエラー**
   - 大きなデータセットの場合は分割処理
   - figsize を小さくする

### パフォーマンス最適化

```python
# メモリ効率の良い設定
chart.calculate_indicators(
    hilbert_window=8,    # 小さなウィンドウ
    her_window=12,       # 小さなウィンドウ
)

chart.plot(
    figsize=(12, 10),    # 小さなサイズ
    show_quantum=False,  # 重い計算をスキップ
    show_hilbert=False
)
```

## 📚 関連ドキュメント

- [Ultimate Efficiency Ratio 技術仕様](ultimate_efficiency_ratio_specification.md)
- [量子効果アルゴリズム詳細](quantum_algorithms_detail.md)
- [拡張アイデア集](ultimate_efficiency_ratio_enhancement_ideas.md)

## 🤝 サポート

ご質問やバグ報告は、プロジェクトのIssueトラッカーまでお願いします。

---

**🚀 Ultimate Efficiency Ratio V3.0 で次世代のテクニカル分析を体験してください！** 