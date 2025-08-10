# LightGBM ATRリスク調整トレンドフォローモデル V1

## 概要

TTF（Trading Technical Framework）システム内で動作する高度な機械学習ベースのトレンドフォローモデルです。量子アルゴリズム系インジケーターとATRベースのリスク調整を組み合わせた革新的なアプローチを採用しています。

## 主な特徴

### 🎯 **革新的な目的変数設計**
- **ATRベースリスク調整**: 5期間ATRによる市場ボラティリティ正規化
- **実践的損益基準**: 3ATR損切り・7ATR利益確定
- **時間制約**: 300ローソク足以内での達成条件
- **3クラス分類**: 買い成功(+1)、売り成功(-1)、失敗・中立(0)

### 📊 **量子アルゴリズム系特徴量**
- **ハイパーMAMA**: 適応型移動平均（正規化済み）
- **ハイパーER**: 効率比指標（0-1範囲）
- **FRAMA**: フラクタル適応移動平均（正規化済み）
- **フェーザートレンド**: フェーザー分析トレンド強度（0-1範囲）
- **ハイパートレンドインデックス**: 高度なトレンド指標（正規化済み）
- **X_ATR**: 拡張ATR（正規化済み）
- **差分特徴量**: 各指標の1,3,5,8,34,55,89,144期間差分
- **合計54次元**: 高品質特徴空間

### 🤖 **LightGBM多クラス分類**
- **3クラス同時予測**: 買い・売り・中立の統合判定
- **時系列交差検証**: 金融データに適した検証手法
- **ウォークフォワード分析**: 70%訓練・10%スライドの動的評価
- **特徴量重要度分析**: SHAP値による詳細解析

### 📈 **包括的評価システム**
- **分類性能**: Accuracy, Precision, Recall, F1-Score, AUC
- **トレーディング指標**: Signal Accuracy, Hit Rate, False Positive Rate
- **リスク評価**: シャープレシオ, 最大ドローダウン, 連続失敗分析

### 💼 **実践的バックテスト**
- **高信頼度シグナル**: 60%以上の確信度でのみエントリー
- **リスクパリティ**: ATRベースの適応的ポジションサイジング
- **エクイティカーブ**: 詳細な資産推移分析

## ディレクトリ構造

```
ml/trend_follow_v1/
├── main.py                    # メインパイプライン
├── data_pipeline.py           # データ処理
├── feature_engineering.py    # 特徴量エンジニアリング
├── target_calculation.py     # 目的変数計算
├── model_training.py          # モデル訓練
├── evaluation.py              # 評価指標
├── backtesting.py             # バックテスト
├── config/
│   ├── model_config.yaml      # モデル設定
│   └── feature_config.yaml    # 特徴量設定
├── models/                    # 訓練済みモデル保存
├── results/                   # 結果出力
└── README.md                  # 本ファイル
```

## 使用方法

### 基本的な使用法

```python
from ml.trend_follow_v1.main import TrendFollowV1Pipeline

# パイプライン初期化
pipeline = TrendFollowV1Pipeline()

# 完全パイプライン実行
results = pipeline.run_full_pipeline()

# 結果確認
print(f"モデル精度: {results['model_performance']['accuracy']:.4f}")
print(f"総リターン: {results['backtest_performance']['total_return']:.2%}")
```

### 個別コンポーネントの使用

```python
from ml.trend_follow_v1 import (
    TrendFollowDataPipeline,
    TrendFollowFeatureEngineering,
    TrendFollowTargetCalculation
)

# データ処理
data_pipeline = TrendFollowDataPipeline()
data = data_pipeline.load_market_data()

# 特徴量計算
feature_eng = TrendFollowFeatureEngineering()
features = feature_eng.calculate_all_features(data)

# 目的変数計算
target_calc = TrendFollowTargetCalculation()
targets = target_calc.calculate_atr_normalized_target(data)
```

### コマンドライン実行

```bash
# メインパイプライン実行
cd /home/vapor/dev/ttf
python -m ml.trend_follow_v1.main

# 個別コンポーネントテスト
python ml/trend_follow_v1/feature_engineering.py
python ml/trend_follow_v1/target_calculation.py
```

## 設定ファイル

### model_config.yaml
- LightGBMパラメータ
- 交差検証設定
- バックテスト設定
- 評価設定

### feature_config.yaml
- 6つのコアインジケーター設定
- 差分特徴量期間設定
- データ前処理設定

## 性能目標

- **年間シャープレシオ**: > 1.5
- **最大ドローダウン**: < 15%
- **勝率**: > 60%
- **利益因子**: > 1.5
- **ATRベース成功率**: > 40%

## 技術仕様

### 依存関係
- Python 3.8+
- LightGBM
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- TTF システム

### データ要件
- OHLCV形式の価格データ
- 最低1000行のデータ推奨
- 設定ファイル対応のBinanceデータソース

### 計算要件
- メモリ: 4GB以上推奨
- CPU: マルチコア推奨
- 実行時間: 大規模データで数分〜数十分

## 実装詳細

### 特徴量エンジニアリング
1. **インジケーター計算**: 6つの量子アルゴリズム系指標
2. **正規化処理**: 価格系指標の終値正規化
3. **差分特徴量**: 8期間の時系列差分
4. **NaN処理**: 前方・後方埋めによる欠損値処理

### 目的変数計算
1. **ATR計算**: 5期間のTrue Range平均
2. **シグナル評価**: MAMA位置関係に基づく判定
3. **損益判定**: 3ATR損切り・7ATR利益確定
4. **時間制限**: 300ローソク足以内の達成条件

### モデル訓練
1. **データ分割**: 70%訓練・15%検証・15%テスト
2. **時系列CV**: 5分割時系列交差検証
3. **ウォークフォワード**: 70%訓練・10%スライド
4. **ハイパーパラメータ**: LightGBM多クラス分類最適化

### バックテスト
1. **シグナル生成**: 60%以上の確信度フィルター
2. **ポジション管理**: リスクパリティベース
3. **エグジット**: 利確・損切り・時間制限・反対シグナル
4. **評価**: 包括的なリスク・リターン分析

## トラブルシューティング

### よくある問題

1. **インジケーター計算エラー**
   - データの最小期間不足の可能性
   - インジケーターライブラリの互換性確認

2. **メモリ不足**
   - データサイズの縮小
   - バッチ処理の検討

3. **収束しないモデル**
   - 学習率の調整
   - 特徴量の正規化確認

### ログ確認
```bash
tail -f ml/trend_follow_v1/results/training.log
```

## 更新履歴

### v1.0.0 (2025-01-27)
- 初回リリース
- 完全な仕様書実装
- 6つのコアインジケーター統合
- ATRベース目的変数実装
- LightGBM多クラス分類
- 包括的評価・バックテストシステム

## ライセンス

TTF（Trading Technical Framework）プロジェクトの一部として提供されます。

## 開発者

TTF Development Team

---

**注意**: このモデルは教育・研究目的で開発されました。実際の取引での使用は自己責任で行ってください。