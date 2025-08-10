# LightGBM ATRリスク調整トレンドフォローモデル仕様書

## 1. プロジェクト概要

### 1.1 目的
高度な技術指標を活用したLightGBM機械学習モデルにより、ATRベースのリスク調整を行った最適なトレンドフォロー戦略を実現する。5期間ATRによる正規化リターンを目的変数とし、実践的なトレーディングロジックに基づいた予測モデルを構築する。

### 1.2 コンセプト
- **ATRリスク正規化**: 市場ボラティリティを考慮したリスク調整リターン
- **実践的損益基準**: 3ATR損切り・7ATR利益確定の明確な基準
- **量子アルゴリズム系インジケーター**: 最新の量子効率理論とアダプティブフィルタリング

## 2. 特徴量設計

### 2.1 コアインジケーター特徴量

#### 2.1.1 正規化が必要な指標（終値で除算）
1. **ハイパーMAMA** (`indicators/hyper_mama.py`)
   - `mama_normalized = mama_value / close_price`
   - `fama_normalized = fama_value / close_price`

2. **FRAMA** (`indicators/mesa_frama.py` または `indicators/hyper_frama.py`)
   - `frama_normalized = frama_value / close_price`

3. **X_ATR** (`indicators/volatility/x_atr.py`)
   - `x_atr_normalized = x_atr_value / close_price`

#### 2.1.2 正規化不要な指標（既に正規化済み）
4. **ハイパーER** (`indicators/hyper_efficiency_ratio.py`)
   - `hyper_er_value` (0-1範囲)

5. **フェーザートレンド** (`indicators/trend_filter/phasor_trend_filter.py`)
   - `phasor_values` (0-1範囲のトレンド強度値)

6. **ハイパートレンドインデックス** (`indicators/hyper_trend_index.py`)
   - `hyper_trend_index_value` (正規化済み)

### 2.2 時系列差分特徴量

各コアインジケーターについて、以下の期間との差分を計算：
- 1日前、3日前、5日前、8日前、34日前、55日前、89日前、144日前

```python
# 例：ハイパーMAMAの差分特徴量
mama_diff_1 = mama_normalized - mama_normalized.shift(1)
mama_diff_3 = mama_normalized - mama_normalized.shift(3)
mama_diff_5 = mama_normalized - mama_normalized.shift(5)
# ... 144日前まで
```

### 2.3 特徴量総数
- コアインジケーター: 6種類
- 各インジケーターの差分: 8期間 × 6種類 = 48種類
- **合計特徴量数: 54種類**

## 3. 目的変数設計

### 3.1 ATRベースリスク調整リターン目的変数

#### 3.1.1 ATR正規化リターンスコア (ANRS: ATR-Normalized Return Score)

```python
def calculate_atr_normalized_target(data, mama_signals, atr_period=5, stop_atr_mult=3, target_atr_mult=7, max_bars=300):
    """
    ATRベースのリスク調整リターンを目的変数として計算
    
    Args:
        data: 価格データ (OHLCV)
        mama_signals: MAMA位置関係シグナル (+1: 買い, -1: 売り, 0: 中立)
        atr_period: ATR計算期間 (デフォルト: 5)
        stop_atr_mult: 損切りライン (デフォルト: 3ATR)
        target_atr_mult: 利益目標 (デフォルト: 7ATR)
        max_bars: 最大保有期間（ローソク足本数、デフォルト: 300）
    
    Returns:
        ANRS: ATR正規化リターンスコア (+1: 成功, -1: 成功, 0: 失敗・中立)
    """
    
    # 1. ATR計算 (5期間)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift(1))
    low_close = np.abs(data['low'] - data['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=atr_period).mean()
    
    target_values = np.zeros(len(data))
    
    for i in range(len(data) - 1):
        signal = mama_signals[i]
        if signal == 0 or np.isnan(atr.iloc[i]):
            continue
            
        current_atr = atr.iloc[i]
        
        if signal == 1:  # 買いシグナル
            entry_price = data['close'].iloc[i]
            stop_loss = data['low'].iloc[i] - (stop_atr_mult * current_atr)
            profit_target = entry_price + (target_atr_mult * current_atr)
            
            # 将来の価格動向をチェック（最大300本以内）
            max_achieved = False
            stop_hit = False
            max_check_bars = min(i + max_bars + 1, len(data))
            
            for j in range(i + 1, max_check_bars):
                low_price = data['low'].iloc[j]
                high_price = data['high'].iloc[j]
                
                # 損切りラインチェック
                if low_price <= stop_loss:
                    stop_hit = True
                    break
                
                # 利益目標達成チェック
                if high_price >= profit_target:
                    max_achieved = True
                    break
            
            # 結果判定（300本以内で利益目標達成かつ損切り未発生）
            if max_achieved and not stop_hit:
                target_values[i] = 1  # 成功
            else:
                target_values[i] = 0  # 失敗
                
        elif signal == -1:  # 売りシグナル
            entry_price = data['close'].iloc[i]
            stop_loss = data['high'].iloc[i] + (stop_atr_mult * current_atr)
            profit_target = entry_price - (target_atr_mult * current_atr)
            
            # 将来の価格動向をチェック（最大300本以内）
            max_achieved = False
            stop_hit = False
            max_check_bars = min(i + max_bars + 1, len(data))
            
            for j in range(i + 1, max_check_bars):
                low_price = data['low'].iloc[j]
                high_price = data['high'].iloc[j]
                
                # 損切りラインチェック
                if high_price >= stop_loss:
                    stop_hit = True
                    break
                
                # 利益目標達成チェック
                if low_price <= profit_target:
                    max_achieved = True
                    break
            
            # 結果判定（300本以内で利益目標達成かつ損切り未発生）
            if max_achieved and not stop_hit:
                target_values[i] = -1  # 成功
            else:
                target_values[i] = 0  # 失敗
    
    return target_values
```

#### 3.1.2 目的変数の特徴
- **リスク調整**: ATRベースの正規化により市場ボラティリティを考慮
- **明確な損益基準**: 3ATR損切り、7ATR利益確定の客観的基準
- **時間制約**: 300ローソク足以内での達成を条件とする実用的制限
- **3クラス分類**: 買い成功(+1)、売り成功(-1)、失敗・中立(0)
- **実践的**: 実際のトレーディングロジックに即した評価

#### 3.1.3 クラス分布調整

```python
def analyze_target_distribution(target_values):
    """
    目的変数の分布を分析
    """
    unique, counts = np.unique(target_values, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    total_signals = np.sum(target_values != 0)
    success_rate = np.sum(np.abs(target_values)) / total_signals if total_signals > 0 else 0
    
    print(f"目的変数分布: {distribution}")
    print(f"シグナル成功率: {success_rate:.2%}")
    
    return distribution, success_rate
```

## 4. データ処理フロー

### 4.1 データ取得
`visualization/z_adaptive_channel_chart.py`の実装を参考：

```python
def load_market_data(config_path: str) -> pd.DataFrame:
    """設定ファイルから実際の相場データを取得"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Binanceデータソースを使用
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # データローダーとプロセッサー
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データ処理
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    return processed_data
```

### 4.2 特徴量計算パイプライン

```python
def calculate_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """全ての特徴量を計算"""
    
    # 1. インジケーター初期化
    hyper_mama = HyperMAMA()
    hyper_er = HyperER()
    frama = FRAMA()  # または HyperFRAMA
    phasor_trend = PhasorTrendFilter()
    hyper_trend_idx = HyperTrendIndex()
    x_atr = X_ATR()
    
    # 2. インジケーター計算
    hyper_mama_result = hyper_mama.calculate(data)
    hyper_er_result = hyper_er.calculate(data)
    frama_result = frama.calculate(data)
    phasor_result = phasor_trend.calculate(data)
    hyper_trend_result = hyper_trend_idx.calculate(data)
    x_atr_result = x_atr.calculate(data)
    
    # 3. 正規化処理
    features = pd.DataFrame(index=data.index)
    
    # 正規化が必要な特徴量
    features['mama_norm'] = hyper_mama_result.mama / data['close']
    features['fama_norm'] = hyper_mama_result.fama / data['close']
    features['frama_norm'] = frama_result.values / data['close']
    features['x_atr_norm'] = x_atr_result.values / data['close']
    
    # 正規化不要な特徴量
    features['hyper_er'] = hyper_er_result.values
    features['phasor_values'] = phasor_result.values
    features['hyper_trend_idx'] = hyper_trend_result.values
    
    # 4. 差分特徴量生成
    lag_periods = [1, 3, 5, 8, 34, 55, 89, 144]
    
    for col in features.columns:
        for lag in lag_periods:
            features[f'{col}_diff_{lag}'] = features[col] - features[col].shift(lag)
    
    return features
```

### 4.3 目的変数計算

```python
def calculate_target(data: pd.DataFrame, mama_signals: np.ndarray) -> pd.Series:
    """ATR正規化リターンスコア(ANRS)を計算"""
    
    # 上記の calculate_atr_normalized_target 関数を使用
    anrs = calculate_atr_normalized_target(data, mama_signals)
    
    return anrs
```

## 5. モデル構築手法

### 5.1 LightGBMパラメータ

```python
lgb_params = {
    'objective': 'multiclass',  # 3クラス分類 (+1, -1, 0)
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'max_depth': 8,
    'num_boost_round': 1000,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'verbose': -1,
    'class_weight': 'balanced'  # クラス不均衡対策
}
```

### 5.2 交差検証戦略

#### 5.2.1 時系列分割検証
```python
from sklearn.model_selection import TimeSeriesSplit

# 5分割時系列交差検証
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(features):
    X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
    y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
    
    # モデル訓練・評価
```

#### 5.2.2 ウォークフォワード分析
- 訓練期間: 70% （CSVデータの最初の70%）
- 検証期間: 15% （CSVデータの次の15%）
- テスト期間: 15% （CSVデータの最後の15%）
- 更新頻度: 訓練期間の10%ずつスライド

### 5.3 特徴量重要度分析
- SHAP値による特徴量寄与度分析
- Permutation Importanceによる実用的重要度
- 時期別重要度変化の追跡

## 6. 評価方法

### 6.1 モデル性能評価指標

#### 6.1.1 多クラス分類性能
- **Accuracy**: 全体的な分類精度
- **Macro F1-Score**: 各クラスの平均F1スコア
- **Weighted F1-Score**: クラス頻度を考慮したF1スコア
- **Confusion Matrix**: 混同行列による詳細分析
- **Class-wise Precision/Recall**: クラス別の精密度・再現率

#### 6.1.2 トレーディング特化指標
- **Signal Accuracy**: シグナル方向の正解率
- **Risk-Adjusted Return**: ATRベースのリスク調整後リターン
- **Hit Rate**: 利益目標達成率
- **Stop Loss Rate**: 損切り回避率

#### 6.1.3 リスク評価
- **シャープレシオ**: リスク調整後リターンの品質
- **最大ドローダウン**: 最大損失幅
- **勝率**: 予測成功率
- **利益因子**: 総利益/総損失比

### 6.2 バックテスト設計

#### 6.2.1 シグナル生成ルール
```python
def generate_trading_signals(predictions: np.ndarray, 
                           confidence_threshold: float = 0.6) -> np.ndarray:
    """
    モデル予測値からトレーディングシグナルを生成
    
    Args:
        predictions: LightGBMモデルの予測確率 (3クラス分類)
        confidence_threshold: シグナル生成の信頼度閾値
    
    Returns:
        signals: +1(買い), -1(売り), 0(中立)
    """
    # 各クラスの予測確率を取得
    prob_buy = predictions[:, 1]    # 買い成功クラス(+1)の確率
    prob_sell = predictions[:, 2]   # 売り成功クラス(-1)の確率
    prob_neutral = predictions[:, 0] # 失敗・中立クラス(0)の確率
    
    signals = np.zeros(len(predictions))
    
    # 高い信頼度で成功が予測される場合のみシグナル生成
    buy_mask = (prob_buy >= confidence_threshold) & (prob_buy > prob_sell) & (prob_buy > prob_neutral)
    sell_mask = (prob_sell >= confidence_threshold) & (prob_sell > prob_buy) & (prob_sell > prob_neutral)
    
    signals[buy_mask] = 1   # 買いシグナル
    signals[sell_mask] = -1 # 売りシグナル
    
    return signals
```

#### 6.2.2 ポジション管理
- **エントリー**: 高信頼度で成功が予測されるシグナルのみ
- **エグジット**: 反対方向の高信頼度シグナル発生時、または300本経過時
- **ストップロス**: 3ATRベースの固定ストップ
- **利益確定**: 7ATRベースの利益目標
- **最大保有期間**: 300ローソク足（実用的な時間制約）
- **ポジションサイズ**: リスクパリティベース

### 6.3 アウトサンプルテスト

#### 6.3.1 データ分割方法
```python
def split_data_by_percentage(data, train_pct=70, val_pct=15, test_pct=15):
    """
    CSVデータを%ベースで分割
    
    Args:
        data: 全CSVデータ
        train_pct: 訓練データの割合（デフォルト: 70%）
        val_pct: 検証データの割合（デフォルト: 15%）
        test_pct: テストデータの割合（デフォルト: 15%）
    
    Returns:
        train_data, val_data, test_data
    """
    total_len = len(data)
    
    train_end = int(total_len * train_pct / 100)
    val_end = int(total_len * (train_pct + val_pct) / 100)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
```

#### 6.3.2 ウォークフォワード実装
```python
def walk_forward_analysis(data, train_pct=70, step_pct=10):
    """
    %ベースウォークフォワード分析
    
    Args:
        data: 全CSVデータ
        train_pct: 訓練期間の割合
        step_pct: スライドステップの割合
    """
    total_len = len(data)
    train_size = int(total_len * train_pct / 100)
    step_size = int(total_len * step_pct / 100)
    
    results = []
    
    for start_idx in range(0, total_len - train_size, step_size):
        train_end = start_idx + train_size
        test_end = min(train_end + step_size, total_len)
        
        train_data = data[start_idx:train_end]
        test_data = data[train_end:test_end]
        
        # モデル訓練・評価
        model_result = train_and_evaluate(train_data, test_data)
        results.append(model_result)
    
    return results
```

## 7. 実装方針

### 7.1 ディレクトリ構造
```
ml/
├── LIGHTGBM_TREND_FOLLOW_SPECIFICATION.md  # 本仕様書
├── data_pipeline.py                        # データ処理パイプライン
├── feature_engineering.py                  # 特徴量エンジニアリング
├── target_calculation.py                   # 目的変数計算
├── model_training.py                       # モデル訓練
├── backtesting.py                          # バックテスト実行
├── evaluation.py                           # 評価指標計算
├── config/
│   ├── model_config.yaml                   # モデル設定
│   └── feature_config.yaml                 # 特徴量設定
├── models/                                 # 訓練済みモデル保存
├── results/                                # 結果出力
└── notebooks/                              # 実験・分析用ノートブック
```

### 7.2 開発ステップ

#### Phase 1: データパイプライン構築
1. 設定ファイル対応データ取得機能
2. インジケーター計算パイプライン
3. 特徴量生成・正規化処理
4. 目的変数計算実装

#### Phase 2: モデル開発
1. LightGBMモデル実装
2. 交差検証システム
3. ハイパーパラメータ最適化
4. 特徴量選択アルゴリズム

#### Phase 3: 評価・最適化
1. バックテストシステム
2. 性能評価指標算出
3. リスク管理機能
4. レポート生成機能

### 7.3 技術スタック
- **機械学習**: LightGBM, scikit-learn
- **データ処理**: pandas, numpy
- **可視化**: matplotlib, seaborn, plotly
- **最適化**: optuna
- **並列処理**: joblib, multiprocessing

## 8. 期待される成果

### 8.1 性能目標
- **年間シャープレシオ**: > 1.5
- **最大ドローダウン**: < 15%
- **勝率**: > 60%
- **利益因子**: > 1.5
- **ATRベース成功率**: > 40%

### 8.2 イノベーションポイント
1. **ATRリスク正規化**: 市場ボラティリティに応じた適応的リスク管理
2. **実践的目的変数**: 3ATR損切り・7ATR利益確定の現実的な基準
3. **量子アルゴリズム系インジケーター**: 従来手法を超越した高精度予測
4. **多クラス分類**: 買い成功・売り成功・失敗の3クラス同時予測

### 8.3 期待される効果
1. **リスク調整リターン最大化**: ATRベースの正規化による一貫した収益性
2. **実践的トレーディング**: 明確な損益基準による再現可能な戦略
3. **市場適応性**: ボラティリティ変化に対応する動的リスク管理
4. **高精度予測**: 量子アルゴリズム系指標による優位性確保

---

**注記**: 本仕様書は、ATRベースのリスク調整を核とした実践的なトレンドフォローシステムの設計書です。量子アルゴリズム系インジケーターと機械学習を融合し、市場ボラティリティに適応する革新的なトレーディングシステムを構築します。