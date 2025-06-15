#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from ultimate_trend_range_detector_v3_enhanced import UltimateTrendRangeDetectorV3Enhanced
import time
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def load_data_from_config(config_path: str) -> pd.DataFrame:
    """
    設定ファイルから実際の相場データを読み込む
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        処理済みのデータフレーム
    """
    # 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("\n📊 実際の相場データを読み込み・処理中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初のシンボルのデータを取得
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"✅ 実際の相場データ読み込み完了: {first_symbol}")
    print(f"📅 期間: {data.index.min()} → {data.index.max()}")
    print(f"📊 データ数: {len(data)}")
    
    return data


def generate_enhanced_market_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    V3エンハンス用の市場データ生成
    """
    np.random.seed(42)
    
    data = []
    current_price = 100.0
    market_state = 'range'
    state_duration = 0
    volatility_regime = 'normal'
    
    for i in range(n_samples):
        # 市場状態の管理
        if state_duration <= 0:
            # 新しい状態を決定
            if market_state == 'range':
                market_state = np.random.choice(['trend_up', 'trend_down', 'range'], 
                                               p=[0.35, 0.35, 0.30])
            else:
                market_state = np.random.choice(['range', 'trend_up', 'trend_down'], 
                                               p=[0.50, 0.25, 0.25])
            
            # 状態の持続期間
            if 'trend' in market_state:
                state_duration = np.random.randint(60, 201)  # 60-200期間の長期トレンド
            else:
                state_duration = np.random.randint(30, 121)  # 30-120期間のレンジ
            
            # ボラティリティレジームの変更
            volatility_regime = np.random.choice(['low', 'normal', 'high'], 
                                                p=[0.3, 0.5, 0.2])
        
        # ボラティリティレジームに応じた基本変動
        if volatility_regime == 'low':
            base_vol = 0.008
        elif volatility_regime == 'normal':
            base_vol = 0.015
        else:  # high
            base_vol = 0.025
        
        # 市場状態に応じた価格生成
        if market_state == 'trend_up':
            # 上昇トレンド
            trend_strength = np.random.uniform(0.0015, 0.005)
            noise_factor = 0.6  # トレンド時はノイズを抑制
            price_change = (trend_strength * current_price + 
                           np.random.normal(0, current_price * base_vol * noise_factor))
            true_regime = 1
            
        elif market_state == 'trend_down':
            # 下降トレンド
            trend_strength = np.random.uniform(-0.005, -0.0015)
            noise_factor = 0.6
            price_change = (trend_strength * current_price + 
                           np.random.normal(0, current_price * base_vol * noise_factor))
            true_regime = 1
            
        else:  # range
            # レンジ相場（平均回帰特性）
            if i >= 30:
                recent_mean = np.mean([d['close'] for d in data[-30:]])
                mean_reversion_force = (recent_mean - current_price) * 0.08
            else:
                mean_reversion_force = 0
            
            noise_factor = 1.2  # レンジ時はノイズを増加
            price_change = (mean_reversion_force + 
                           np.random.normal(0, current_price * base_vol * noise_factor))
            true_regime = 0
        
        # 価格更新
        current_price += price_change
        current_price = max(current_price, 10.0)
        
        # OHLC生成（より現実的な日内変動）
        intraday_vol = current_price * base_vol * 0.8
        high_bias = np.random.uniform(0.2, 1.5)
        low_bias = np.random.uniform(0.2, 1.5)
        
        high = current_price + intraday_vol * high_bias
        low = current_price - intraday_vol * low_bias
        
        # Open価格は前の終値に近い値
        if i > 0:
            prev_close = data[-1]['close']
            gap = np.random.normal(0, current_price * 0.005)  # 小さなギャップ
            open_price = prev_close + gap
        else:
            open_price = current_price
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'true_regime': true_regime,
            'market_state': market_state,
            'volatility_regime': volatility_regime
        })
        
        state_duration -= 1
    
    return pd.DataFrame(data)


def evaluate_performance_enhanced(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """
    エンハンス版パフォーマンス評価
    """
    # 基本統計
    correct = np.sum(predicted == actual)
    total = len(predicted)
    accuracy = correct / total
    
    # 混同行列
    tp = np.sum((predicted == 1) & (actual == 1))  # True Positive (トレンド正解)
    tn = np.sum((predicted == 0) & (actual == 0))  # True Negative (レンジ正解)
    fp = np.sum((predicted == 1) & (actual == 0))  # False Positive (レンジをトレンドと誤判定)
    fn = np.sum((predicted == 0) & (actual == 1))  # False Negative (トレンドをレンジと誤判定)
    
    # 詳細メトリクス
    precision_trend = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_range = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_trend = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_range = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    f1_trend = 2 * (precision_trend * recall_trend) / (precision_trend + recall_trend) if (precision_trend + recall_trend) > 0 else 0
    f1_range = 2 * (precision_range * recall_range) / (precision_range + recall_range) if (precision_range + recall_range) > 0 else 0
    
    # バランス精度
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Matthews相関係数（MCC）
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision_trend': precision_trend,
        'precision_range': precision_range,
        'recall_trend': recall_trend,
        'recall_range': recall_range,
        'f1_trend': f1_trend,
        'f1_range': f1_range,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    }


def plot_results_enhanced_with_real_data(data: pd.DataFrame, results: dict, 
                                        actual_signals: Optional[np.ndarray] = None,
                                        save_path: str = None):
    """
    実際の相場データ用の高度な結果可視化（背景色付き）
    """
    fig, axes = plt.subplots(5, 1, figsize=(18, 14))
    
    # 1. 価格とシグナル
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], label='Close Price', alpha=0.8, linewidth=1.5, color='black')
    
    # 予測結果を背景色で表示（実際のデータ用）
    trend_mask_pred = results['signal'] == 1
    range_mask_pred = results['signal'] == 0
    
    # 背景色を追加
    ax1.fill_between(data.index, data['close'].min(), data['close'].max(), 
                     where=trend_mask_pred, alpha=0.1, color='green', 
                     label='Predicted Trend Periods', interpolate=True)
    ax1.fill_between(data.index, data['close'].min(), data['close'].max(), 
                     where=range_mask_pred, alpha=0.1, color='red', 
                     label='Predicted Range Periods', interpolate=True)
    
    # V3エンハンス予測信号をマーカーで表示
    trend_pred_idx = np.where(results['signal'] == 1)[0]
    range_pred_idx = np.where(results['signal'] == 0)[0]
    
    if len(trend_pred_idx) > 0:
        ax1.scatter(data.index[trend_pred_idx], data['close'].iloc[trend_pred_idx], 
                   c='darkgreen', marker='^', s=20, alpha=0.8, label='V3 Enhanced Trend Signals')
    if len(range_pred_idx) > 0:
        ax1.scatter(data.index[range_pred_idx], data['close'].iloc[range_pred_idx], 
                   c='darkred', marker='v', s=20, alpha=0.8, label='V3 Enhanced Range Signals')
    
    ax1.set_title('🚀 V3.0 Enhanced - Real Market Data Analysis with Predicted Backgrounds', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 主要指標ダッシュボード
    ax2 = axes[1]
    ax2.plot(data.index, results['efficiency_ratio'], label='Enhanced Efficiency Ratio', color='blue', alpha=0.8)
    ax2.plot(data.index, results['choppiness_index']/100, label='Choppiness Index (normalized)', color='red', alpha=0.8)
    ax2.plot(data.index, results['adx']/100, label='ADX (normalized)', color='purple', alpha=0.8)
    
    # 背景色を指標チャートにも追加
    ax2.fill_between(data.index, 0, 1, where=trend_mask_pred, alpha=0.05, color='green', interpolate=True)
    ax2.fill_between(data.index, 0, 1, where=range_mask_pred, alpha=0.05, color='red', interpolate=True)
    
    ax2.axhline(y=0.618, color='green', linestyle='--', alpha=0.5, label='Golden Ratio')
    ax2.axhline(y=0.382, color='orange', linestyle='--', alpha=0.5, label='Silver Ratio')
    ax2.set_title('📊 Enhanced Core Indicators Dashboard', fontweight='bold')
    ax2.set_ylabel('Indicator Values', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 信頼度と適応的閾値
    ax3 = axes[2]
    ax3.plot(data.index, results['confidence'], label='V3 Enhanced Confidence Score', color='darkblue', linewidth=2)
    ax3.plot(data.index, results['adaptive_threshold'], label='Dynamic Adaptive Threshold', color='purple', 
             linestyle='--', alpha=0.8)
    
    # 背景色を信頼度チャートにも追加
    ax3.fill_between(data.index, 0, 1, where=trend_mask_pred, alpha=0.05, color='green', interpolate=True)
    ax3.fill_between(data.index, 0, 1, where=range_mask_pred, alpha=0.05, color='red', interpolate=True)
    
    ax3.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='High Confidence Level')
    ax3.fill_between(data.index, 0, results['confidence'], alpha=0.2, color='blue')
    ax3.set_title('🧠 Enhanced Confidence & Dynamic Intelligence', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. エンハンス補助指標
    ax4 = axes[3]
    ax4.plot(data.index, results['momentum_consistency'], label='Enhanced Momentum Consistency', 
             color='darkgreen', alpha=0.8)
    ax4.plot(data.index, results['volatility_adjustment'], label='Volatility Adjustment', 
             color='orange', alpha=0.8)
    
    # 背景色を補助指標チャートにも追加
    ax4.fill_between(data.index, 0, 2, where=trend_mask_pred, alpha=0.05, color='green', interpolate=True)
    ax4.fill_between(data.index, 0, 2, where=range_mask_pred, alpha=0.05, color='red', interpolate=True)
    
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong Momentum')
    ax4.set_title('🎯 Enhanced Market Analysis', fontweight='bold')
    ax4.set_ylabel('Factor Values', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. シグナル分布と統計
    ax5 = axes[4]
    
    # シグナルの時系列表示
    signal_colors = ['red' if s == 0 else 'green' for s in results['signal']]
    ax5.scatter(data.index, results['signal'], c=signal_colors, alpha=0.6, s=10)
    ax5.set_ylim(-0.5, 1.5)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Range', 'Trend'])
    
    # 背景色をシグナルチャートにも追加
    ax5.fill_between(data.index, -0.5, 1.5, where=trend_mask_pred, alpha=0.1, color='green', 
                     interpolate=True, label='Trend Periods')
    ax5.fill_between(data.index, -0.5, 1.5, where=range_mask_pred, alpha=0.1, color='red', 
                     interpolate=True, label='Range Periods')
    
    # 信頼度による色分け
    high_conf_mask = results['confidence'] >= 0.8
    med_conf_mask = (results['confidence'] >= 0.6) & (results['confidence'] < 0.8)
    
    if np.sum(high_conf_mask) > 0:
        ax5.scatter(data.index[high_conf_mask], results['signal'][high_conf_mask], 
                   c='gold', marker='*', s=30, alpha=0.8, label='High Confidence')
    
    ax5.set_title('📈 Signal Distribution & Confidence Analysis with Background', fontweight='bold')
    ax5.set_ylabel('Signal', fontweight='bold')
    ax5.set_xlabel('Time', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 チャートを保存しました: {save_path}")
    plt.show()


def analyze_real_market_performance(data: pd.DataFrame, results: dict) -> dict:
    """
    実際の相場データでのパフォーマンス分析
    """
    analysis = {}
    
    # シグナル統計
    total_signals = len(results['signal'])
    trend_signals = np.sum(results['signal'] == 1)
    range_signals = np.sum(results['signal'] == 0)
    
    analysis['signal_distribution'] = {
        'total': total_signals,
        'trend_count': trend_signals,
        'range_count': range_signals,
        'trend_ratio': trend_signals / total_signals,
        'range_ratio': range_signals / total_signals
    }
    
    # 信頼度分析
    high_conf_count = np.sum(results['confidence'] >= 0.8)
    med_conf_count = np.sum((results['confidence'] >= 0.6) & (results['confidence'] < 0.8))
    low_conf_count = np.sum(results['confidence'] < 0.6)
    
    analysis['confidence_distribution'] = {
        'high_confidence': high_conf_count,
        'medium_confidence': med_conf_count,
        'low_confidence': low_conf_count,
        'high_conf_ratio': high_conf_count / total_signals,
        'avg_confidence': np.mean(results['confidence'])
    }
    
    # 指標統計
    analysis['indicator_stats'] = {
        'er_mean': np.mean(results['efficiency_ratio'][results['efficiency_ratio'] > 0]),
        'er_std': np.std(results['efficiency_ratio'][results['efficiency_ratio'] > 0]),
        'chop_mean': np.mean(results['choppiness_index'][results['choppiness_index'] > 0]),
        'chop_std': np.std(results['choppiness_index'][results['choppiness_index'] > 0]),
        'adx_mean': np.mean(results['adx'][results['adx'] > 0]),
        'adx_std': np.std(results['adx'][results['adx'] > 0])
    }
    
    # シグナル遷移分析
    signal_changes = np.diff(results['signal'])
    trend_to_range = np.sum(signal_changes == -1)
    range_to_trend = np.sum(signal_changes == 1)
    
    analysis['signal_transitions'] = {
        'trend_to_range': trend_to_range,
        'range_to_trend': range_to_trend,
        'total_transitions': trend_to_range + range_to_trend,
        'avg_signal_duration': total_signals / (trend_to_range + range_to_trend) if (trend_to_range + range_to_trend) > 0 else 0
    }
    
    return analysis


def plot_results_v3(data: pd.DataFrame, results: dict, save_path: str = None):
    """
    V3版の高度な結果可視化
    """
    fig, axes = plt.subplots(6, 1, figsize=(18, 16))
    
    # 1. 価格とシグナル比較
    ax1 = axes[0]
    ax1.plot(data['close'], label='Close Price', alpha=0.8, linewidth=1.5, color='black')
    
    # 真の市場状態を背景色で表示
    trend_mask_true = data['true_regime'] == 1
    range_mask_true = data['true_regime'] == 0
    
    ax1.fill_between(range(len(data)), data['close'].min(), data['close'].max(), 
                     where=trend_mask_true, alpha=0.1, color='green', label='True Trend Periods')
    ax1.fill_between(range(len(data)), data['close'].min(), data['close'].max(), 
                     where=range_mask_true, alpha=0.1, color='red', label='True Range Periods')
    
    # V3予測信号をマーカーで表示
    trend_pred_idx = np.where(results['signal'] == 1)[0]
    range_pred_idx = np.where(results['signal'] == 0)[0]
    
    if len(trend_pred_idx) > 0:
        ax1.scatter(trend_pred_idx, data['close'].iloc[trend_pred_idx], 
                   c='darkgreen', marker='^', s=15, alpha=0.7, label='V3 Trend Signals')
    if len(range_pred_idx) > 0:
        ax1.scatter(range_pred_idx, data['close'].iloc[range_pred_idx], 
                   c='darkred', marker='v', s=15, alpha=0.7, label='V3 Range Signals')
    
    ax1.set_title('🚀 V3.0 Ultimate - Price & Revolutionary Signal Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 主要指標ダッシュボード
    ax2 = axes[1]
    ax2.plot(results['efficiency_ratio'], label='Efficiency Ratio', color='blue', alpha=0.8)
    ax2.plot(results['choppiness_index']/100, label='Choppiness Index (normalized)', color='red', alpha=0.8)
    ax2.plot(results['adx']/100, label='ADX (normalized)', color='purple', alpha=0.8)
    ax2.axhline(y=0.618, color='green', linestyle='--', alpha=0.5, label='Golden Ratio')
    ax2.axhline(y=0.382, color='orange', linestyle='--', alpha=0.5, label='Silver Ratio')
    ax2.set_title('📊 Core Indicators Dashboard', fontweight='bold')
    ax2.set_ylabel('Indicator Values', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 信頼度と適応的閾値
    ax3 = axes[2]
    ax3.plot(results['confidence'], label='V3 Confidence Score', color='darkblue', linewidth=2)
    ax3.plot(results['adaptive_threshold'], label='Adaptive Threshold', color='purple', 
             linestyle='--', alpha=0.8)
    ax3.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='High Confidence Level')
    ax3.fill_between(range(len(results['confidence'])), 0, results['confidence'], 
                     alpha=0.2, color='blue')
    ax3.set_title('🧠 Confidence & Adaptive Intelligence', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. モメンタム一貫性とボラティリティ調整
    ax4 = axes[3]
    ax4.plot(results['momentum_consistency'], label='Momentum Consistency', 
             color='darkgreen', alpha=0.8)
    ax4.plot(results['volatility_adjustment'], label='Volatility Adjustment', 
             color='orange', alpha=0.8)
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('🎯 Advanced Market Analysis', fontweight='bold')
    ax4.set_ylabel('Factor Values', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 判定精度の時系列分析
    ax5 = axes[4]
    # 移動平均精度を計算
    window = 100
    rolling_accuracy = []
    for i in range(window, len(data)):
        pred_window = results['signal'][i-window:i]
        actual_window = data['true_regime'].values[i-window:i]
        acc = np.sum(pred_window == actual_window) / window
        rolling_accuracy.append(acc)
    
    # プロットの調整
    x_rolling = range(window, len(data))
    ax5.plot(x_rolling, rolling_accuracy, color='red', linewidth=2, 
             label=f'Rolling Accuracy ({window}-period)')
    ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Target')
    ax5.axhline(y=np.mean(rolling_accuracy), color='blue', linestyle=':', 
               alpha=0.7, label=f'Average: {np.mean(rolling_accuracy):.3f}')
    ax5.fill_between(x_rolling, 0.8, rolling_accuracy, 
                     where=np.array(rolling_accuracy) >= 0.8, 
                     alpha=0.3, color='green', label='Above Target')
    ax5.set_title('📈 Real-time Accuracy Performance', fontweight='bold')
    ax5.set_ylabel('Accuracy', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. エラー分析
    ax6 = axes[5]
    errors = (results['signal'] != data['true_regime'].values).astype(int)
    cumulative_error_rate = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    
    ax6.plot(cumulative_error_rate, color='red', linewidth=2, label='Cumulative Error Rate')
    ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='20% Error Target')
    ax6.fill_between(range(len(cumulative_error_rate)), 0, errors, 
                     alpha=0.2, color='red', label='Individual Errors')
    ax6.set_title('🔍 Error Analysis & Learning Curve', fontweight='bold')
    ax6.set_ylabel('Error Rate', fontweight='bold')
    ax6.set_xlabel('Time', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def main():
    """
    V3エンハンス版メイン実行 - バランス調整版
    """
    print("🚀 人類史上最強トレンド/レンジ判別インジケーター V3.0 ENHANCED - BALANCED EDITION")
    print("=" * 120)
    print("🎯 目標: 実用的で高精度なバランス判別")
    print("💎 バランス技術: 適度閾値 + 実用ER + 柔軟判定 + 実用性重視")
    print("📊 新機能: 実際の相場データテスト対応")
    print("=" * 120)
    
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description='V3エンハンス版 トレンド/レンジ判別テスト（バランス調整版）')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--real-data', '-r', action='store_true', help='実際の相場データを使用')
    parser.add_argument('--synthetic', '-s', action='store_true', help='合成データを使用（デフォルト）')
    parser.add_argument('--output', '-o', type=str, help='結果画像の保存パス')
    args = parser.parse_args()
    
    # データの選択
    if args.real_data:
        print("\n📊 実際の相場データモードを選択")
        try:
            data = load_data_from_config(args.config)
            data_type = "実際の相場データ"
            
            # 実際のデータには真の市場状態がないため、ダミーを作成
            print("⚠️  注意: 実際の相場データでは真の市場状態が不明のため、パフォーマンス評価は参考値です")
            
        except Exception as e:
            print(f"❌ 実際の相場データの読み込みに失敗: {e}")
            print("🔄 合成データに切り替えます...")
            data = generate_enhanced_market_data(2500)
            data_type = "合成データ（フォールバック）"
    else:
        print("\n📊 合成データモードを選択")
        data = generate_enhanced_market_data(2500)
        data_type = "高度な合成データ"
    
    print(f"✅ データ準備完了: {data_type}")
    print(f"   📈 データ数: {len(data)}件")
    
    # データ統計の表示
    if 'true_regime' in data.columns:
        actual_trend_count = sum(data['true_regime'])
        actual_range_count = len(data) - actual_trend_count
        print(f"   📈 真のトレンド期間: {actual_trend_count}件 ({actual_trend_count/len(data)*100:.1f}%)")
        print(f"   📉 真のレンジ期間: {actual_range_count}件 ({actual_range_count/len(data)*100:.1f}%)")
        
        if 'market_state' in data.columns:
            state_dist = data['market_state'].value_counts()
            print(f"   🔄 市場状態分布: {dict(state_dist)}")
    else:
        print("   📊 実際の相場データ - 価格範囲:")
        print(f"      最高値: {data['high'].max():.2f}")
        print(f"      最安値: {data['low'].min():.2f}")
        print(f"      終値範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 2. V3エンハンスバランス版初期化
    print("\n🔧 Ultimate V3 Enhanced Balanced 初期化中...")
    detector_enhanced = UltimateTrendRangeDetectorV3Enhanced(
        er_period=20,      # バランス調整された期間
        chop_period=14,    # 標準期間
        adx_period=14,     # 標準期間
        vol_period=18      # バランス調整された期間
    )
    print("✅ V3エンハンスバランス版初期化完了")
    print("   🧠 バランス構成: ER(35%) + Chop(30%) + ADX(25%) + Momentum(10%)")
    print("   ⚡ バランス機能: 適度閾値 + 実用判定 + 柔軟フィルタ + 実用性重視")
    
    # 3. 計算実行
    print(f"\n⚡ V3エンハンスバランス版 {data_type}解析実行中...")
    start_time = time.time()
    
    results = detector_enhanced.calculate(data)
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    print(f"✅ 計算完了 (処理時間: {calculation_time:.2f}秒)")
    print(f"   ⚡ 処理速度: {len(data)/calculation_time:.0f} データ/秒")
    
    # 4. パフォーマンス評価（合成データの場合のみ）
    if 'true_regime' in data.columns:
        print("\n📈 V3エンハンスバランス版パフォーマンス評価中...")
        
        # 初期期間をスキップして評価
        skip_initial = 50
        predicted_signals = results['signal'][skip_initial:]
        actual_signals = data['true_regime'].values[skip_initial:]
        
        performance = evaluate_performance_enhanced(predicted_signals, actual_signals)
        
        # パフォーマンス結果表示
        print("\n" + "="*90)
        print("🏆 **V3.0 ENHANCED BALANCED PERFORMANCE RESULTS**")
        print("="*90)
        print(f"📊 総合精度: {performance['accuracy']:.4f} ({performance['accuracy']*100:.2f}%)")
        print(f"⚖️  バランス精度: {performance['balanced_accuracy']:.4f} ({performance['balanced_accuracy']*100:.2f}%)")
        print(f"💎 MCC（品質指標）: {performance['mcc']:.4f}")
        
        # 評価判定
        if performance['accuracy'] >= 0.75:
            print(f"🎉🏆 **バランス設定で75%以上達成！実用性と精度の完璧な両立！** 🏆🎉")
            achievement_status = "BALANCED SUCCESS"
        elif performance['accuracy'] >= 0.70:
            print(f"⭐🔥 **70%突破！バランス取れた高性能！** 🔥⭐")
            achievement_status = "PRACTICAL ACHIEVEMENT"
        else:
            print(f"📈💫 **バランス調整により実用性向上中...** 💫📈")
            achievement_status = "PRACTICAL PROGRESS"
        
        print(f"\n📈 **トレンド判別詳細**")
        print(f"   - 精度 (Precision): {performance['precision_trend']:.4f} ({performance['precision_trend']*100:.1f}%)")
        print(f"   - 再現率 (Recall): {performance['recall_trend']:.4f} ({performance['recall_trend']*100:.1f}%)")
        print(f"   - F1スコア: {performance['f1_trend']:.4f}")
        
        print(f"\n📉 **レンジ判別詳細**")
        print(f"   - 精度 (Precision): {performance['precision_range']:.4f} ({performance['precision_range']*100:.1f}%)")
        print(f"   - 再現率 (Recall): {performance['recall_range']:.4f} ({performance['recall_range']*100:.1f}%)")
        print(f"   - F1スコア: {performance['f1_range']:.4f}")
    
    # 5. 実際の相場データ分析
    print("\n📊 実際の相場データ分析実行中...")
    market_analysis = analyze_real_market_performance(data, results)
    
    # 6. 技術統計
    print("\n" + "="*90)
    print("🔬 **V3エンハンスバランス版技術統計**")
    print("="*90)
    summary = results['summary']
    print(f"📊 予測統計:")
    print(f"   - トレンド期間: {summary['trend_bars']}件 ({summary['trend_ratio']*100:.1f}%)")
    print(f"   - レンジ期間: {summary['range_bars']}件 ({(1-summary['trend_ratio'])*100:.1f}%)")
    print(f"   - 平均信頼度: {summary['avg_confidence']:.4f}")
    print(f"   - 高信頼度比率: {summary['high_confidence_ratio']*100:.1f}%")
    
    print(f"\n🎯 バランス指標統計:")
    stats = market_analysis['indicator_stats']
    print(f"   - バランス効率比: 平均 {stats['er_mean']:.4f} ± {stats['er_std']:.4f}")
    print(f"   - Choppiness Index: 平均 {stats['chop_mean']:.2f} ± {stats['chop_std']:.2f}")
    print(f"   - ADX: 平均 {stats['adx_mean']:.2f} ± {stats['adx_std']:.2f}")
    
    # 7. シグナル分析
    signal_dist = market_analysis['signal_distribution']
    conf_dist = market_analysis['confidence_distribution']
    transitions = market_analysis['signal_transitions']
    
    print(f"\n🔄 **シグナル分析:**")
    print(f"   📈 トレンドシグナル: {signal_dist['trend_count']}件 ({signal_dist['trend_ratio']*100:.1f}%)")
    print(f"   📉 レンジシグナル: {signal_dist['range_count']}件 ({signal_dist['range_ratio']*100:.1f}%)")
    print(f"   🔄 シグナル遷移: {transitions['total_transitions']}回")
    print(f"   ⏱️  平均シグナル持続: {transitions['avg_signal_duration']:.1f}期間")
    
    # シグナル分布の妥当性チェック
    trend_ratio = signal_dist['trend_ratio']
    if 0.4 <= trend_ratio <= 0.65:
        print(f"   ✅ シグナル分布バランス: 良好 (トレンド{trend_ratio*100:.1f}%)")
    elif trend_ratio > 0.7:
        print(f"   ⚠️  シグナル分布: トレンド判定多め (トレンド{trend_ratio*100:.1f}%)")
    elif trend_ratio < 0.35:
        print(f"   ⚠️  シグナル分布: レンジ判定多め (トレンド{trend_ratio*100:.1f}%)")
    else:
        print(f"   ⚠️  シグナル分布: やや偏り有り (トレンド{trend_ratio*100:.1f}%)")
    
    print(f"\n💎 **信頼度分析:**")
    print(f"   - 高信頼度(≥80%): {conf_dist['high_confidence']}件 ({conf_dist['high_conf_ratio']*100:.1f}%)")
    print(f"   - 中信頼度(60-80%): {conf_dist['medium_confidence']}件")
    print(f"   - 低信頼度(<60%): {conf_dist['low_confidence']}件")
    print(f"   - 平均信頼度: {conf_dist['avg_confidence']:.4f}")
    
    # 8. 可視化
    print(f"\n📊 V3エンハンスバランス版 {data_type}結果可視化中...")
    output_path = args.output or f'ultimate_trend_range_v3_enhanced_balanced_{data_type.replace(" ", "_").lower()}_results.png'
    
    if 'true_regime' in data.columns:
        plot_results_v3(data, results, output_path)
    else:
        plot_results_enhanced_with_real_data(data, results, save_path=output_path)
    
    print(f"✅ 可視化完了 ({output_path})")
    
    # 9. 最終評価
    print("\n" + "="*100)
    print("🏆 **V3.0 ENHANCED BALANCED FINAL EVALUATION**")
    print("="*100)
    
    if 'true_regime' in data.columns:
        final_score = performance['accuracy']
        print(f"🎖️  最終評価: {achievement_status}")
        print(f"📊 総合精度: {final_score*100:.2f}%")
        print(f"📊 バランス精度: {performance['balanced_accuracy']*100:.2f}%")
        print(f"💎 品質指標(MCC): {performance['mcc']:.4f}")
        
        if final_score >= 0.75:
            print(f"\n🎉🏆 **バランス設定で75%以上達成！実用性・精度・バランスの三位一体！** 🏆🎉")
        elif final_score >= 0.70:
            print(f"\n⭐🔥 **実用的な70%以上の精度を実現！** 🔥⭐")
        else:
            print(f"\n📈💫 **バランス調整により実用性を重視した安定運用！** 💫📈")
    else:
        print(f"🎖️  実際の相場データ解析完了")
        print(f"📊 解析データ: {data_type}")
        print(f"📊 総シグナル数: {signal_dist['total']}")
        print(f"📊 平均信頼度: {conf_dist['avg_confidence']:.4f}")
        print(f"💎 高信頼度比率: {conf_dist['high_conf_ratio']*100:.1f}%")
        
        print(f"\n🚀 **バランス調整版による実用的な相場解析完了！**")
        print("💎 V3エンハンスバランス版により、実用性と精度を両立した判別を実現！")
    
    print(f"\n🌟 **バランス技術:**")
    print("   🚀 適度閾値システム: 基本0.40、範囲0.20-0.65の実用設定")
    print("   💎 実用効率比: 中程度トレンドも適度にブーストで実用検出")
    print("   🎯 柔軟モメンタム一貫性: バランス重み付き、0.5%閾値で実用検出")
    print("   🧠 実用多段階判定: 2指標以上一致の柔軟システム")
    print("   🔧 バランスノイズフィルタ: 実用性重視のバランスフィルタ")
    
    print("\nV3.0 ENHANCED BALANCED EDITION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main() 