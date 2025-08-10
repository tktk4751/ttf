#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
シンプルスムーザー比較チャート

すべてのスムーザーを直接Numba関数で実装し、循環インポートを回避:
- UltimateSmoother: ジョン・エーラーズのアルティメットスムーザー
- SuperSmoother: 2極と3極のスーパースムーザー
- FRAMA: フラクタル適応移動平均
- UnscentedKalmanFilter: 無香料カルマンフィルター  
- AdaptiveKalman: 適応カルマンフィルター
"""

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yaml
from numba import njit
import math

# プロジェクトルートの追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config() -> Dict:
    """設定ファイルを読み込む"""
    config_path = project_root / "config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: config.yaml not found. Using default settings.")
        return {
            'binance_data': {
                'symbol': 'SOL',
                'timeframe': '4h',
                'start': '2023-01-01',
                'end': '2024-12-31'
            }
        }


def generate_sample_data(num_points: int = 1000) -> pd.DataFrame:
    """リアルな市場データ風のサンプルデータを生成"""
    np.random.seed(42)
    
    # 日付範囲を生成
    dates = pd.date_range(start='2023-01-01', periods=num_points, freq='4H')
    
    # 基本価格トレンド（多様なパターン）
    trend = np.linspace(100, 180, num_points)
    
    # ランダムウォーク
    random_walk = np.cumsum(np.random.randn(num_points) * 0.5)
    
    # 複数の周期的変動
    cycle1 = 8 * np.sin(2 * np.pi * np.arange(num_points) / 50)  # 短期サイクル
    cycle2 = 5 * np.sin(2 * np.pi * np.arange(num_points) / 100) # 中期サイクル
    cycle3 = 3 * np.sin(2 * np.pi * np.arange(num_points) / 200) # 長期サイクル
    
    # ボラティリティクラスター
    volatility = 1 + 0.5 * np.sin(2 * np.pi * np.arange(num_points) / 150)
    noise = np.random.randn(num_points) * volatility
    
    # 価格シリーズの合成
    close_prices = trend + random_walk + cycle1 + cycle2 + cycle3 + noise
    
    # OHLC生成
    high_offset = np.abs(np.random.randn(num_points)) * volatility
    low_offset = np.abs(np.random.randn(num_points)) * volatility
    open_offset = np.random.randn(num_points) * 0.5
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + open_offset,
        'high': close_prices + high_offset,
        'low': close_prices - low_offset,
        'close': close_prices,
        'volume': np.random.lognormal(8, 1, num_points)
    })
    
    # 価格の整合性を保つ
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])
    
    return data


# ===== Numba Smoother Functions =====

@njit(fastmath=True, cache=True)
def calculate_ultimate_smoother_numba(price: np.ndarray, period: float = 20.0) -> np.ndarray:
    """アルティメットスムーザー計算"""
    length = len(price)
    
    # 係数の計算
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1.0 + c2 - c3) / 4.0
    
    ultimate_smoother = np.zeros(length, dtype=np.float64)
    
    # 初期値
    for i in range(min(3, length)):
        ultimate_smoother[i] = price[i]
    
    # 計算
    for i in range(3, length):
        if i >= 2:
            ultimate_smoother[i] = ((1.0 - c1) * price[i] + 
                                   (2.0 * c1 - c2) * price[i-1] - 
                                   (c1 + c3) * price[i-2] + 
                                   c2 * ultimate_smoother[i-1] + 
                                   c3 * ultimate_smoother[i-2])
        else:
            ultimate_smoother[i] = price[i]
    
    return ultimate_smoother


@njit(fastmath=True, cache=True)
def calculate_super_smoother_2pole_numba(source: np.ndarray, length: int) -> np.ndarray:
    """2極スーパースムーザー計算"""
    data_length = len(source)
    result = np.zeros(data_length)
    
    if data_length < 3 or length < 2:
        return result
    
    # 係数計算
    PI = 2 * math.asin(1)
    arg = math.sqrt(2) * PI / length
    a1 = math.exp(-arg)
    b1 = 2 * a1 * math.cos(arg)
    
    coef3 = -math.pow(a1, 2)
    coef2 = b1
    coef1 = 1 - coef2 - coef3
    
    # 初期値
    result[0] = source[0]
    if data_length > 1:
        result[1] = source[1]
    
    # 計算
    for i in range(2, data_length):
        if not np.isnan(source[i]):
            result[i] = (coef1 * source[i] + 
                        coef2 * result[i-1] + 
                        coef3 * result[i-2])
        else:
            result[i] = result[i-1]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_super_smoother_3pole_numba(source: np.ndarray, length: int) -> np.ndarray:
    """3極スーパースムーザー計算"""
    data_length = len(source)
    result = np.zeros(data_length)
    
    if data_length < 4 or length < 2:
        return result
    
    # 係数計算
    PI = 2 * math.asin(1)
    arg = PI / length
    a1 = math.exp(-arg)
    b1 = 2 * a1 * math.cos(1.738 * arg)
    c1 = math.pow(a1, 2)
    
    coef4 = math.pow(c1, 2)
    coef3 = -(c1 + b1 * c1)
    coef2 = b1 + c1
    coef1 = 1 - coef2 - coef3 - coef4
    
    # 初期値
    result[0] = source[0]
    if data_length > 1:
        result[1] = source[1]
    if data_length > 2:
        result[2] = source[2]
    
    # 計算
    for i in range(3, data_length):
        if not np.isnan(source[i]):
            result[i] = (coef1 * source[i] + 
                        coef2 * result[i-1] + 
                        coef3 * result[i-2] + 
                        coef4 * result[i-3])
        else:
            result[i] = result[i-1]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_frama_simple_numba(price: np.ndarray, high: np.ndarray, low: np.ndarray, 
                                n: int, fc: int, sc: int) -> np.ndarray:
    """FRAMA簡易版計算"""
    length = len(price)
    frama = np.zeros(length, dtype=np.float64)
    
    # 初期値
    for i in range(length):
        frama[i] = price[i] if i < n else np.nan
    
    # w = log(2/(SC+1))
    w = np.log(2.0 / (sc + 1))
    
    # 計算
    for i in range(n, length):
        if np.isnan(price[i]):
            frama[i] = frama[i-1] if i > 0 else np.nan
            continue
        
        len1 = n // 2
        
        # H1, L1
        h1 = -np.inf
        l1 = np.inf
        for j in range(len1):
            if i - j >= 0:
                if high[i - j] > h1:
                    h1 = high[i - j]
                if low[i - j] < l1:
                    l1 = low[i - j]
        
        n1 = (h1 - l1) / len1
        
        # H2, L2
        h2 = -np.inf
        l2 = np.inf
        for j in range(len1, n):
            if i - j >= 0:
                if high[i - j] > h2:
                    h2 = high[i - j]
                if low[i - j] < l2:
                    l2 = low[i - j]
        
        n2 = (h2 - l2) / len1
        
        # H3, L3
        h3 = -np.inf
        l3 = np.inf
        for j in range(n):
            if i - j >= 0:
                if high[i - j] > h3:
                    h3 = high[i - j]
                if low[i - j] < l3:
                    l3 = low[i - j]
        
        n3 = (h3 - l3) / n
        
        # フラクタル次元計算
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            dimen = 1.0
        
        # アルファ計算
        alpha1 = np.exp(w * (dimen - 1.0))
        
        if alpha1 > 1.0:
            oldalpha = 1.0
        elif alpha1 < 0.01:
            oldalpha = 0.01
        else:
            oldalpha = alpha1
        
        oldN = (2.0 - oldalpha) / oldalpha
        N = (((sc - fc) * (oldN - 1.0)) / (sc - 1.0)) + fc
        alpha_ = 2.0 / (N + 1.0)
        
        min_alpha = 2.0 / (sc + 1.0)
        if alpha_ < min_alpha:
            final_alpha = min_alpha
        elif alpha_ > 1.0:
            final_alpha = 1.0
        else:
            final_alpha = alpha_
        
        # FRAMA計算
        if i == n:
            frama[i] = price[i]
        else:
            frama[i] = (1.0 - final_alpha) * frama[i-1] + final_alpha * price[i]
    
    return frama


@njit(fastmath=True, cache=True)
def calculate_adaptive_kalman_numba(signal: np.ndarray, process_noise: float = 1e-5) -> np.ndarray:
    """適応カルマンフィルター計算"""
    length = len(signal)
    filtered_signal = np.zeros(length)
    
    if length > 0:
        state = signal[0]
        error_cov = 1.0
        filtered_signal[0] = state
    
    for i in range(1, length):
        # 予測
        predicted_state = state
        predicted_covariance = error_cov + process_noise
        
        # 適応ノイズ推定
        if i > 5:
            recent_variance = 0.0
            for j in range(min(5, i)):
                if i - j >= 0:
                    diff = signal[i-j] - filtered_signal[i-j]
                    recent_variance += diff * diff
            observation_noise = recent_variance / 5.0 + 1e-6
        else:
            observation_noise = 1e-3
        
        # カルマンゲイン
        kalman_gain = predicted_covariance / (predicted_covariance + observation_noise)
        
        # 更新
        innovation = signal[i] - predicted_state
        state = predicted_state + kalman_gain * innovation
        error_cov = (1 - kalman_gain) * predicted_covariance
        
        filtered_signal[i] = state
    
    return filtered_signal


@njit(fastmath=True, cache=True)
def calculate_simple_ukf_numba(prices: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """UKF簡易版計算"""
    n = len(prices)
    filtered_prices = np.zeros(n)
    
    if n < 5:
        return prices.copy()
    
    # 初期状態 [価格, 速度, 加速度]
    x = np.array([prices[0], 0.0, 0.0])
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.01]])
    Q = np.array([[0.001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.00001]])
    
    filtered_prices[0] = prices[0]
    
    for t in range(1, n):
        # 簡易予測
        x[0] = x[0] + x[1] + 0.5 * x[2]
        x[1] = x[1] * 0.95 + x[2]
        x[2] = x[2] * 0.9
        
        # 共分散更新
        for i in range(3):
            for j in range(3):
                P[i, j] += Q[i, j]
        
        # 観測ノイズ
        if t >= 10:
            window_var = 0.0
            for k in range(10):
                if t - k >= 0:
                    diff = prices[t-k] - filtered_prices[t-k]
                    window_var += diff * diff
            R = window_var / 10.0 + 0.0001
        else:
            R = 0.01
        
        # カルマンゲイン（簡易版）
        K0 = P[0, 0] / (P[0, 0] + R)
        
        # 状態更新
        innovation = prices[t] - x[0]
        x[0] = x[0] + K0 * innovation
        
        # 共分散更新
        P[0, 0] = (1 - K0) * P[0, 0]
        
        filtered_prices[t] = x[0]
    
    return filtered_prices


def calculate_all_smoothers_simple(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """すべてのスムーザーを計算（簡易版）"""
    results = {}
    
    # 価格データの準備
    close_prices = data['close'].values.astype(np.float64)
    high_prices = data['high'].values.astype(np.float64)
    low_prices = data['low'].values.astype(np.float64)
    
    print("📊 スムーザー計算開始...")
    
    try:
        # Ultimate Smoother
        print("🔄 Ultimate Smoother 計算中...")
        results['UltimateSmoother'] = calculate_ultimate_smoother_numba(close_prices, 20.0)
        print("✅ Ultimate Smoother 完了")
        
    except Exception as e:
        print(f"❌ Ultimate Smoother 失敗: {e}")
        results['UltimateSmoother'] = close_prices.copy()
    
    try:
        # Super Smoother 2-pole
        print("🔄 Super Smoother (2極) 計算中...")
        results['SuperSmoother_2pole'] = calculate_super_smoother_2pole_numba(close_prices, 15)
        print("✅ Super Smoother (2極) 完了")
        
    except Exception as e:
        print(f"❌ Super Smoother (2極) 失敗: {e}")
        results['SuperSmoother_2pole'] = close_prices.copy()
    
    try:
        # Super Smoother 3-pole
        print("🔄 Super Smoother (3極) 計算中...")
        results['SuperSmoother_3pole'] = calculate_super_smoother_3pole_numba(close_prices, 15)
        print("✅ Super Smoother (3極) 完了")
        
    except Exception as e:
        print(f"❌ Super Smoother (3極) 失敗: {e}")
        results['SuperSmoother_3pole'] = close_prices.copy()
    
    try:
        # FRAMA
        print("🔄 FRAMA 計算中...")
        results['FRAMA'] = calculate_frama_simple_numba(close_prices, high_prices, low_prices, 16, 1, 198)
        print("✅ FRAMA 完了")
        
    except Exception as e:
        print(f"❌ FRAMA 失敗: {e}")
        results['FRAMA'] = close_prices.copy()
    
    try:
        # Adaptive Kalman
        print("🔄 Adaptive Kalman 計算中...")
        results['AdaptiveKalman'] = calculate_adaptive_kalman_numba(close_prices, 1e-5)
        print("✅ Adaptive Kalman 完了")
        
    except Exception as e:
        print(f"❌ Adaptive Kalman 失敗: {e}")
        results['AdaptiveKalman'] = close_prices.copy()
    
    try:
        # Simple UKF
        print("🔄 UKF (簡易版) 計算中...")
        results['UKF_Simple'] = calculate_simple_ukf_numba(close_prices, 0.001)
        print("✅ UKF (簡易版) 完了")
        
    except Exception as e:
        print(f"❌ UKF (簡易版) 失敗: {e}")
        results['UKF_Simple'] = close_prices.copy()
    
    # 元の価格
    results['Original'] = close_prices
    
    return results


def create_comparison_chart(data: pd.DataFrame, smoother_results: Dict[str, np.ndarray], config: Dict):
    """比較チャートを作成"""
    print("\n🎨 チャート作成中...")
    
    symbol = config.get('binance_data', {}).get('symbol', 'SOL')
    timeframe = config.get('binance_data', {}).get('timeframe', '4h')
    
    # 日付の準備
    if 'timestamp' in data.columns:
        dates = pd.to_datetime(data['timestamp'])
    else:
        dates = data.index
    
    # カラーパレット
    colors = {
        'Original': '#333333',
        'UltimateSmoother': '#FF6B6B',
        'SuperSmoother_2pole': '#4ECDC4', 
        'SuperSmoother_3pole': '#45B7D1',
        'FRAMA': '#96CEB4',
        'UKF_Simple': '#FFEAA7',
        'AdaptiveKalman': '#DDA0DD'
    }
    
    # フィギュア設定
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, height_ratios=[3, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # データ範囲の制限
    display_range = min(500, len(data))
    start_idx = max(0, len(data) - display_range)
    dates_display = dates[start_idx:]
    
    # メインチャート
    ax_main = fig.add_subplot(gs[0, :])
    
    # 価格チャート
    for name, values in smoother_results.items():
        if len(values) > 0:
            values_display = values[start_idx:]
            line_width = 2.5 if name == 'Original' else 1.5
            alpha = 0.9 if name == 'Original' else 0.7
            
            ax_main.plot(dates_display, values_display, 
                        label=name, color=colors.get(name, '#888888'),
                        linewidth=line_width, alpha=alpha)
    
    ax_main.set_title(f'全スムーザー比較 - {symbol} {timeframe}', fontsize=16, fontweight='bold')
    ax_main.set_ylabel('価格', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 差分チャート
    ax_diff = fig.add_subplot(gs[1, :])
    original_values = smoother_results['Original'][start_idx:]
    
    for name, values in smoother_results.items():
        if name != 'Original' and len(values) > 0:
            values_display = values[start_idx:]
            diff = values_display - original_values
            ax_diff.plot(dates_display, diff, 
                        label=f'{name} - Original', 
                        color=colors.get(name, '#888888'),
                        linewidth=1.2, alpha=0.7)
    
    ax_diff.set_title('元価格との差分', fontsize=12, fontweight='bold')
    ax_diff.set_ylabel('差分', fontsize=10)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_diff.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 移動平均乖離率チャート
    ax_dev = fig.add_subplot(gs[2, :])
    ma_period = 20
    original_ma = np.convolve(original_values, np.ones(ma_period)/ma_period, mode='same')
    
    for name, values in smoother_results.items():
        if len(values) > 0:
            values_display = values[start_idx:]
            deviation = ((values_display - original_ma) / original_ma) * 100
            ax_dev.plot(dates_display, deviation, 
                       label=f'{name}', 
                       color=colors.get(name, '#888888'),
                       linewidth=1.2, alpha=0.7)
    
    ax_dev.set_title(f'移動平均(MA{ma_period})からの乖離率 (%)', fontsize=12, fontweight='bold')
    ax_dev.set_ylabel('乖離率 (%)', fontsize=10)
    ax_dev.grid(True, alpha=0.3)
    ax_dev.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_dev.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # ボラティリティチャート
    ax_vol = fig.add_subplot(gs[3, :])
    
    for name, values in smoother_results.items():
        if len(values) > 0:
            values_display = values[start_idx:]
            # ローリング標準偏差でボラティリティを計算
            window = 20
            volatility = pd.Series(values_display).rolling(window=window, min_periods=1).std()
            ax_vol.plot(dates_display, volatility, 
                       label=f'{name}', 
                       color=colors.get(name, '#888888'),
                       linewidth=1.2, alpha=0.7)
    
    ax_vol.set_title(f'ローリングボラティリティ (窓{window})', fontsize=12, fontweight='bold')
    ax_vol.set_ylabel('ボラティリティ', fontsize=10)
    ax_vol.set_xlabel('日時', fontsize=10)
    ax_vol.grid(True, alpha=0.3)
    ax_vol.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 日付フォーマット設定
    for ax in [ax_main, ax_diff, ax_dev, ax_vol]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # タイトル
    fig.suptitle(f'全スムーザーインジケーター比較分析\n{symbol} {timeframe} - 最新 {display_range} データポイント', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ファイル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_smoothers_comparison_{symbol}_{timeframe}_{timestamp}.png"
    filepath = project_root / filename
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📁 チャート保存完了: {filepath}")
        
    except Exception as e:
        print(f"⚠️ チャート保存失敗: {e}")
    
    plt.show()


def calculate_smoother_statistics(smoother_results: Dict[str, np.ndarray]) -> Dict:
    """統計計算"""
    stats = {}
    original = smoother_results.get('Original', np.array([]))
    
    if len(original) == 0:
        return stats
    
    for name, values in smoother_results.items():
        if name == 'Original' or len(values) == 0:
            continue
            
        # 有効データのみ
        valid_mask = ~np.isnan(values) & ~np.isnan(original)
        if not np.any(valid_mask):
            continue
            
        valid_values = values[valid_mask]
        valid_original = original[valid_mask]
        
        # 統計計算
        mae = np.mean(np.abs(valid_values - valid_original))
        rmse = np.sqrt(np.mean((valid_values - valid_original) ** 2))
        correlation = np.corrcoef(valid_values, valid_original)[0, 1]
        smoothness = np.std(np.diff(valid_values))
        
        stats[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'Smoothness': smoothness,
            'Data_Points': len(valid_values)
        }
    
    return stats


def print_statistics(stats: Dict):
    """統計情報を表示"""
    print("\n📊 スムーザー性能統計:")
    print("=" * 80)
    print(f"{'スムーザー名':<20} {'MAE':<8} {'RMSE':<8} {'相関':<6} {'平滑度':<8} {'データ点':<8}")
    print("-" * 80)
    
    for name, stat in stats.items():
        print(f"{name:<20} {stat['MAE']:<8.3f} {stat['RMSE']:<8.3f} "
              f"{stat['Correlation']:<6.3f} {stat['Smoothness']:<8.3f} {stat['Data_Points']:<8}")
    
    print("=" * 80)
    print("MAE: 平均絶対誤差（小さいほど良い）")
    print("RMSE: 二乗平均平方根誤差（小さいほど良い）") 
    print("相関: 元価格との相関係数（1に近いほど良い）")
    print("平滑度: 平滑さの指標（小さいほど滑らか）")


def main():
    """メイン処理"""
    print("🚀 全スムーザー比較チャート作成開始\n")
    
    try:
        # 設定読み込み
        config = load_config()
        print("✅ 設定ファイル読み込み完了")
        
        # サンプルデータ生成
        print("📊 サンプル市場データ生成中...")
        data = generate_sample_data(1000)
        print(f"✅ データ生成完了: {len(data)}件")
        
        # スムーザー計算
        smoother_results = calculate_all_smoothers_simple(data)
        
        # 統計計算・表示
        stats = calculate_smoother_statistics(smoother_results)
        print_statistics(stats)
        
        # チャート作成
        create_comparison_chart(data, smoother_results, config)
        
        print("\n🎉 全スムーザー比較チャート作成完了!")
        
    except KeyboardInterrupt:
        print("\n⏹️  ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()