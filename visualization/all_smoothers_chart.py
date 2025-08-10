#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
すべてのスムーザーインジケーター比較チャート

indicators/smoother/ ディレクトリ内のすべてのスムーザーを比較するmatplotlibチャート:
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
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yaml

# プロジェクトルートの追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# データローダーの追加
try:
    from data_loader import DataLoader
except ImportError:
    print("Warning: DataLoader not found. Using fallback data generation.")
    DataLoader = None

# スムーザーインジケーターのインポート
try:
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    from indicators.smoother.super_smoother import SuperSmoother  
    from indicators.smoother.frama import FRAMA
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.smoother.adaptive_kalman import AdaptiveKalman
    
    print("✅ すべてのスムーザーインジケーターの読み込み完了")
    
except ImportError as e:
    print(f"❌ スムーザーインジケーターの読み込みに失敗: {e}")
    print("Fallback: Numba関数を直接使用します")
    
    # Numba関数の直接インポート
    from indicators.smoother.ultimate_smoother import calculate_ultimate_smoother
    from indicators.smoother.super_smoother import calculate_super_smoother_numba
    from indicators.smoother.frama import calculate_frama_core
    from indicators.smoother.unscented_kalman_filter import calculate_unscented_kalman_filter, estimate_volatility
    from indicators.smoother.adaptive_kalman import adaptive_kalman_filter


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


def load_market_data(config: Dict) -> pd.DataFrame:
    """市場データを読み込む"""
    if DataLoader is not None:
        try:
            # 設定からパラメータを取得
            binance_config = config.get('binance_data', {})
            symbol = binance_config.get('symbol', 'SOL')
            timeframe = binance_config.get('timeframe', '4h')
            
            loader = DataLoader(config)
            data = loader.load_binance_data()
            
            if data is not None and len(data) > 0:
                print(f"✅ 市場データ読み込み完了: {symbol} {timeframe} ({len(data)}件)")
                return data
            else:
                print("⚠️ 市場データが空です。サンプルデータを生成します。")
                
        except Exception as e:
            print(f"⚠️ DataLoader使用中にエラー: {e}")
    
    # フォールバック: サンプルデータ生成
    print("📊 サンプル市場データを生成中...")
    return generate_sample_data()


def generate_sample_data(num_points: int = 1000) -> pd.DataFrame:
    """リアルな市場データ風のサンプルデータを生成"""
    np.random.seed(42)
    
    # 日付範囲を生成
    dates = pd.date_range(start='2023-01-01', periods=num_points, freq='4H')
    
    # 基本価格トレンド
    trend = np.linspace(100, 180, num_points)
    
    # ランダムウォーク
    random_walk = np.cumsum(np.random.randn(num_points) * 0.5)
    
    # 周期的変動
    cycle1 = 5 * np.sin(2 * np.pi * np.arange(num_points) / 50)
    cycle2 = 3 * np.sin(2 * np.pi * np.arange(num_points) / 100)
    
    # ノイズ
    noise = np.random.randn(num_points) * 1.5
    
    # 価格シリーズの合成
    close_prices = trend + random_walk + cycle1 + cycle2 + noise
    
    # OHLC生成
    high_offset = np.abs(np.random.randn(num_points)) * 2
    low_offset = np.abs(np.random.randn(num_points)) * 2
    open_offset = np.random.randn(num_points) * 1
    
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


def create_smoother_instances() -> Dict:
    """すべてのスムーザーインスタンスを作成"""
    smoothers = {}
    
    try:
        # UltimateSmoother (固定期間モード)
        smoothers['UltimateSmoother'] = UltimateSmoother(
            period=20.0,
            src_type='close',
            period_mode='fixed'  # 動的モードは依存関係が複雑なため固定モードを使用
        )
        print("✅ UltimateSmoother作成完了")
        
    except Exception as e:
        print(f"⚠️ UltimateSmoother作成失敗: {e}")
        smoothers['UltimateSmoother'] = None
    
    try:
        # SuperSmoother (2極)
        smoothers['SuperSmoother_2pole'] = SuperSmoother(
            length=15,
            num_poles=2,
            src_type='close'
        )
        print("✅ SuperSmoother(2極)作成完了")
        
    except Exception as e:
        print(f"⚠️ SuperSmoother(2極)作成失敗: {e}")
        smoothers['SuperSmoother_2pole'] = None
        
    try:
        # SuperSmoother (3極)
        smoothers['SuperSmoother_3pole'] = SuperSmoother(
            length=15,
            num_poles=3,
            src_type='close'
        )
        print("✅ SuperSmoother(3極)作成完了")
        
    except Exception as e:
        print(f"⚠️ SuperSmoother(3極)作成失敗: {e}")
        smoothers['SuperSmoother_3pole'] = None
    
    try:
        # FRAMA
        smoothers['FRAMA'] = FRAMA(
            period=16,  # 偶数である必要がある
            src_type='hl2',
            fc=1,
            sc=198
        )
        print("✅ FRAMA作成完了")
        
    except Exception as e:
        print(f"⚠️ FRAMA作成失敗: {e}")
        smoothers['FRAMA'] = None
    
    try:
        # UnscentedKalmanFilter
        smoothers['UKF'] = UnscentedKalmanFilter(
            src_type='close',
            alpha=0.001,
            beta=2.0,
            kappa=0.0,
            process_noise_scale=0.001,
            volatility_window=10,
            adaptive_noise=True
        )
        print("✅ UKF作成完了")
        
    except Exception as e:
        print(f"⚠️ UKF作成失敗: {e}")
        smoothers['UKF'] = None
    
    try:
        # AdaptiveKalman
        smoothers['AdaptiveKalman'] = AdaptiveKalman(
            process_noise=1e-5,
            src_type='close',
            min_observation_noise=1e-6,
            adaptation_window=5
        )
        print("✅ AdaptiveKalman作成完了")
        
    except Exception as e:
        print(f"⚠️ AdaptiveKalman作成失敗: {e}")
        smoothers['AdaptiveKalman'] = None
    
    return smoothers


def calculate_all_smoothers(data: pd.DataFrame, smoothers: Dict) -> Dict[str, np.ndarray]:
    """すべてのスムーザーを計算"""
    results = {}
    prices = data['close'].values
    
    print("\n📊 スムーザー計算開始...")
    
    for name, smoother in smoothers.items():
        if smoother is None:
            print(f"⏭️  {name}: スキップ（インスタンス作成失敗）")
            continue
            
        try:
            print(f"🔄 {name} 計算中...")
            
            if name == 'UltimateSmoother':
                result = smoother.calculate(data)
                results[name] = result.values
                
            elif name.startswith('SuperSmoother'):
                result = smoother.calculate(data)
                results[name] = result.values
                
            elif name == 'FRAMA':
                result = smoother.calculate(data)
                results[name] = result.values
                
            elif name == 'UKF':
                result = smoother.calculate(data)
                results[name] = result.filtered_values
                
            elif name == 'AdaptiveKalman':
                result = smoother.calculate(data)
                results[name] = result.filtered_signal
                
            print(f"✅ {name} 計算完了")
            
        except Exception as e:
            print(f"❌ {name} 計算失敗: {e}")
            # フォールバック: 元の価格を使用
            results[name] = prices.copy()
    
    # 元の価格も追加
    results['Original'] = prices
    
    return results


def create_comparison_chart(data: pd.DataFrame, smoother_results: Dict[str, np.ndarray], config: Dict):
    """比較チャートを作成"""
    print("\n🎨 チャート作成中...")
    
    # 設定
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
        'UKF': '#FFEAA7',
        'AdaptiveKalman': '#DDA0DD'
    }
    
    # フィギュアとサブプロット設定
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[3, 1, 1], hspace=0.3, wspace=0.3)
    
    # メインチャート
    ax_main = fig.add_subplot(gs[0, :])
    
    # データ範囲の制限（最新1000点）
    display_range = min(1000, len(data))
    start_idx = max(0, len(data) - display_range)
    
    dates_display = dates[start_idx:]
    
    # 価格チャート
    for name, values in smoother_results.items():
        if len(values) > 0:
            values_display = values[start_idx:]
            line_width = 2.5 if name == 'Original' else 1.5
            alpha = 0.8 if name == 'Original' else 0.7
            
            ax_main.plot(dates_display, values_display, 
                        label=name, color=colors.get(name, '#888888'),
                        linewidth=line_width, alpha=alpha)
    
    ax_main.set_title(f'全スムーザー比較 - {symbol} {timeframe}', fontsize=16, fontweight='bold')
    ax_main.set_ylabel('価格', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # X軸の日付フォーマット
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
    
    # 統計サブプロット1: 元価格との差分
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
    
    # X軸の日付フォーマット
    ax_diff.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_diff.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_diff.xaxis.get_majorticklabels(), rotation=45)
    
    # 統計サブプロット2: 移動平均からの乖離
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
    ax_dev.set_xlabel('日時', fontsize=10)
    ax_dev.grid(True, alpha=0.3)
    ax_dev.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_dev.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # X軸の日付フォーマット
    ax_dev.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax_dev.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_dev.xaxis.get_majorticklabels(), rotation=45)
    
    # タイトルとサブタイトル
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
    """スムーザーの統計情報を計算"""
    stats = {}
    original = smoother_results.get('Original', np.array([]))
    
    if len(original) == 0:
        return stats
    
    for name, values in smoother_results.items():
        if name == 'Original' or len(values) == 0:
            continue
            
        # 有効なデータのみ使用
        valid_mask = ~np.isnan(values) & ~np.isnan(original)
        if not np.any(valid_mask):
            continue
            
        valid_values = values[valid_mask]
        valid_original = original[valid_mask]
        
        # 統計計算
        mae = np.mean(np.abs(valid_values - valid_original))
        rmse = np.sqrt(np.mean((valid_values - valid_original) ** 2))
        
        # 相関係数
        correlation = np.corrcoef(valid_values, valid_original)[0, 1]
        
        # ラグ計算（遅延度）
        lag = calculate_lag(valid_original, valid_values)
        
        # 平滑度（連続する値の差の標準偏差）
        smoothness = np.std(np.diff(valid_values))
        
        stats[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'Lag': lag,
            'Smoothness': smoothness,
            'Data_Points': len(valid_values)
        }
    
    return stats


def calculate_lag(original: np.ndarray, smoothed: np.ndarray, max_lag: int = 20) -> int:
    """ラグ（遅延）を計算"""
    if len(original) < max_lag * 2:
        return 0
        
    best_correlation = -1
    best_lag = 0
    
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(original, smoothed)[0, 1]
        else:
            corr = np.corrcoef(original[:-lag], smoothed[lag:])[0, 1]
        
        if corr > best_correlation:
            best_correlation = corr
            best_lag = lag
    
    return best_lag


def print_statistics(stats: Dict):
    """統計情報を印刷"""
    print("\n📊 スムーザー性能統計:")
    print("=" * 80)
    print(f"{'スムーザー名':<20} {'MAE':<8} {'RMSE':<8} {'相関':<6} {'遅延':<6} {'平滑度':<8} {'データ点':<8}")
    print("-" * 80)
    
    for name, stat in stats.items():
        print(f"{name:<20} {stat['MAE']:<8.3f} {stat['RMSE']:<8.3f} "
              f"{stat['Correlation']:<6.3f} {stat['Lag']:<6} "
              f"{stat['Smoothness']:<8.3f} {stat['Data_Points']:<8}")
    
    print("=" * 80)
    print("MAE: 平均絶対誤差（小さいほど良い）")
    print("RMSE: 二乗平均平方根誤差（小さいほど良い）")
    print("相関: 元価格との相関係数（1に近いほど良い）")
    print("遅延: 遅延サンプル数（小さいほど良い）")
    print("平滑度: 平滑さの指標（小さいほど滑らか）")


def main():
    """メイン処理"""
    print("🚀 全スムーザー比較チャート作成開始\n")
    
    try:
        # 設定読み込み
        config = load_config()
        print("✅ 設定ファイル読み込み完了")
        
        # 市場データ読み込み
        data = load_market_data(config)
        
        if data is None or len(data) == 0:
            print("❌ データの読み込みに失敗しました")
            return
        
        # データの最小要件確認
        if len(data) < 100:
            print("⚠️ データが少なすぎます。最低100件必要です。")
            return
        
        # スムーザーインスタンス作成
        smoothers = create_smoother_instances()
        
        # 有効なスムーザーの確認
        valid_smoothers = {k: v for k, v in smoothers.items() if v is not None}
        if not valid_smoothers:
            print("❌ 有効なスムーザーがありません")
            return
        
        print(f"📈 有効なスムーザー数: {len(valid_smoothers)}")
        
        # 全スムーザー計算
        smoother_results = calculate_all_smoothers(data, valid_smoothers)
        
        # 統計計算
        stats = calculate_smoother_statistics(smoother_results)
        print_statistics(stats)
        
        # チャート作成
        create_comparison_chart(data, smoother_results, config)
        
        print("\n🎉 全スムーザー比較チャート作成完了!")
        
    except KeyboardInterrupt:
        print("\n⏹️  ユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print("詳細エラー:")
        traceback.print_exc()


if __name__ == "__main__":
    main()