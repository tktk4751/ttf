#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
フルスムーザー比較チャート

@indicators/smoother/ ディレクトリ内のすべてのスムーザーをインポートして使用:
- UltimateSmoother: ジョン・エーラーズのアルティメットスムーザー
- SuperSmoother: 2極と3極のスーパースムーザー
- FRAMA: フラクタル適応移動平均
- UnscentedKalmanFilter: 無香料カルマンフィルター  
- AdaptiveKalman: 適応カルマンフィルター

実際のインディケータークラスインスタンスを使用することで、
簡易実装よりも正確な性能メトリクスと適切なチャート表示を実現。
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

# プロジェクトルートの追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# スムーザーインディケーターをインポート
try:
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    from indicators.smoother.super_smoother import SuperSmoother
    from indicators.smoother.frama import FRAMA
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.smoother.adaptive_kalman import AdaptiveKalman
    print("✅ すべてのスムーザーインディケーターのインポート完了")
except ImportError as e:
    print(f"❌ スムーザーインディケーターのインポートエラー: {e}")
    print("フォールバック: 利用可能なモジュールのみ使用")
    UltimateSmoother = None
    SuperSmoother = None
    FRAMA = None
    UnscentedKalmanFilter = None
    AdaptiveKalman = None


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


def calculate_all_smoothers_full(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """すべてのスムーザーを計算（実際のインディケーター使用）"""
    results = {}
    
    # 元の価格データ
    results['Original'] = data['close'].values
    
    print("📊 フルスムーザー計算開始...")
    
    # Ultimate Smoother
    if UltimateSmoother is not None:
        try:
            print("🔄 Ultimate Smoother 計算中...")
            ultimate_smoother = UltimateSmoother(period=20.0, src_type='close')
            us_result = ultimate_smoother.calculate(data)
            results['UltimateSmoother'] = us_result.values
            print("✅ Ultimate Smoother 完了")
        except Exception as e:
            print(f"❌ Ultimate Smoother 失敗: {e}")
            results['UltimateSmoother'] = data['close'].values.copy()
    else:
        print("⏭️ Ultimate Smoother をスキップ（未インポート）")
    
    # Super Smoother (2-pole)
    if SuperSmoother is not None:
        try:
            print("🔄 Super Smoother (2極) 計算中...")
            super_smoother_2 = SuperSmoother(length=15, num_poles=2, src_type='close')
            ss2_result = super_smoother_2.calculate(data)
            results['SuperSmoother_2pole'] = ss2_result.values
            print("✅ Super Smoother (2極) 完了")
        except Exception as e:
            print(f"❌ Super Smoother (2極) 失敗: {e}")
            results['SuperSmoother_2pole'] = data['close'].values.copy()
    else:
        print("⏭️ Super Smoother をスキップ（未インポート）")
    
    # Super Smoother (3-pole)
    if SuperSmoother is not None:
        try:
            print("🔄 Super Smoother (3極) 計算中...")
            super_smoother_3 = SuperSmoother(length=15, num_poles=3, src_type='close')
            ss3_result = super_smoother_3.calculate(data)
            results['SuperSmoother_3pole'] = ss3_result.values
            print("✅ Super Smoother (3極) 完了")
        except Exception as e:
            print(f"❌ Super Smoother (3極) 失敗: {e}")
            results['SuperSmoother_3pole'] = data['close'].values.copy()
    else:
        print("⏭️ Super Smoother (3極) をスキップ（未インポート）")
    
    # FRAMA
    if FRAMA is not None:
        try:
            print("🔄 FRAMA 計算中...")
            frama = FRAMA(period=16, fc=1, sc=198, src_type='close')
            frama_result = frama.calculate(data)
            results['FRAMA'] = frama_result.values
            print("✅ FRAMA 完了")
        except Exception as e:
            print(f"❌ FRAMA 失敗: {e}")
            results['FRAMA'] = data['close'].values.copy()
    else:
        print("⏭️ FRAMA をスキップ（未インポート）")
    
    # Adaptive Kalman
    if AdaptiveKalman is not None:
        try:
            print("🔄 Adaptive Kalman 計算中...")
            adaptive_kalman = AdaptiveKalman(process_noise=1e-5, src_type='close')
            ak_result = adaptive_kalman.calculate(data)
            results['AdaptiveKalman'] = ak_result.filtered_signal
            print("✅ Adaptive Kalman 完了")
        except Exception as e:
            print(f"❌ Adaptive Kalman 失敗: {e}")
            results['AdaptiveKalman'] = data['close'].values.copy()
    else:
        print("⏭️ Adaptive Kalman をスキップ（未インポート）")
    
    # Unscented Kalman Filter
    if UnscentedKalmanFilter is not None:
        try:
            print("🔄 Unscented Kalman Filter 計算中...")
            ukf = UnscentedKalmanFilter(
                alpha=0.1,  # 修正されたパラメータ 
                beta=2.0,
                kappa=0.0,
                process_noise_scale=0.01,  # 修正されたパラメータ
                src_type='close'
            )
            ukf_result = ukf.calculate(data)
            results['UKF'] = ukf_result.filtered_values
            print("✅ Unscented Kalman Filter 完了")
        except Exception as e:
            print(f"❌ Unscented Kalman Filter 失敗: {e}")
            traceback.print_exc()
            results['UKF'] = data['close'].values.copy()
    else:
        print("⏭️ UKF をスキップ（未インポート）")
    
    return results


def create_full_comparison_chart(data: pd.DataFrame, smoother_results: Dict[str, np.ndarray], config: Dict):
    """フル比較チャートを作成"""
    print("\n🎨 フルチャート作成中...")
    
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
    
    # フィギュア設定
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 2, height_ratios=[3, 1, 1, 1, 1], hspace=0.35, wspace=0.3)
    
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
            line_width = 2.5 if name == 'Original' else 1.8
            alpha = 0.9 if name == 'Original' else 0.75
            
            ax_main.plot(dates_display, values_display, 
                        label=name, color=colors.get(name, '#888888'),
                        linewidth=line_width, alpha=alpha)
    
    ax_main.set_title(f'全スムーザーインディケーター比較 (実装版) - {symbol} {timeframe}', 
                     fontsize=16, fontweight='bold')
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
                        linewidth=1.5, alpha=0.75)
    
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
                       linewidth=1.5, alpha=0.75)
    
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
                       linewidth=1.5, alpha=0.75)
    
    ax_vol.set_title(f'ローリングボラティリティ (窓{window})', fontsize=12, fontweight='bold')
    ax_vol.set_ylabel('ボラティリティ', fontsize=10)
    ax_vol.grid(True, alpha=0.3)
    ax_vol.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 相関係数チャート
    ax_corr = fig.add_subplot(gs[4, :])
    
    # ローリング相関を計算
    correlation_window = 50
    for name, values in smoother_results.items():
        if name != 'Original' and len(values) > 0:
            values_display = values[start_idx:]
            # ローリング相関を計算
            rolling_corr = []
            for i in range(len(values_display)):
                start = max(0, i - correlation_window + 1)
                end = i + 1
                if end - start >= 10:  # 最小サンプル数
                    corr = np.corrcoef(original_values[start:end], values_display[start:end])[0, 1]
                    rolling_corr.append(corr if not np.isnan(corr) else 0)
                else:
                    rolling_corr.append(0)
            
            ax_corr.plot(dates_display, rolling_corr, 
                        label=f'{name}', 
                        color=colors.get(name, '#888888'),
                        linewidth=1.5, alpha=0.75)
    
    ax_corr.set_title(f'ローリング相関係数 (窓{correlation_window})', fontsize=12, fontweight='bold')
    ax_corr.set_ylabel('相関係数', fontsize=10)
    ax_corr.set_xlabel('日時', fontsize=10)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_corr.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_corr.axhline(y=1, color='green', linestyle='--', alpha=0.3)
    ax_corr.set_ylim(-0.2, 1.1)
    
    # 日付フォーマット設定
    for ax in [ax_main, ax_diff, ax_dev, ax_vol, ax_corr]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # タイトル
    fig.suptitle(f'全スムーザーインディケーター比較分析 (実装版)\\n{symbol} {timeframe} - 最新 {display_range} データポイント', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ファイル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"full_smoothers_comparison_{symbol}_{timeframe}_{timestamp}.png"
    filepath = project_root / filename
    
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📁 フルチャート保存完了: {filepath}")
        
    except Exception as e:
        print(f"⚠️ フルチャート保存失敗: {e}")
    
    plt.show()


def calculate_full_smoother_statistics(smoother_results: Dict[str, np.ndarray]) -> Dict:
    """フルスムーザー統計計算"""
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
        
        if len(valid_values) < 10:  # 最小データ数チェック
            continue
        
        # 統計計算
        mae = np.mean(np.abs(valid_values - valid_original))
        rmse = np.sqrt(np.mean((valid_values - valid_original) ** 2))
        correlation = np.corrcoef(valid_values, valid_original)[0, 1]
        
        # スムーズネス（変化率の標準偏差）
        smoothness = np.std(np.diff(valid_values))
        
        # ラグ計算（最大相関でのラグ）
        lag = 0
        if len(valid_values) > 50:
            max_lag = min(20, len(valid_values) // 4)
            cross_corr = np.correlate(valid_values, valid_original, mode='full')
            lags = np.arange(-max_lag, max_lag + 1)
            if len(lags) <= len(cross_corr):
                mid = len(cross_corr) // 2
                start = mid - max_lag
                end = mid + max_lag + 1
                lag_corr = cross_corr[start:end]
                lag = lags[np.argmax(lag_corr)]
        
        # 信号対雑音比
        signal_power = np.var(valid_values)
        noise_power = np.var(valid_values - valid_original)
        snr = signal_power / (noise_power + 1e-10)
        
        stats[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': correlation,
            'Smoothness': smoothness,
            'Lag': lag,
            'SNR': snr,
            'Data_Points': len(valid_values)
        }
    
    return stats


def print_full_statistics(stats: Dict):
    """フル統計情報を表示"""
    print("\n📊 フルスムーザー性能統計:")
    print("=" * 100)
    print(f"{'スムーザー名':<20} {'MAE':<8} {'RMSE':<8} {'相関':<6} {'平滑度':<8} {'ラグ':<5} {'SNR':<6} {'データ点':<8}")
    print("-" * 100)
    
    for name, stat in stats.items():
        print(f"{name:<20} {stat['MAE']:<8.3f} {stat['RMSE']:<8.3f} "
              f"{stat['Correlation']:<6.3f} {stat['Smoothness']:<8.3f} "
              f"{stat['Lag']:<5} {stat['SNR']:<6.2f} {stat['Data_Points']:<8}")
    
    print("=" * 100)
    print("MAE: 平均絶対誤差（小さいほど良い）")
    print("RMSE: 二乗平均平方根誤差（小さいほど良い）") 
    print("相関: 元価格との相関係数（1に近いほど良い）")
    print("平滑度: 平滑さの指標（小さいほど滑らか）")
    print("ラグ: 遅延バー数（小さいほど良い）")
    print("SNR: 信号対雑音比（大きいほど良い）")


def main():
    """メイン処理"""
    print("🚀 フルスムーザー比較チャート作成開始\\n")
    
    try:
        # 設定読み込み
        config = load_config()
        print("✅ 設定ファイル読み込み完了")
        
        # サンプルデータ生成
        print("📊 サンプル市場データ生成中...")
        data = generate_sample_data(1000)
        print(f"✅ データ生成完了: {len(data)}件")
        
        # フルスムーザー計算
        smoother_results = calculate_all_smoothers_full(data)
        
        # 統計計算・表示
        stats = calculate_full_smoother_statistics(smoother_results)
        print_full_statistics(stats)
        
        # チャート作成
        create_full_comparison_chart(data, smoother_results, config)
        
        print("\\n🎉 フルスムーザー比較チャート作成完了!")
        
    except KeyboardInterrupt:
        print("\\n⏹️  ユーザーによって中断されました")
    except Exception as e:
        print(f"\\n❌ エラーが発生しました: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()