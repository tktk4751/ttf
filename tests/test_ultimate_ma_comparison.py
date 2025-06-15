#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UltimateMA V1 vs V2 vs V3 Performance Comparison Test
実際のBinanceデータでの性能比較検証システム
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# matplotlibのフォント警告を無効化
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# UltimateMAの各バージョンをインポート
from indicators.ultimate_ma import UltimateMA  # V1 (original)
from indicators.ultimate_ma_v3 import UltimateMAV3  # V3 (latest)

# V2は存在しないため、V1の改良版として設定
class UltimateMAV2(UltimateMA):
    """UltimateMA V2 (V1の改良版として定義)"""
    def __init__(self, **kwargs):
        # V2はV1の改良版として、より保守的なパラメータを使用
        super().__init__(
            super_smooth_period=kwargs.get('super_smooth_period', 12),
            zero_lag_period=kwargs.get('zero_lag_period', 24),
            realtime_window=kwargs.get('realtime_window', 55),
            src_type=kwargs.get('src_type', 'hlc3'),
            slope_index=kwargs.get('slope_index', 2),
            range_threshold=kwargs.get('range_threshold', 0.008)
        )


def load_binance_data(symbol='BTC', market_type='spot', timeframe='4h', data_dir='data/binance'):
    """
    Binanceデータを直接読み込む
    """
    file_path = f"{data_dir}/{symbol}/{market_type}/{timeframe}/historical_data.csv"
    
    print(f"📂 データファイル読み込み中: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ データファイルが見つかりません: {file_path}")
        return None
    
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        # タイムスタンプをインデックスに設定
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 必要なカラムが存在するか確認
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 必要なカラムが不足しています: {missing_columns}")
            return None
        
        # データ型を数値に変換
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # NaNを除去
        df = df.dropna()
        
        print(f"✅ データ読み込み成功: {symbol} {market_type} {timeframe}")
        print(f"📊 データ期間: {df.index.min()} - {df.index.max()}")
        print(f"📈 データ数: {len(df)}件")
        
        return df
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None


def calculate_performance_metrics(data: pd.DataFrame, ma_values: np.ndarray, trend_signals: np.ndarray, version_name: str) -> dict:
    """
    各バージョンの性能指標を計算
    
    Args:
        data: 元のOHLCデータ
        ma_values: 移動平均値
        trend_signals: トレンド信号
        version_name: バージョン名
    
    Returns:
        dict: 性能指標
    """
    close_prices = data['close'].values
    
    # 1. ノイズ除去効果
    raw_volatility = np.nanstd(close_prices)
    filtered_volatility = np.nanstd(ma_values)
    noise_reduction = (raw_volatility - filtered_volatility) / raw_volatility if raw_volatility > 0 else 0
    
    # 2. トレンド検出精度
    price_changes = np.diff(close_prices)
    ma_changes = np.diff(ma_values)
    
    # 方向一致度（価格変化とMA変化の方向が一致する割合）
    direction_accuracy = np.mean(np.sign(price_changes) == np.sign(ma_changes)) if len(price_changes) > 0 else 0
    
    # 3. 遅延分析
    # 価格の転換点とMAの転換点の遅延を計算
    price_turning_points = []
    ma_turning_points = []
    
    for i in range(2, len(close_prices) - 2):
        # 価格の転換点検出
        if ((close_prices[i-1] < close_prices[i] > close_prices[i+1]) or 
            (close_prices[i-1] > close_prices[i] < close_prices[i+1])):
            price_turning_points.append(i)
    
    for i in range(2, len(ma_values) - 2):
        # MAの転換点検出
        if ((ma_values[i-1] < ma_values[i] > ma_values[i+1]) or 
            (ma_values[i-1] > ma_values[i] < ma_values[i+1])):
            ma_turning_points.append(i)
    
    # 平均遅延計算
    if price_turning_points and ma_turning_points:
        delays = []
        for pt in price_turning_points:
            closest_ma_tp = min(ma_turning_points, key=lambda x: abs(x - pt))
            if closest_ma_tp > pt:  # MAが価格より後の場合のみ
                delays.append(closest_ma_tp - pt)
        avg_delay = np.mean(delays) if delays else 0
    else:
        avg_delay = 0
    
    # 4. トレンド信号統計
    if trend_signals is not None and len(trend_signals) > 0:
        up_signals = np.sum(trend_signals == 1)
        down_signals = np.sum(trend_signals == -1)
        range_signals = np.sum(trend_signals == 0)
        total_signals = len(trend_signals)
        
        # トレンド継続性（同じ信号が連続する平均長）
        signal_changes = np.diff(trend_signals)
        trend_continuity = len(trend_signals) / (np.sum(signal_changes != 0) + 1) if len(signal_changes) > 0 else 1
    else:
        up_signals = down_signals = range_signals = total_signals = 0
        trend_continuity = 0
    
    # 5. 価格追従性
    # MAと価格の相関係数
    correlation = np.corrcoef(close_prices, ma_values)[0, 1] if len(close_prices) == len(ma_values) else 0
    
    # 6. 安定性指標
    # MAの変動係数（標準偏差/平均）
    ma_stability = np.nanstd(ma_values) / np.nanmean(np.abs(ma_values)) if np.nanmean(np.abs(ma_values)) > 0 else 0
    
    return {
        'version': version_name,
        'noise_reduction': {
            'raw_volatility': raw_volatility,
            'filtered_volatility': filtered_volatility,
            'reduction_ratio': noise_reduction,
            'reduction_percentage': noise_reduction * 100
        },
        'trend_detection': {
            'direction_accuracy': direction_accuracy,
            'avg_delay': avg_delay,
            'trend_continuity': trend_continuity
        },
        'signal_distribution': {
            'up_signals': up_signals,
            'down_signals': down_signals,
            'range_signals': range_signals,
            'total_signals': total_signals,
            'up_ratio': up_signals / total_signals if total_signals > 0 else 0,
            'down_ratio': down_signals / total_signals if total_signals > 0 else 0,
            'range_ratio': range_signals / total_signals if total_signals > 0 else 0
        },
        'quality_metrics': {
            'correlation': correlation,
            'stability': ma_stability,
            'turning_points_price': len(price_turning_points),
            'turning_points_ma': len(ma_turning_points)
        }
    }


def run_version_comparison(data: pd.DataFrame, symbol: str) -> dict:
    """
    各バージョンでの計算を実行し、結果を比較
    
    Args:
        data: OHLCデータ
        symbol: シンボル名
    
    Returns:
        dict: 各バージョンの結果
    """
    results = {}
    
    print(f"\n🔬 {symbol} - UltimateMA バージョン比較テスト開始")
    print("="*60)
    
    # V1テスト
    print("\n📊 UltimateMA V1 テスト中...")
    try:
        start_time = time.time()
        uma_v1 = UltimateMA(
            super_smooth_period=10,
            zero_lag_period=21,
            realtime_window=89,
            src_type='hlc3',
            slope_index=1,
            range_threshold=0.005
        )
        v1_result = uma_v1.calculate(data)
        v1_calc_time = time.time() - start_time
        
        v1_performance = calculate_performance_metrics(
            data, 
            v1_result.values, 
            v1_result.trend_signals, 
            'V1'
        )
        v1_performance['calc_time'] = v1_calc_time
        v1_performance['current_trend'] = v1_result.current_trend
        v1_performance['result'] = v1_result
        
        results['V1'] = v1_performance
        print(f"✅ V1 完了 (時間: {v1_calc_time:.3f}秒)")
        
    except Exception as e:
        print(f"❌ V1 エラー: {e}")
        results['V1'] = None
    
    # V2テスト
    print("\n📊 UltimateMA V2 テスト中...")
    try:
        start_time = time.time()
        uma_v2 = UltimateMAV2(
            super_smooth_period=12,
            zero_lag_period=24,
            realtime_window=55,
            src_type='hlc3',
            slope_index=2,
            range_threshold=0.008
        )
        v2_result = uma_v2.calculate(data)
        v2_calc_time = time.time() - start_time
        
        v2_performance = calculate_performance_metrics(
            data, 
            v2_result.values, 
            v2_result.trend_signals, 
            'V2'
        )
        v2_performance['calc_time'] = v2_calc_time
        v2_performance['current_trend'] = v2_result.current_trend
        v2_performance['result'] = v2_result
        
        results['V2'] = v2_performance
        print(f"✅ V2 完了 (時間: {v2_calc_time:.3f}秒)")
        
    except Exception as e:
        print(f"❌ V2 エラー: {e}")
        results['V2'] = None
    
    # V3テスト
    print("\n📊 UltimateMA V3 テスト中...")
    try:
        start_time = time.time()
        uma_v3 = UltimateMAV3(
            super_smooth_period=8,
            zero_lag_period=16,
            realtime_window=34,
            quantum_window=16,
            fractal_window=16,
            entropy_window=16,
            src_type='hlc3',
            slope_index=2,
            base_threshold=0.002,
            min_confidence=0.15
        )
        v3_result = uma_v3.calculate(data)
        v3_calc_time = time.time() - start_time
        
        # V3は特別な性能指標も含む
        v3_performance = calculate_performance_metrics(
            data, 
            v3_result.values, 
            v3_result.trend_signals, 
            'V3'
        )
        v3_performance['calc_time'] = v3_calc_time
        v3_performance['current_trend'] = v3_result.current_trend
        v3_performance['result'] = v3_result
        
        # V3特有の指標
        if hasattr(v3_result, 'trend_confidence'):
            v3_performance['v3_specific'] = {
                'avg_confidence': np.nanmean(v3_result.trend_confidence),
                'max_confidence': np.nanmax(v3_result.trend_confidence),
                'quantum_strength': np.nanmean(np.abs(v3_result.quantum_state)) if hasattr(v3_result, 'quantum_state') else 0,
                'mtf_consensus': np.nanmean(v3_result.multi_timeframe_consensus) if hasattr(v3_result, 'multi_timeframe_consensus') else 0,
                'fractal_dimension': np.nanmean(v3_result.fractal_dimension) if hasattr(v3_result, 'fractal_dimension') else 0,
                'entropy_level': np.nanmean(v3_result.entropy_level) if hasattr(v3_result, 'entropy_level') else 0
            }
        
        results['V3'] = v3_performance
        print(f"✅ V3 完了 (時間: {v3_calc_time:.3f}秒)")
        
    except Exception as e:
        print(f"❌ V3 エラー: {e}")
        results['V3'] = None
    
    return results


def plot_comparison_results(data: pd.DataFrame, results: dict, symbol: str):
    """
    比較結果を可視化
    
    Args:
        data: 元のOHLCデータ
        results: 各バージョンの結果
        symbol: シンボル名
    """
    print(f"\n📊 {symbol} 比較チャート作成中...")
    
    # 有効な結果のみを取得
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("❌ 比較に十分な結果がありません")
        return
    
    # 図の作成
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle(f'🚀 UltimateMA V1 vs V2 vs V3 Performance Comparison - {symbol}', 
                 fontsize=16, fontweight='bold')
    
    x_axis = data.index
    colors = {'V1': 'red', 'V2': 'blue', 'V3': 'green'}
    
    # 1. 価格チャートと各バージョンのMA
    ax1 = axes[0, 0]
    ax1.plot(x_axis, data['close'], color='black', alpha=0.7, linewidth=1.0, label='Close Price')
    
    for version, result_data in valid_results.items():
        if 'result' in result_data:
            ma_values = result_data['result'].values
            ax1.plot(x_axis, ma_values, color=colors[version], linewidth=1.5, 
                    label=f'UltimateMA {version}', alpha=0.8)
    
    ax1.set_title('💰 Price Chart with UltimateMA Versions', fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ノイズ除去効果比較
    ax2 = axes[0, 1]
    versions = list(valid_results.keys())
    noise_reductions = [valid_results[v]['noise_reduction']['reduction_percentage'] for v in versions]
    
    bars = ax2.bar(versions, noise_reductions, color=[colors[v] for v in versions], alpha=0.7)
    ax2.set_title('🔇 Noise Reduction Effectiveness', fontweight='bold')
    ax2.set_ylabel('Noise Reduction (%)')
    ax2.set_ylim(0, max(noise_reductions) * 1.2 if noise_reductions else 1)
    
    # 値をバーの上に表示
    for bar, value in zip(bars, noise_reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. 方向精度比較
    ax3 = axes[1, 0]
    direction_accuracies = [valid_results[v]['trend_detection']['direction_accuracy'] * 100 for v in versions]
    
    bars = ax3.bar(versions, direction_accuracies, color=[colors[v] for v in versions], alpha=0.7)
    ax3.set_title('🎯 Direction Accuracy', fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(0, 100)
    
    for bar, value in zip(bars, direction_accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 遅延比較
    ax4 = axes[1, 1]
    delays = [valid_results[v]['trend_detection']['avg_delay'] for v in versions]
    
    bars = ax4.bar(versions, delays, color=[colors[v] for v in versions], alpha=0.7)
    ax4.set_title('⏱️ Average Delay (periods)', fontweight='bold')
    ax4.set_ylabel('Delay (periods)')
    
    for bar, value in zip(bars, delays):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. 計算時間比較
    ax5 = axes[2, 0]
    calc_times = [valid_results[v]['calc_time'] * 1000 for v in versions]  # ミリ秒に変換
    
    bars = ax5.bar(versions, calc_times, color=[colors[v] for v in versions], alpha=0.7)
    ax5.set_title('⚡ Calculation Time', fontweight='bold')
    ax5.set_ylabel('Time (ms)')
    
    for bar, value in zip(bars, calc_times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(calc_times) * 0.02,
                f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 6. 相関係数比較
    ax6 = axes[2, 1]
    correlations = [valid_results[v]['quality_metrics']['correlation'] * 100 for v in versions]
    
    bars = ax6.bar(versions, correlations, color=[colors[v] for v in versions], alpha=0.7)
    ax6.set_title('📈 Price Correlation', fontweight='bold')
    ax6.set_ylabel('Correlation (%)')
    ax6.set_ylim(0, 100)
    
    for bar, value in zip(bars, correlations):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 7. トレンド継続性比較
    ax7 = axes[3, 0]
    continuities = [valid_results[v]['trend_detection']['trend_continuity'] for v in versions]
    
    bars = ax7.bar(versions, continuities, color=[colors[v] for v in versions], alpha=0.7)
    ax7.set_title('📊 Trend Continuity', fontweight='bold')
    ax7.set_ylabel('Average Trend Length')
    
    for bar, value in zip(bars, continuities):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + max(continuities) * 0.02,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. 総合スコア
    ax8 = axes[3, 1]
    
    # 総合スコア計算（各指標を正規化して合計）
    scores = {}
    for version in versions:
        result = valid_results[version]
        
        # 各指標を0-1に正規化（高いほど良い）
        noise_score = min(result['noise_reduction']['reduction_percentage'] / 30, 1.0)  # 30%を最大とする
        accuracy_score = result['trend_detection']['direction_accuracy']
        delay_score = max(0, 1.0 - result['trend_detection']['avg_delay'] / 10)  # 10期間を最大遅延とする
        correlation_score = result['quality_metrics']['correlation']
        speed_score = max(0, 1.0 - result['calc_time'] / 1.0)  # 1秒を最大時間とする
        
        # V3の特別スコア
        if version == 'V3' and 'v3_specific' in result:
            v3_specific = result['v3_specific']
            confidence_score = v3_specific['avg_confidence']
            quantum_score = min(v3_specific['quantum_strength'] / 0.1, 1.0)
            total_score = (noise_score + accuracy_score + delay_score + correlation_score + 
                          speed_score + confidence_score + quantum_score) / 7
        else:
            total_score = (noise_score + accuracy_score + delay_score + correlation_score + speed_score) / 5
        
        scores[version] = total_score * 100
    
    score_values = list(scores.values())
    bars = ax8.bar(versions, score_values, color=[colors[v] for v in versions], alpha=0.7)
    ax8.set_title('🏆 Overall Performance Score', fontweight='bold')
    ax8.set_ylabel('Score (%)')
    ax8.set_ylim(0, 100)
    
    for bar, value in zip(bars, score_values):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # X軸の日付フォーマット（価格チャートのみ）
    ax1.tick_params(axis='x', rotation=45)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    
    # ファイル保存
    filename = f"tests/ultimate_ma_comparison_{symbol.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 比較チャート保存完了: {filename}")
    
    plt.show()
    plt.close()
    
    return scores


def print_detailed_comparison(results: dict, symbol: str):
    """
    詳細な比較結果を表示
    
    Args:
        results: 各バージョンの結果
        symbol: シンボル名
    """
    print(f"\n{'='*80}")
    print(f"🏆 **{symbol} - UltimateMA バージョン比較 詳細結果**")
    print("="*80)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 有効な結果がありません")
        return
    
    # 各バージョンの詳細結果
    for version, result in valid_results.items():
        print(f"\n📊 **UltimateMA {version}**")
        print("-" * 40)
        
        # 基本情報
        print(f"現在のトレンド: {result['current_trend'].upper()}")
        print(f"計算時間: {result['calc_time']:.3f}秒")
        
        # ノイズ除去
        noise = result['noise_reduction']
        print(f"\n🔇 ノイズ除去効果:")
        print(f"  - 元のボラティリティ: {noise['raw_volatility']:.4f}")
        print(f"  - フィルター後: {noise['filtered_volatility']:.4f}")
        print(f"  - 除去率: {noise['reduction_percentage']:.2f}%")
        
        # トレンド検出
        trend = result['trend_detection']
        print(f"\n🎯 トレンド検出性能:")
        print(f"  - 方向精度: {trend['direction_accuracy']*100:.2f}%")
        print(f"  - 平均遅延: {trend['avg_delay']:.2f}期間")
        print(f"  - トレンド継続性: {trend['trend_continuity']:.2f}")
        
        # 信号分布
        signals = result['signal_distribution']
        print(f"\n📈 信号分布:")
        print(f"  - 上昇: {signals['up_signals']}回 ({signals['up_ratio']*100:.1f}%)")
        print(f"  - 下降: {signals['down_signals']}回 ({signals['down_ratio']*100:.1f}%)")
        print(f"  - レンジ: {signals['range_signals']}回 ({signals['range_ratio']*100:.1f}%)")
        
        # 品質指標
        quality = result['quality_metrics']
        print(f"\n📊 品質指標:")
        print(f"  - 価格相関: {quality['correlation']:.4f}")
        print(f"  - 安定性: {quality['stability']:.4f}")
        print(f"  - 転換点検出: 価格{quality['turning_points_price']}個, MA{quality['turning_points_ma']}個")
        
        # V3特有の指標
        if version == 'V3' and 'v3_specific' in result:
            v3_spec = result['v3_specific']
            print(f"\n🌌 V3特有指標:")
            print(f"  - 平均信頼度: {v3_spec['avg_confidence']:.3f}")
            print(f"  - 最大信頼度: {v3_spec['max_confidence']:.3f}")
            print(f"  - 量子強度: {v3_spec['quantum_strength']:.3f}")
            print(f"  - MTF合意度: {v3_spec['mtf_consensus']:.3f}")
            print(f"  - フラクタル次元: {v3_spec['fractal_dimension']:.3f}")
            print(f"  - エントロピー: {v3_spec['entropy_level']:.3f}")
    
    # 勝者判定
    print(f"\n🏆 **総合評価**")
    print("="*40)
    
    # 各カテゴリーでの勝者
    categories = {
        'ノイズ除去': 'noise_reduction.reduction_percentage',
        '方向精度': 'trend_detection.direction_accuracy',
        '低遅延': 'trend_detection.avg_delay',  # 低いほど良い
        '計算速度': 'calc_time',  # 低いほど良い
        '価格相関': 'quality_metrics.correlation'
    }
    
    winners = {}
    for category, metric_path in categories.items():
        best_version = None
        best_value = None
        
        for version, result in valid_results.items():
            # ネストした辞書から値を取得
            value = result
            for key in metric_path.split('.'):
                value = value[key]
            
            if best_value is None:
                best_value = value
                best_version = version
            else:
                # 遅延と計算時間は低いほど良い
                if category in ['低遅延', '計算速度']:
                    if value < best_value:
                        best_value = value
                        best_version = version
                else:
                    if value > best_value:
                        best_value = value
                        best_version = version
        
        winners[category] = (best_version, best_value)
        print(f"{category}: {best_version} ({best_value:.3f})")
    
    # 総合勝者
    version_scores = {}
    for version in valid_results.keys():
        score = 0
        for category, (winner, _) in winners.items():
            if winner == version:
                score += 1
        version_scores[version] = score
    
    overall_winner = max(version_scores, key=version_scores.get)
    print(f"\n🥇 **総合勝者: UltimateMA {overall_winner}** (勝利カテゴリー: {version_scores[overall_winner]}/{len(categories)})")
    
    # 推奨用途
    print(f"\n💡 **推奨用途:**")
    for version in valid_results.keys():
        if version == 'V1':
            print(f"  - V1: シンプルで高速な処理が必要な場合")
        elif version == 'V2':
            print(f"  - V2: バランスの取れた性能が必要な場合")
        elif version == 'V3':
            print(f"  - V3: 最高精度の分析と高度な指標が必要な場合")


def test_multiple_symbols():
    """複数のシンボルで比較テスト"""
    symbols = ['BTC', 'ETH', 'ADA']
    all_results = {}
    
    print("🚀 UltimateMA V1 vs V2 vs V3 - 複数シンボル比較テスト")
    print("="*80)
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"🔬 {symbol} テスト開始")
        print("="*60)
        
        # データ読み込み
        data = load_binance_data(symbol=symbol, market_type='spot', timeframe='4h')
        
        if data is None:
            print(f"❌ {symbol}のデータ読み込みに失敗")
            continue
        
        # 最新1000件を使用
        if len(data) > 1000:
            data = data.tail(1000)
            print(f"📊 最新1000件のデータを使用")
        
        # バージョン比較実行
        results = run_version_comparison(data, symbol)
        
        # 結果表示
        print_detailed_comparison(results, symbol)
        
        # チャート作成
        scores = plot_comparison_results(data, results, symbol)
        
        all_results[symbol] = {
            'results': results,
            'scores': scores
        }
        
        print(f"✅ {symbol} テスト完了")
    
    # 全体サマリー
    if all_results:
        print(f"\n{'='*80}")
        print("🏆 **全シンボル総合結果**")
        print("="*80)
        
        version_wins = {'V1': 0, 'V2': 0, 'V3': 0}
        
        for symbol, data in all_results.items():
            if 'scores' in data:
                scores = data['scores']
                winner = max(scores, key=scores.get)
                version_wins[winner] += 1
                print(f"{symbol}: {winner} (スコア: {scores[winner]:.1f}%)")
        
        print(f"\n🥇 **最終勝者統計:**")
        for version, wins in version_wins.items():
            print(f"  - {version}: {wins}勝")
        
        overall_champion = max(version_wins, key=version_wins.get)
        print(f"\n👑 **総合チャンピオン: UltimateMA {overall_champion}**")


def main():
    print("🚀 UltimateMA V1 vs V2 vs V3 Performance Comparison Test")
    print("実際のBinanceデータでの包括的性能比較検証システム")
    print("="*80)
    
    # 複数シンボルでのテスト実行
    test_multiple_symbols()
    
    print(f"\n✅ UltimateMA バージョン比較テスト完了")
    print("📊 生成されたチャートファイルをご確認ください。")
    print("🌟 各バージョンの特徴を理解して、用途に応じて選択してください！")


if __name__ == "__main__":
    main() 