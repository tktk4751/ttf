#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UltimateMA V3 Real Data Demo with Matplotlib Visualization
実際のBinanceデータを直接読み込んでテストし、チャートで可視化するデモ
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

# UltimateMA V3のインポート
from ultimate_ma_v3 import UltimateMAV3


def load_binance_data(symbol='BTC', market_type='spot', timeframe='4h', data_dir='data/binance'):
    """
    Binanceデータを直接読み込む
    
    Args:
        symbol: シンボル名 (BTC, ETH, etc.)
        market_type: 市場タイプ (spot, future)
        timeframe: 時間足 (1h, 4h, 1d, etc.)
        data_dir: データディレクトリのパス
    
    Returns:
        pd.DataFrame: OHLCVデータ
    """
    # 現在のディレクトリがindicatorsの場合、一つ上に移動
    if os.path.basename(os.getcwd()) == 'indicators':
        data_dir = f"../{data_dir}"
    
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
        print(f"💰 価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return None


def plot_ultimate_ma_v3_real_data(data: pd.DataFrame, result, symbol: str, save_path: str = None):
    """
    実際のBinanceデータでのUltimateMA V3結果を包括的に可視化
    
    Args:
        data: 元のOHLCデータ
        result: UltimateMA V3の計算結果
        symbol: シンボル名
        save_path: 保存パス（Noneの場合は自動生成）
    """
    n_points = len(data)
    print(f"📊 {symbol} UltimateMA V3チャート描画中... データ点数: {n_points}")
    
    # 時系列インデックスの準備
    x_axis = data.index
    use_datetime = True
    
    # 図の作成（8つのサブプロット）
    fig, axes = plt.subplots(8, 1, figsize=(20, 28))
    
    title = f'🚀 UltimateMA V3 - {symbol} Real Binance Data Analysis\n📊 実際の相場データ ({n_points}件) | 期間: {data.index.min().strftime("%Y-%m-%d")} - {data.index.max().strftime("%Y-%m-%d")}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 1. 価格チャートとUltimateMA V3
    ax1 = axes[0]
    
    # ローソク足風の価格表示
    ax1.plot(x_axis, data['high'], color='lightgray', alpha=0.3, linewidth=0.5, label='High')
    ax1.plot(x_axis, data['low'], color='lightgray', alpha=0.3, linewidth=0.5, label='Low')
    ax1.plot(x_axis, data['close'], color='black', alpha=0.7, linewidth=1.0, label='Close Price')
    
    # UltimateMA V3ライン
    ax1.plot(x_axis, result.values, color='blue', linewidth=2.0, label='UltimateMA V3', alpha=0.9)
    
    # トレンド背景色
    try:
        signals = result.trend_signals
        if len(signals) > 0:
            current_signal = signals[0]
            start_idx = 0
            
            for i in range(1, len(signals) + 1):
                if i == len(signals) or signals[i] != current_signal:
                    end_idx = i - 1
                    
                    if current_signal == 1:  # 上昇トレンド
                        color = 'lightgreen'
                        alpha = 0.15
                    elif current_signal == -1:  # 下降トレンド
                        color = 'lightcoral'
                        alpha = 0.15
                    else:  # レンジ
                        color = 'lightyellow'
                        alpha = 0.1
                    
                    if start_idx < len(x_axis) and end_idx < len(x_axis):
                        ax1.axvspan(x_axis[start_idx], x_axis[end_idx], color=color, alpha=alpha, zorder=0)
                    
                    if i < len(signals):
                        start_idx = i
                        current_signal = signals[i]
    except Exception as e:
        print(f"⚠️  背景色設定でエラー: {e}")
    
    ax1.set_title(f'💰 {symbol} Price Chart with UltimateMA V3 (Green=Up, Red=Down, Yellow=Range)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 信頼度とトレンド強度
    ax2 = axes[1]
    
    # 信頼度をカラーマップで表示
    scatter = ax2.scatter(x_axis, result.trend_confidence, c=result.trend_confidence, 
                         cmap='viridis', alpha=0.7, s=12, zorder=2)
    ax2.plot(x_axis, result.trend_confidence, color='orange', alpha=0.5, linewidth=1.0, zorder=1)
    
    # 信頼度閾値ライン
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Confidence (0.5)')
    ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Ultra Confidence (0.7)')
    
    ax2.set_title('🔥 Trend Confidence Levels', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Confidence Level', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # カラーバーの追加
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Confidence Level', fontsize=10)
    
    # 3. 量子状態とMTF合意度
    ax3 = axes[2]
    ax3.plot(x_axis, result.quantum_state, label='🌌 Quantum State', color='purple', linewidth=1.5, alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x_axis, result.multi_timeframe_consensus, label='🔄 MTF Consensus', color='blue', linewidth=1.2, alpha=0.7)
    
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3_twin.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong Consensus (0.8)')
    
    ax3.set_title('🌌 Quantum State & Multi-Timeframe Consensus', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Quantum State', fontsize=12, color='purple')
    ax3_twin.set_ylabel('MTF Consensus', fontsize=12, color='blue')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. フラクタル次元とエントロピー
    ax4 = axes[3]
    ax4.plot(x_axis, result.fractal_dimension, label='🌀 Fractal Dimension', color='green', linewidth=1.5, alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x_axis, result.entropy_level, label='🔬 Entropy Level', color='red', linewidth=1.2, alpha=0.7)
    
    # 理想的な値のライン
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal Fractal (1.0)')
    ax4_twin.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Medium Entropy (0.5)')
    
    ax4.set_title('🌀 Fractal Dimension & Entropy Analysis', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Fractal Dimension', fontsize=12, color='green')
    ax4_twin.set_ylabel('Entropy Level', fontsize=12, color='red')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. ボラティリティレジーム
    ax5 = axes[4]
    
    # ボラティリティレジームを色分け表示
    regime_colors = {0: 'blue', 1: 'green', 2: 'red'}
    regime_labels = {0: '低ボラティリティ', 1: '正常', 2: '高ボラティリティ'}
    
    for regime_value in [0, 1, 2]:
        mask = result.volatility_regime == regime_value
        if np.any(mask):
            ax5.scatter(np.array(x_axis)[mask], data['close'].values[mask], 
                       c=regime_colors[regime_value], label=regime_labels[regime_value], 
                       alpha=0.6, s=8)
    
    ax5.plot(x_axis, data['close'], color='black', alpha=0.3, linewidth=0.8)
    ax5.set_title('📊 Volatility Regime Detection', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Price (USD)', fontsize=12)
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. ヒルベルト変換 - 振幅と位相
    ax6 = axes[5]
    ax6.plot(x_axis, result.amplitude, label='Instantaneous Amplitude', color='purple', linewidth=1.2, alpha=0.8)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(x_axis, result.phase, label='Instantaneous Phase', color='orange', linewidth=1.0, alpha=0.7)
    
    # ノイズ閾値
    if len(result.amplitude) > 0:
        noise_threshold = np.mean(result.amplitude) * 0.3
        ax6.axhline(y=noise_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Noise Threshold ({noise_threshold:.1f})')
    
    ax6.set_title('🌀 Hilbert Transform - Amplitude & Phase', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Amplitude', fontsize=12, color='purple')
    ax6_twin.set_ylabel('Phase (radians)', fontsize=12, color='orange')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # 7. リアルタイムトレンド検出
    ax7 = axes[6]
    positive_mask = result.realtime_trends > 0
    negative_mask = result.realtime_trends < 0
    
    ax7.fill_between(x_axis, 0, result.realtime_trends, where=positive_mask, 
                    color='green', alpha=0.6, label='Bullish Trend')
    ax7.fill_between(x_axis, 0, result.realtime_trends, where=negative_mask, 
                    color='red', alpha=0.6, label='Bearish Trend')
    ax7.plot(x_axis, result.realtime_trends, color='black', linewidth=0.8, alpha=0.7)
    
    ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax7.set_title('⚡ Real-Time Trend Detector', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Trend Strength', fontsize=12)
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    # 8. 統合分析サマリー（最新の統計）
    ax8 = axes[7]
    
    # 最新の分析指標
    latest_confidence = result.trend_confidence[-1] if len(result.trend_confidence) > 0 else 0
    latest_quantum = result.quantum_state[-1] if len(result.quantum_state) > 0 else 0
    latest_mtf = result.multi_timeframe_consensus[-1] if len(result.multi_timeframe_consensus) > 0 else 0
    latest_fractal = result.fractal_dimension[-1] if len(result.fractal_dimension) > 0 else 0
    latest_entropy = result.entropy_level[-1] if len(result.entropy_level) > 0 else 0
    
    metrics = [
        latest_confidence,
        latest_mtf,
        1.0 - latest_entropy,  # 予測可能性
        min(1.0, abs(latest_quantum) * 2),  # 量子強度（正規化）
        latest_fractal
    ]
    
    metric_names = ['Current\nConfidence', 'MTF\nConsensus', 'Predictability', 'Quantum\nStrength', 'Fractal\nDimension']
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    bars = ax8.bar(metric_names, metrics, color=colors, alpha=0.7)
    ax8.set_title(f'🎯 {symbol} Current Analysis Summary', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Score (0-1)', fontsize=12)
    ax8.set_ylim(0, 1)
    
    # 値をバーの上に表示
    for bar, metric in zip(bars, metrics):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{metric:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax8.grid(True, alpha=0.3)
    plt.setp(ax8.get_xticklabels(), rotation=0, ha='center')
    
    # X軸の設定（日付フォーマット）
    for ax in axes[:-1]:  # 最後のサマリーチャート以外
        ax.tick_params(axis='x', rotation=45)
        if use_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    axes[-1].set_xlabel('Analysis Metrics', fontsize=12)
    
    # レイアウト調整
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    
    # 統計情報をテキストで追加
    noise_reduction = ((np.nanstd(result.raw_values) - np.nanstd(result.values)) / 
                      np.nanstd(result.raw_values) * 100) if np.nanstd(result.raw_values) > 0 else 0
    
    confident_signals = result.trend_confidence[result.trend_confidence > 0]
    avg_confidence = np.mean(confident_signals) if len(confident_signals) > 0 else 0
    
    up_signals = np.sum(result.trend_signals == 1)
    down_signals = np.sum(result.trend_signals == -1)
    range_signals = np.sum(result.trend_signals == 0)
    
    stats_text = f"""{symbol} UltimateMA V3 - Real Data Analysis:
Current Trend: {result.current_trend.upper()} (Confidence: {result.current_confidence:.3f})
Data Period: {data.index.min().strftime('%Y-%m-%d')} - {data.index.max().strftime('%Y-%m-%d')}
Data Points: {len(data)} | Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}
Noise Reduction: {noise_reduction:.1f}% | Average Confidence: {avg_confidence:.3f}
Trend Distribution: Up {up_signals}({up_signals/len(data)*100:.1f}%) | Down {down_signals}({down_signals/len(data)*100:.1f}%) | Range {range_signals}({range_signals/len(data)*100:.1f}%)
Quantum State: {np.mean(result.quantum_state):.3f} | MTF Consensus: {np.mean(result.multi_timeframe_consensus):.3f}
Fractal Dimension: {np.mean(result.fractal_dimension):.3f} | Entropy: {np.mean(result.entropy_level):.3f}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # ファイル保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ {symbol} UltimateMA V3チャート保存完了 ({save_path})")
    else:
        filename = f"ultimate_ma_v3_{symbol.lower()}_real_data_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ {symbol} UltimateMA V3チャート保存完了 ({filename})")
    
    plt.show()  # チャートを表示
    plt.close()


def analyze_ultimate_ma_v3_performance(result) -> dict:
    """UltimateMA V3のパフォーマンス分析"""
    # 基本ノイズ除去効果
    raw_std = np.nanstd(result.raw_values)
    final_std = np.nanstd(result.values)
    noise_reduction_ratio = (raw_std - final_std) / raw_std if raw_std > 0 else 0.0
    
    # トレンド統計
    trend_signals = result.trend_signals
    up_periods = np.sum(trend_signals == 1)
    down_periods = np.sum(trend_signals == -1)
    range_periods = np.sum(trend_signals == 0)
    total_periods = len(trend_signals)
    
    # 信頼度統計
    confident_signals = result.trend_confidence[result.trend_confidence > 0]
    high_confidence_signals = result.trend_confidence[result.trend_confidence > 0.5]
    ultra_confidence_signals = result.trend_confidence[result.trend_confidence > 0.7]
    
    # 量子分析統計
    quantum_stats = {
        'mean_quantum_state': np.nanmean(result.quantum_state),
        'quantum_strength': np.nanmean(np.abs(result.quantum_state)),
        'mtf_consensus_avg': np.nanmean(result.multi_timeframe_consensus),
        'fractal_dimension_avg': np.nanmean(result.fractal_dimension),
        'entropy_level_avg': np.nanmean(result.entropy_level),
        'predictability': 1.0 - np.nanmean(result.entropy_level)
    }
    
    return {
        'noise_reduction': {
            'raw_volatility': raw_std,
            'filtered_volatility': final_std,
            'reduction_ratio': noise_reduction_ratio,
            'reduction_percentage': noise_reduction_ratio * 100
        },
        'trend_analysis': {
            'total_periods': total_periods,
            'up_periods': up_periods,
            'down_periods': down_periods,
            'range_periods': range_periods,
            'up_ratio': up_periods / total_periods if total_periods > 0 else 0,
            'down_ratio': down_periods / total_periods if total_periods > 0 else 0,
            'range_ratio': range_periods / total_periods if total_periods > 0 else 0,
            'current_trend': result.current_trend,
            'current_confidence': result.current_confidence
        },
        'confidence_analysis': {
            'total_confident_signals': len(confident_signals),
            'high_confidence_signals': len(high_confidence_signals),
            'ultra_confidence_signals': len(ultra_confidence_signals),
            'avg_confidence': np.mean(confident_signals) if len(confident_signals) > 0 else 0,
            'max_confidence': np.max(result.trend_confidence),
            'confidence_ratio': len(confident_signals) / total_periods if total_periods > 0 else 0
        },
        'quantum_analysis': quantum_stats
    }


def test_with_visualization(symbol='BTC', n_points=1000):
    """指定されたシンボルでテストし、可視化する"""
    print(f"\n{'='*20} {symbol} 可視化テスト {'='*20}")
    
    # データ読み込み
    data = load_binance_data(symbol=symbol, market_type='spot', timeframe='4h')
    
    if data is None:
        print(f"❌ {symbol}のデータ読み込みに失敗しました")
        return None
    
    # 指定された件数のデータを使用
    if len(data) > n_points:
        data = data.tail(n_points)
        print(f"📊 最新{n_points}件のデータを使用: {data.index.min()} - {data.index.max()}")
    
    # UltimateMA V3初期化
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
    
    # 計算実行
    print(f"⚡ {symbol} UltimateMA V3計算実行中...")
    start_time = time.time()
    result = uma_v3.calculate(data)
    calc_time = time.time() - start_time
    
    print(f"✅ {symbol} 計算完了 (時間: {calc_time:.2f}秒, 速度: {len(data)/calc_time:.0f} データ/秒)")
    
    # パフォーマンス分析
    performance = analyze_ultimate_ma_v3_performance(result)
    
    # 結果表示
    noise_stats = performance['noise_reduction']
    trend_stats = performance['trend_analysis']
    conf_stats = performance['confidence_analysis']
    quantum_stats = performance['quantum_analysis']
    
    print(f"\n📊 {symbol} 結果サマリー:")
    print(f"   現在のトレンド: {trend_stats['current_trend'].upper()} (信頼度: {trend_stats['current_confidence']:.3f})")
    print(f"   ノイズ除去率: {noise_stats['reduction_percentage']:.1f}%")
    print(f"   平均信頼度: {conf_stats['avg_confidence']:.3f}")
    print(f"   量子強度: {quantum_stats['quantum_strength']:.3f}")
    print(f"   MTF合意度: {quantum_stats['mtf_consensus_avg']:.3f}")
    print(f"   予測可能性: {quantum_stats['predictability']:.3f}")
    
    # チャート描画
    print(f"\n📊 {symbol} チャート描画中...")
    plot_ultimate_ma_v3_real_data(data, result, symbol)
    
    return {
        'data': data,
        'result': result,
        'performance': performance
    }


def main():
    print("🚀 UltimateMA V3 - Real Binance Data Demo with Matplotlib Visualization")
    print("量子ニューラル・フラクタル・エントロピー統合分析システム")
    print("実際のBinanceデータでのテスト + チャート可視化")
    print("="*80)
    
    # 利用可能なシンボル
    available_symbols = ['BTC', 'ETH', 'ADA', 'ATOM', 'AVAX']
    
    print(f"\n📊 利用可能なシンボル: {', '.join(available_symbols)}")
    print("各シンボルでテストし、詳細なチャートを生成します。")
    
    # 各シンボルでテスト実行
    results = {}
    
    for symbol in available_symbols[:3]:  # 最初の3つのシンボルでテスト
        try:
            result = test_with_visualization(symbol=symbol, n_points=800)
            if result:
                results[symbol] = result
                print(f"✅ {symbol} テスト完了")
            else:
                print(f"❌ {symbol} テスト失敗")
        except Exception as e:
            print(f"❌ {symbol} テスト中にエラー: {e}")
        
        print("-" * 60)
    
    # 総合結果
    if results:
        print(f"\n{'='*80}")
        print("🏆 **総合テスト結果**")
        print("="*80)
        
        for symbol, data in results.items():
            perf = data['performance']
            
            noise_reduction = perf['noise_reduction']['reduction_percentage']
            avg_confidence = perf['confidence_analysis']['avg_confidence']
            current_trend = perf['trend_analysis']['current_trend']
            
            print(f"\n{symbol}:")
            print(f"  現在のトレンド: {current_trend.upper()}")
            print(f"  ノイズ除去率: {noise_reduction:.1f}%")
            print(f"  平均信頼度: {avg_confidence:.3f}")
            print(f"  チャートファイル: ultimate_ma_v3_{symbol.lower()}_real_data_analysis.png")
    
    print(f"\n✅ UltimateMA V3 Real Data Demo with Visualization 完了")
    print("🌟 実際のBinanceデータでの量子ニューラル分析 + チャート可視化完了！")
    print("📊 生成されたチャートファイルをご確認ください。")


if __name__ == "__main__":
    main() 