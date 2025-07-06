#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **Quantum Adaptive Volatility Channel (QAVC) - BTCテスト & 可視化** 🚀

BTCの実データでQAVCの性能をテストし、美しいチャートで可視化します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from indicators.quantum_adaptive_volatility_channel import QuantumAdaptiveVolatilityChannel


def load_binance_data(symbol='BTC', market_type='spot', timeframe='4h', data_dir='data/binance'):
    """Binanceデータを直接読み込む"""
    file_path = f"{data_dir}/{symbol}/{market_type}/{timeframe}/historical_data.csv"
    
    print(f"📂 データファイル読み込み中: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ データファイルが見つかりません: {file_path}")
        print("🔄 サンプルデータを生成します...")
        return generate_sample_btc_data()
    
    try:
        df = pd.read_csv(file_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name != 'timestamp':
            # インデックスがない場合は時系列インデックスを作成
            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='4H')
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 必要なカラムが不足しています: {missing_columns}")
            return generate_sample_btc_data()
        
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        print(f"✅ データ読み込み成功: {symbol} {market_type} {timeframe}")
        print(f"📊 データ期間: {df.index.min()} - {df.index.max()}")
        print(f"📈 データ数: {len(df)}件")
        
        return df
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        print("🔄 サンプルデータを生成します...")
        return generate_sample_btc_data()


def generate_sample_btc_data(n_points=1500):
    """サンプルBTCデータを生成"""
    print("🔄 サンプルBTCデータを生成中...")
    
    np.random.seed(42)
    
    # 現実的なBTC価格パターンを作成
    base_price = 50000
    
    # トレンド成分
    trend = np.cumsum(np.random.randn(n_points) * 0.002)
    
    # サイクル成分（短期と長期）
    short_cycle = np.sin(np.linspace(0, 20*np.pi, n_points)) * 0.3
    long_cycle = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.8
    
    # ボラティリティイベント
    volatility_events = np.zeros(n_points)
    for i in range(5):  # 5回のボラティリティイベント
        event_start = np.random.randint(100, n_points-100)
        event_length = np.random.randint(20, 50)
        event_magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
        volatility_events[event_start:event_start+event_length] = event_magnitude
    
    # 価格組み立て
    log_price = np.log(base_price) + trend + short_cycle + long_cycle + volatility_events
    close_prices = np.exp(log_price)
    
    # ノイズ追加
    noise = np.random.randn(n_points) * 0.01
    close_prices *= (1 + noise)
    
    # OHLC作成
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_points) * 0.02))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_points) * 0.02))
    open_prices = close_prices * (1 + np.random.randn(n_points) * 0.005)
    
    # 日付インデックス作成
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='4H')
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)
    
    print(f"✅ サンプルデータ生成完了 ({n_points}件)")
    
    return df


def test_qavc_with_btc():
    """BTCデータでQAVCをテスト"""
    print("🚀 BTC データでのQAVC分析開始")
    print("="*60)
    
    # BTCデータ読み込み
    btc_data = load_binance_data(symbol='BTC', market_type='spot', timeframe='4h')
    
    if btc_data is None or len(btc_data) == 0:
        print("❌ BTCデータの取得に失敗しました")
        return None
    
    # 最新1000件を使用
    if len(btc_data) > 1000:
        btc_data = btc_data.tail(1000)
        print(f"📊 最新1000件のデータを使用: {btc_data.index.min()} - {btc_data.index.max()}")
    
    # QAVC初期化
    print("\n🔧 QAVC初期化中...")
    qavc = QuantumAdaptiveVolatilityChannel(
        volatility_period=21,      # ATR期間
        base_multiplier=2.0,       # 基本チャネル幅倍率
        fractal_window=50,         # フラクタル解析ウィンドウ
        spectral_window=64,        # スペクトル解析ウィンドウ
        entropy_window=50,         # エントロピー解析ウィンドウ
        entropy_max_scale=5,       # エントロピー最大スケール
        src_type='hlc3'           # 価格ソース
    )
    
    # 計算実行
    print("⚡ QAVC計算実行中...")
    start_time = time.time()
    result = qavc.calculate(btc_data)
    calc_time = time.time() - start_time
    
    # 結果の検証
    if result is None:
        print("❌ QAVC計算に失敗しました")
        return None
    
    # 統計計算
    total_breakout_signals = int(np.sum(np.abs(result.breakout_signals)))
    total_exit_signals = int(np.sum(np.abs(result.exit_signals)))
    
    # 有効なシグナル強度
    valid_strengths = result.signal_strength[result.signal_strength > 0]
    avg_signal_strength = np.mean(valid_strengths) if len(valid_strengths) > 0 else 0.0
    
    # 分析サマリー
    analysis_summary = qavc.get_analysis_summary()
    
    print(f"\n✅ QAVC計算完了!")
    print(f"⚡ 計算時間: {calc_time:.3f}秒 ({len(btc_data)}ポイント)")
    print(f"🎯 現在レジーム: {result.current_regime}")
    print(f"💪 トレンド強度: {result.current_trend_strength:.3f}")
    print(f"🌪️ ボラティリティレベル: {result.current_volatility_level}")
    print(f"📊 ブレイクアウト信号: {total_breakout_signals}回")
    print(f"🚪 エグジット信号: {total_exit_signals}回")
    print(f"🔥 平均シグナル強度: {avg_signal_strength:.3f}")
    print(f"🏆 チャネル効率: {analysis_summary.get('channel_efficiency', 0.0):.3f}")
    print(f"🌀 フラクタル次元: {analysis_summary.get('latest_fractal_dimension', 1.5):.3f}")
    print(f"📡 支配的サイクル: {analysis_summary.get('latest_dominant_cycle', 20.0):.1f}")
    
    return btc_data, result, analysis_summary


def create_qavc_chart(btc_data, result, analysis_summary):
    """Beautiful QAVC Chart Creation"""
    print("\n📈 Creating Beautiful QAVC Chart...")
    
    # Display period (Latest 300 periods)
    display_periods = 300
    if len(btc_data) > display_periods:
        plot_data = btc_data.tail(display_periods)
        plot_result_slice = slice(-display_periods, None)
    else:
        plot_data = btc_data
        plot_result_slice = slice(None)
    
    # Color theme
    colors = {
        'price': '#1f77b4',
        'channel_fill': '#87CEEB',
        'upper_channel': '#4169E1',
        'lower_channel': '#4169E1',
        'midline': '#FF6347',
        'breakout_up': '#00FF00',
        'breakout_down': '#FF0000',
        'exit': '#FFD700',
        'regime_colors': {
            'range': '#87CEEB',
            'trend': '#98FB98', 
            'breakout': '#FFB6C1',
            'crash': '#FF6347'
        }
    }
    
    # Create chart with English fonts
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Use English font
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle('🚀 Quantum Adaptive Volatility Channel (QAVC) - BTC Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Main Chart (Price + Channel)
    ax1 = axes[0]
    
    # Price line
    x_indices = range(len(plot_data))
    ax1.plot(x_indices, plot_data['close'], color=colors['price'], linewidth=1.5, alpha=0.8, label='BTC Price')
    
    # Verify channel data is valid
    upper_data = result.upper_channel[plot_result_slice]
    lower_data = result.lower_channel[plot_result_slice]
    midline_data = result.midline[plot_result_slice]
    
    # Remove NaN values and ensure proper scaling
    valid_indices = np.isfinite(upper_data) & np.isfinite(lower_data) & np.isfinite(midline_data)
    
    if np.any(valid_indices):
        # Channel fill
        ax1.fill_between(x_indices, upper_data, lower_data,
                        alpha=0.2, color=colors['channel_fill'], label='QAVC Channel Zone')
        
        # Channel lines with thicker lines for visibility
        ax1.plot(x_indices, upper_data, 
                '--', color=colors['upper_channel'], linewidth=2.5, alpha=0.9, label='Upper Channel')
        ax1.plot(x_indices, lower_data, 
                '--', color=colors['lower_channel'], linewidth=2.5, alpha=0.9, label='Lower Channel')
        
        # Midline (12-layer filtered)
        ax1.plot(x_indices, midline_data, 
                color=colors['midline'], linewidth=3, alpha=0.9, label='Quantum Filtered Midline')
        
        # Add channel width statistics
        channel_width = np.mean(upper_data - lower_data)
        price_mean = np.mean(plot_data['close'])
        width_percentage = (channel_width / price_mean) * 100
        
        # Add text box with channel info
        textstr = f'Channel Width: {width_percentage:.2f}%\nAvg Width: ${channel_width:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    else:
        print("⚠️ Warning: Channel data contains invalid values!")
    
    # Breakout signals
    breakout_signals = result.breakout_signals[plot_result_slice]
    exit_signals = result.exit_signals[plot_result_slice]
    
    breakout_up_indices = [i for i, signal in enumerate(breakout_signals) if signal == 1]
    breakout_down_indices = [i for i, signal in enumerate(breakout_signals) if signal == -1]
    exit_indices = [i for i, signal in enumerate(exit_signals) if signal != 0]
    
    if breakout_up_indices:
        ax1.scatter([x_indices[i] for i in breakout_up_indices], 
                   [plot_data.iloc[i]['close'] * 1.01 for i in breakout_up_indices],
                   marker='^', s=120, color=colors['breakout_up'], 
                   edgecolor='black', linewidth=2, zorder=5, label='Upward Breakout')
    
    if breakout_down_indices:
        ax1.scatter([x_indices[i] for i in breakout_down_indices], 
                   [plot_data.iloc[i]['close'] * 0.99 for i in breakout_down_indices],
                   marker='v', s=120, color=colors['breakout_down'], 
                   edgecolor='black', linewidth=2, zorder=5, label='Downward Breakout')
    
    if exit_indices:
        ax1.scatter([x_indices[i] for i in exit_indices], 
                   [plot_data.iloc[i]['close'] for i in exit_indices],
                   marker='x', s=100, color=colors['exit'], 
                   linewidth=4, zorder=5, label='Exit Signal')
    
    ax1.set_title('💰 BTC Price + Quantum Adaptive Volatility Channel', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # 2. Regime State Chart
    ax2 = axes[1]
    regime_states = result.regime_state[plot_result_slice]
    regime_names = ['Range', 'Trend', 'Breakout', 'Crash']
    regime_colors_list = []
    
    for state in regime_states:
        state_idx = int(min(max(state, 0), 3))  # Clamp to valid range
        regime_name = regime_names[state_idx]
        if regime_name == 'Range':
            regime_colors_list.append(colors['regime_colors']['range'])
        elif regime_name == 'Trend':
            regime_colors_list.append(colors['regime_colors']['trend'])
        elif regime_name == 'Breakout':
            regime_colors_list.append(colors['regime_colors']['breakout'])
        else:  # Crash
            regime_colors_list.append(colors['regime_colors']['crash'])
    
    ax2.bar(x_indices, np.ones(len(x_indices)), color=regime_colors_list, alpha=0.7)
    ax2.set_title('🧠 Market Regime State', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Regime', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.25, 0.75])
    ax2.set_yticklabels(['Range/Trend', 'Breakout/Crash'])
    
    # Regime legend
    from matplotlib.patches import Patch
    regime_legend = [Patch(facecolor=colors['regime_colors']['range'], label='Range'),
                    Patch(facecolor=colors['regime_colors']['trend'], label='Trend'),
                    Patch(facecolor=colors['regime_colors']['breakout'], label='Breakout'),
                    Patch(facecolor=colors['regime_colors']['crash'], label='Crash')]
    ax2.legend(handles=regime_legend, loc='upper right', fontsize=8)
    
    # 3. Signal Strength Chart
    ax3 = axes[2]
    signal_strength = result.signal_strength[plot_result_slice]
    ax3.plot(x_indices, signal_strength, color='purple', linewidth=2, alpha=0.8)
    ax3.fill_between(x_indices, 0, signal_strength, alpha=0.3, color='purple')
    ax3.set_title('💪 Signal Strength', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Strength', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add horizontal lines for strength levels
    ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Strong Signal')
    ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Weak Signal')
    ax3.legend(fontsize=8)
    
    # 4. Volatility Forecast Chart  
    ax4 = axes[3]
    volatility_forecast = result.volatility_forecast[plot_result_slice]
    ax4.plot(x_indices, volatility_forecast, color='orange', linewidth=2, alpha=0.8)
    ax4.fill_between(x_indices, 0, volatility_forecast, alpha=0.3, color='orange')
    ax4.set_title('📊 Volatility Forecast (GARCH)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Volatility', fontweight='bold')
    ax4.set_xlabel('Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # X-axis labels (apply to all subplots)
    if len(plot_data) > 10:
        step = max(1, len(plot_data) // 8)
        time_positions = list(range(0, len(plot_data), step))
        time_labels = [plot_data.index[i].strftime('%m/%d %H:%M') for i in time_positions]
        
        for ax in axes:
            ax.set_xticks(time_positions)
            ax.set_xticklabels(time_labels, rotation=45)
    
    # Statistics text box
    stats_text = f"""📊 QAVC Statistics:
🎯 Current Regime: {result.current_regime}
💪 Trend Strength: {result.current_trend_strength:.3f}
🌪️ Volatility: {result.current_volatility_level}
🏆 Channel Efficiency: {analysis_summary.get('channel_efficiency', 0.0):.3f}
🌀 Fractal Dimension: {analysis_summary.get('latest_fractal_dimension', 1.5):.3f}
📡 Dominant Cycle: {analysis_summary.get('latest_dominant_cycle', 20.0):.1f}"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save file
    filename = "tests/btc_qavc_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ BTC Chart saved: {filename}")
    
    # Debug information
    print(f"📊 Channel Statistics:")
    print(f"   Upper channel range: {np.min(upper_data):.2f} - {np.max(upper_data):.2f}")
    print(f"   Lower channel range: {np.min(lower_data):.2f} - {np.max(lower_data):.2f}")
    print(f"   Price range: {np.min(plot_data['close']):.2f} - {np.max(plot_data['close']):.2f}")
    print(f"   Channel width: {np.mean(upper_data - lower_data):.2f}")
    
    plt.show()
    plt.close()


def main():
    """Main execution function"""
    print("🚀 BTC Quantum Adaptive Volatility Channel Test")
    print("="*60)
    
    # Test QAVC with BTC data
    test_result = test_qavc_with_btc()
    
    if test_result is None:
        print("❌ Test failed")
        return
    
    btc_data, result, analysis_summary = test_result
    
    # Create beautiful chart
    create_qavc_chart(btc_data, result, analysis_summary)
    
    print(f"\n{'='*60}")
    print("🏆 BTC QAVC Analysis Complete!")
    print("="*60)
    print("📊 Please check the generated chart file.")
    print("🌟 Experience the amazing performance of Quantum Adaptive Volatility Channel!")
    print("🚀 Ultimate cosmic breakout strategy BTC analysis complete!")


if __name__ == "__main__":
    main() 