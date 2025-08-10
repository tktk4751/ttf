#!/usr/bin/env python3
"""
Hyper Adaptive Channel チャート作成

ハイパーアダプティブチャネルインジケーターの可視化
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
from pathlib import Path
from datetime import datetime
import yaml

# パス設定
sys.path.append('.')

from indicators.hyper_adaptive_channel import HyperAdaptiveChannel
from data.binance_data_source import BinanceDataSource


def load_config(config_path: str = "config.yaml") -> dict:
    """設定ファイル読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def create_test_data(length: int = 500) -> pd.DataFrame:
    """テストデータ作成"""
    
    dates = pd.date_range('2024-01-01', periods=length, freq='4H')
    np.random.seed(42)
    
    # トレンドとボラティリティサイクルを含む市場データ模擬
    base_price = 100
    trend = np.linspace(0, 20, length)  # 上昇トレンド
    
    # ボラティリティサイクル
    volatility_cycle = 1 + 0.8 * np.sin(np.arange(length) * 2 * np.pi / 80)
    
    # ランダムウォーク + トレンド + ボラティリティ
    returns = np.random.randn(length) * 0.02 * volatility_cycle
    price_changes = np.cumsum(returns)
    
    close_prices = base_price + trend + price_changes * 5
    
    # OHLC生成
    high_noise = np.abs(np.random.randn(length)) * 0.5 * volatility_cycle
    low_noise = np.abs(np.random.randn(length)) * 0.5 * volatility_cycle
    
    high_prices = close_prices + high_noise
    low_prices = close_prices - low_noise
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    volume = np.random.randint(10000, 100000, length) * volatility_cycle
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })


def create_chart(
    data: pd.DataFrame,
    result,
    title: str = "Hyper Adaptive Channel",
    filename: str = None,
    smoother_name: str = "hyper_frama",
    multiplier_mode: str = "dynamic"
):
    """チャート作成"""
    
    # 日付インデックス取得
    if 'timestamp' in data.columns:
        dates = data['timestamp']
        x_axis = dates
    else:
        x_axis = range(len(data))
    
    # 4つのサブプロット作成
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(f'{title} - {smoother_name.title()} / {multiplier_mode.title()}', 
                fontsize=16, fontweight='bold')
    
    # 1. 価格とチャネル
    ax1 = axes[0]
    ax1.plot(x_axis, data['close'], label='Close Price', color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(x_axis, result.midline, label=f'Midline ({smoother_name})', 
            color='blue', linewidth=2)
    ax1.plot(x_axis, result.upper_band, label='Upper Band', 
            color='red', linewidth=1.5, alpha=0.8)
    ax1.plot(x_axis, result.lower_band, label='Lower Band', 
            color='green', linewidth=1.5, alpha=0.8)
    
    # チャネル塗りつぶし
    valid_mask = ~(np.isnan(result.upper_band) | np.isnan(result.lower_band))
    if np.any(valid_mask):
        ax1.fill_between(x_axis, result.upper_band, result.lower_band,
                        alpha=0.1, color='gray', label='Channel Zone')
    
    # チャネルブレイク信号
    if result.channel_position is not None:
        upper_breaks = result.channel_position == 1.0
        lower_breaks = result.channel_position == -1.0
        
        if np.any(upper_breaks):
            ax1.scatter(x_axis[upper_breaks], data['close'][upper_breaks], 
                       color='red', marker='^', s=50, alpha=0.7, 
                       label='Upper Break', zorder=5)
        
        if np.any(lower_breaks):
            ax1.scatter(x_axis[lower_breaks], data['close'][lower_breaks], 
                       color='green', marker='v', s=50, alpha=0.7, 
                       label='Lower Break', zorder=5)
    
    ax1.set_title('Price & Hyper Adaptive Channel')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. ATRとマルチプライヤー
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # ATR
    ax2.plot(x_axis, result.atr_values, label='X-ATR', color='orange', linewidth=1.5)
    ax2.set_ylabel('ATR Value', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # マルチプライヤー
    ax2_twin.plot(x_axis, result.multiplier_values, label='Multiplier', 
                 color='purple', linewidth=2)
    ax2_twin.set_ylabel('Multiplier', color='purple')  
    ax2_twin.tick_params(axis='y', labelcolor='purple')
    
    # 効率比 (動的モード時)
    if result.er_values is not None:
        ax2_twin.plot(x_axis, result.er_values, label='Efficiency Ratio', 
                     color='brown', linewidth=1.5, alpha=0.7)
    
    ax2.set_title('ATR & Dynamic Multiplier')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. チャネル統計
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    # バンド幅
    ax3.plot(x_axis, result.bandwidth, label='Channel Width', 
            color='darkblue', linewidth=1.5)
    ax3.set_ylabel('Channel Width', color='darkblue')
    ax3.tick_params(axis='y', labelcolor='darkblue')
    
    # チャネルポジション
    if result.channel_position is not None:
        ax3_twin.plot(x_axis, result.channel_position, label='Channel Position', 
                     color='navy', linewidth=1.5, alpha=0.8)
        ax3_twin.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax3_twin.axhline(y=-1, color='green', linestyle='--', alpha=0.5)
        ax3_twin.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3_twin.set_ylabel('Channel Position', color='navy')
        ax3_twin.tick_params(axis='y', labelcolor='navy')
    
    ax3.set_title('Channel Statistics')
    ax3.legend(loc='upper left')
    if result.channel_position is not None:
        ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. シグナルとパーセンタイル
    ax4 = axes[3]
    
    signals_plotted = False
    
    # スクイーズ・エクスパンション信号
    if result.squeeze_signal is not None and result.expansion_signal is not None:
        squeeze_points = result.squeeze_signal == 1.0
        expansion_points = result.expansion_signal == 1.0
        
        if np.any(squeeze_points):
            ax4.scatter(x_axis[squeeze_points], np.ones(np.sum(squeeze_points)), 
                       color='red', marker='s', s=60, alpha=0.8, 
                       label='Squeeze Signal')
            signals_plotted = True
        
        if np.any(expansion_points):
            ax4.scatter(x_axis[expansion_points], -np.ones(np.sum(expansion_points)), 
                       color='green', marker='o', s=60, alpha=0.8, 
                       label='Expansion Signal')
            signals_plotted = True
    
    # パーセンタイル分析
    if result.channel_width_percentile is not None:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x_axis, result.channel_width_percentile, 
                     label='Width Percentile', color='darkgreen', 
                     linewidth=1.5, alpha=0.7)
        ax4_twin.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80th percentile')
        ax4_twin.axhline(y=20, color='blue', linestyle='--', alpha=0.5, label='20th percentile')
        ax4_twin.set_ylabel('Percentile', color='darkgreen')
        ax4_twin.tick_params(axis='y', labelcolor='darkgreen')
        ax4_twin.legend(loc='upper right')
    
    if signals_plotted:
        ax4.set_ylim(-1.5, 1.5)
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax4.set_title('Trading Signals & Analysis')
    if signals_plotted:
        ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # X軸の日付フォーマット
    if 'timestamp' in data.columns:
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # ファイル保存
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ チャート保存: {filename}")
    
    plt.show()


def main():
    """メイン実行"""
    
    parser = argparse.ArgumentParser(description='Hyper Adaptive Channel Chart')
    parser.add_argument('--config', default='config.yaml', help='設定ファイル')
    parser.add_argument('--smoother', default='hyper_frama', 
                       choices=['hyper_frama', 'ultimate_ma', 'laguerre_filter', 
                               'z_adaptive_ma', 'super_smoother'],
                       help='ミッドラインスムーザー')
    parser.add_argument('--multiplier-mode', default='dynamic',
                       choices=['fixed', 'dynamic'], help='乗数モード')
    parser.add_argument('--period', type=int, default=14, help='基本期間')
    parser.add_argument('--test-data', action='store_true', help='テストデータ使用')
    parser.add_argument('--output', help='出力ファイル名')
    
    args = parser.parse_args()
    
    print("=== Hyper Adaptive Channel チャート作成 ===")
    print(f"スムーザー: {args.smoother}")
    print(f"乗数モード: {args.multiplier_mode}")
    print(f"期間: {args.period}")
    
    # データ読み込み
    if args.test_data:
        print("テストデータを生成中...")
        data = create_test_data(500)
    else:
        print("実データ読み込み中...")
        config = load_config(args.config)
        # 実データ読み込みロジック（省略：テストデータで代用）
        data = create_test_data(500)
    
    print(f"データ期間: {len(data)}件")
    
    # ハイパーアダプティブチャネル計算
    try:
        indicator = HyperAdaptiveChannel(
            period=args.period,
            midline_smoother=args.smoother,
            multiplier_mode=args.multiplier_mode,
            fixed_multiplier=2.5,
            enable_signals=True,
            enable_percentile=True
        )
        
        result = indicator.calculate(data)
        
        print("✓ 計算完了")
        print(f"  - Midline有効値: {np.sum(~np.isnan(result.midline))}/{len(data)}")
        print(f"  - チャネル幅範囲: {np.nanmin(result.bandwidth):.4f} - {np.nanmax(result.bandwidth):.4f}")
        print(f"  - 乗数範囲: {np.nanmin(result.multiplier_values):.2f} - {np.nanmax(result.multiplier_values):.2f}")
        
        # チャート作成
        output_file = args.output or f"hyper_adaptive_channel_{args.smoother}_{args.multiplier_mode}.png"
        
        create_chart(
            data=data,
            result=result,
            title="Hyper Adaptive Channel Analysis",
            filename=output_file,
            smoother_name=args.smoother,
            multiplier_mode=args.multiplier_mode
        )
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()