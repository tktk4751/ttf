#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperFRAMAChannelシグナルの完全パラメーター版テスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# パスの追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from signals.implementations.hyper_frama_channel.breakout_entry import HyperFRAMAChannelBreakoutEntrySignal
from signals.implementations.hyper_frama_channel.breakout_exit import HyperFRAMAChannelBreakoutExitSignal


def generate_sample_data():
    """サンプルデータの生成"""
    np.random.seed(42)
    n_days = 300
    
    # 時系列インデックス作成
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='4H')
    
    # トレンドのある価格データ生成
    trend = np.linspace(100, 150, n_days)
    noise = np.random.normal(0, 2, n_days)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 50)
    
    close_base = trend + seasonal + noise
    
    # OHLCV データ生成
    high = close_base + np.random.uniform(0.5, 3, n_days)
    low = close_base - np.random.uniform(0.5, 3, n_days)
    open_prices = np.roll(close_base, 1)  # 前日の終値を開始価格とする
    open_prices[0] = close_base[0]
    
    volume = np.random.uniform(1000000, 5000000, n_days)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_base,
        'volume': volume
    }, index=dates)
    
    return data


def test_hyper_frama_channel_signals_full_params():
    """HyperFRAMAChannelシグナルの完全パラメーター版テスト"""
    
    print("=== HyperFRAMAChannelシグナル 完全パラメーター版 テスト ===")
    
    # テストデータの生成
    data = generate_sample_data()
    print(f"\n使用データ: {len(data)}行")
    
    try:
        # エントリーシグナルの初期化（主要パラメーターのみ設定）
        entry_signal = HyperFRAMAChannelBreakoutEntrySignal(
            band_lookback=1,
            # 短期FRAMA設定
            short_frama_period=8,
            short_frama_src_type='hlc3',
            short_frama_fc=1,
            short_frama_sc=100,
            short_frama_alpha_multiplier=0.5,
            short_frama_enable_indicator_adaptation=False,  # 動的適応なし
            short_frama_smoothing_mode='none',  # 平滑化なし
            # 長期FRAMA設定
            long_frama_period=22,
            long_frama_src_type='hlc3',
            long_frama_fc=1,
            long_frama_sc=200,
            long_frama_alpha_multiplier=0.5,
            long_frama_enable_indicator_adaptation=False,  # 動的適応なし
            long_frama_smoothing_mode='none',  # 平滑化なし
            # チャネル設定
            channel_period=14,
            channel_multiplier_mode="dynamic",
            channel_fixed_multiplier=2.0,
            channel_src_type="hlc3",
            # チャネルHyperFRAMA設定
            channel_hyper_frama_period=16,
            channel_hyper_frama_src_type='hl2',
            channel_hyper_frama_fc=1,
            channel_hyper_frama_sc=198,
            channel_hyper_frama_enable_indicator_adaptation=True,  # チャネル側は動的適応有効
            channel_hyper_frama_adaptation_indicator='hyper_er',
            # チャネルATR設定（デフォルト値使用）
            channel_x_atr_period=12.0,
            channel_x_atr_smoother_type='frama',
            channel_x_atr_period_mode='dynamic',
            # チャネルHyperER設定（デフォルト値使用）
            channel_hyper_er_period=8,
            channel_hyper_er_midline_period=100,
            channel_hyper_er_use_kalman_filter=True,
            channel_hyper_er_kalman_filter_type='simple',
            channel_hyper_er_use_roofing_filter=True,
            channel_hyper_er_detector_type='dft_dominant'
        )
        print("✓ エントリーシグナル（完全パラメーター版）初期化完了")
        
        # エグジットシグナルの初期化（モード1: FRAMA交差）
        exit_signal_mode1 = HyperFRAMAChannelBreakoutExitSignal(
            exit_mode=1,  # FRAMA交差
            src_type='hlc3',
            # 短期FRAMA設定
            short_frama_period=8,
            short_frama_src_type='hlc3',
            short_frama_fc=1,
            short_frama_sc=100,
            short_frama_alpha_multiplier=0.5,
            short_frama_enable_indicator_adaptation=False,  # 動的適応なし
            short_frama_smoothing_mode='none',  # 平滑化なし
            # 長期FRAMA設定
            long_frama_period=22,
            long_frama_src_type='hlc3',
            long_frama_fc=1,
            long_frama_sc=200,
            long_frama_alpha_multiplier=0.5,
            long_frama_enable_indicator_adaptation=False,  # 動的適応なし
            long_frama_smoothing_mode='none'  # 平滑化なし
        )
        print("✓ エグジットシグナル（モード1、完全パラメーター版）初期化完了")
        
        # エグジットシグナルの初期化（モード2: 価格vs長期FRAMA）
        exit_signal_mode2 = HyperFRAMAChannelBreakoutExitSignal(
            exit_mode=2,  # 価格vs長期FRAMA
            src_type='hlc3',
            # 短期FRAMA設定
            short_frama_period=8,
            short_frama_src_type='hlc3',
            short_frama_fc=1,
            short_frama_sc=100,
            short_frama_alpha_multiplier=0.5,
            short_frama_enable_indicator_adaptation=False,  # 動的適応なし
            short_frama_smoothing_mode='none',  # 平滑化なし
            # 長期FRAMA設定
            long_frama_period=22,
            long_frama_src_type='hlc3',
            long_frama_fc=1,
            long_frama_sc=200,
            long_frama_alpha_multiplier=0.5,
            long_frama_enable_indicator_adaptation=False,  # 動的適応なし
            long_frama_smoothing_mode='none'  # 平滑化なし
        )
        print("✓ エグジットシグナル（モード2、完全パラメーター版）初期化完了")
        
        # シグナル計算
        print("\nシグナル計算中...")
        entry_signals = entry_signal.generate(data)
        exit_signals_mode1 = exit_signal_mode1.generate(data)
        exit_signals_mode2 = exit_signal_mode2.generate(data)
        
        print(f"✓ エントリーシグナル計算完了: {len(entry_signals)}個")
        print(f"✓ エグジットシグナル（モード1）計算完了: {len(exit_signals_mode1)}個")
        print(f"✓ エグジットシグナル（モード2）計算完了: {len(exit_signals_mode2)}個")
        
        # チャネル値とFRAMA値を取得
        midline, upper_band, lower_band = entry_signal.get_channel_values(data)
        short_frama, long_frama = entry_signal.get_frama_values(data)
        
        print(f"✓ チャネル値取得完了: {len(upper_band)}個")
        print(f"✓ FRAMA値取得完了: {len(short_frama)}個")
        
        # シグナル統計
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        long_exits_mode1 = np.sum(exit_signals_mode1 == 1)
        short_exits_mode1 = np.sum(exit_signals_mode1 == -1)
        long_exits_mode2 = np.sum(exit_signals_mode2 == 1)
        short_exits_mode2 = np.sum(exit_signals_mode2 == -1)
        
        print(f"\n=== シグナル統計 ===")
        print(f"エントリー - ロング: {long_entries}回, ショート: {short_entries}回")
        print(f"エグジット（モード1） - ロング: {long_exits_mode1}回, ショート: {short_exits_mode1}回")
        print(f"エグジット（モード2） - ロング: {long_exits_mode2}回, ショート: {short_exits_mode2}回")
        
        # シンプルなグラフ作成
        create_simple_chart(data, entry_signals, exit_signals_mode1, exit_signals_mode2,
                           midline, upper_band, lower_band, short_frama, long_frama)
        
        print("\n✓ 完全パラメーター版テスト完了!")
        
    except Exception as e:
        print(f"✗ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


def create_simple_chart(data, entry_signals, exit_signals_mode1, exit_signals_mode2,
                       midline, upper_band, lower_band, short_frama, long_frama):
    """シンプルなシグナルチャートの作成"""
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle('HyperFRAMAChannel Breakout Signals - Full Parameters Test', fontsize=16)
        
        dates = data.index
        close = data['close'].values
        
        # メインチャート
        ax.plot(dates, close, 'k-', linewidth=1, label='Close Price', alpha=0.8)
        ax.plot(dates, midline, 'b-', linewidth=1, label='HyperFRAMA Channel Midline', alpha=0.7)
        ax.plot(dates, upper_band, 'r--', linewidth=1, label='Upper Band', alpha=0.7)
        ax.plot(dates, lower_band, 'g--', linewidth=1, label='Lower Band', alpha=0.7)
        ax.plot(dates, short_frama, 'orange', linewidth=1, label='Short FRAMA(8)', alpha=0.8)
        ax.plot(dates, long_frama, 'purple', linewidth=1, label='Long FRAMA(22)', alpha=0.8)
        
        # エントリーシグナル
        long_entry_mask = entry_signals == 1
        short_entry_mask = entry_signals == -1
        
        if np.any(long_entry_mask):
            ax.scatter(dates[long_entry_mask], close[long_entry_mask], 
                       color='blue', marker='^', s=80, label='Long Entry', alpha=0.8)
        
        if np.any(short_entry_mask):
            ax.scatter(dates[short_entry_mask], close[short_entry_mask], 
                       color='red', marker='v', s=80, label='Short Entry', alpha=0.8)
        
        # エグジットシグナル（両モード）
        long_exit_mask1 = exit_signals_mode1 == 1
        short_exit_mask1 = exit_signals_mode1 == -1
        long_exit_mask2 = exit_signals_mode2 == 1
        short_exit_mask2 = exit_signals_mode2 == -1
        
        if np.any(long_exit_mask1):
            ax.scatter(dates[long_exit_mask1], close[long_exit_mask1], 
                       color='darkblue', marker='s', s=60, label='Long Exit (Mode1)', alpha=0.8)
        
        if np.any(short_exit_mask1):
            ax.scatter(dates[short_exit_mask1], close[short_exit_mask1], 
                       color='darkred', marker='s', s=60, label='Short Exit (Mode1)', alpha=0.8)
        
        if np.any(long_exit_mask2):
            ax.scatter(dates[long_exit_mask2], close[long_exit_mask2], 
                       color='navy', marker='D', s=50, label='Long Exit (Mode2)', alpha=0.6)
        
        if np.any(short_exit_mask2):
            ax.scatter(dates[short_exit_mask2], close[short_exit_mask2], 
                       color='maroon', marker='D', s=50, label='Short Exit (Mode2)', alpha=0.6)
        
        ax.set_title('Full Parameters Test - Entry & Exit Signals')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # X軸フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存
        filename = f"hyper_frama_channel_signals_full_params_test.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ チャートを保存しました: {filename}")
        
        # 表示
        plt.show()
        
    except Exception as e:
        print(f"チャート作成エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hyper_frama_channel_signals_full_params()