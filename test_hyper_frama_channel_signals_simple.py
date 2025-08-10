#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperFRAMAChannelシグナルのシンプルテスト（軽量版）
"""

import numpy as np
import pandas as pd
import sys
import os

# パスの追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from signals.implementations.hyper_frama_channel.breakout_entry import HyperFRAMAChannelBreakoutEntrySignal
from signals.implementations.hyper_frama_channel.breakout_exit import HyperFRAMAChannelBreakoutExitSignal


def generate_sample_data():
    """軽量サンプルデータの生成"""
    np.random.seed(42)
    n_days = 100  # データ量を削減
    
    # 時系列インデックス作成
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='1D')
    
    # シンプルなトレンドデータ生成
    trend = np.linspace(100, 120, n_days)
    noise = np.random.normal(0, 1, n_days)
    
    close_base = trend + noise
    
    # OHLCV データ生成
    high = close_base + np.random.uniform(0.2, 1, n_days)
    low = close_base - np.random.uniform(0.2, 1, n_days)
    open_prices = np.roll(close_base, 1)
    open_prices[0] = close_base[0]
    
    volume = np.random.uniform(100000, 500000, n_days)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_base,
        'volume': volume
    }, index=dates)
    
    return data


def test_hyper_frama_channel_signals_simple():
    """HyperFRAMAChannelシグナルのシンプルテスト"""
    
    print("=== HyperFRAMAChannelシグナル シンプルテスト ===")
    
    # テストデータの生成
    data = generate_sample_data()
    print(f"\n使用データ: {len(data)}行")
    
    try:
        print("\n1. 軽量版エントリーシグナル初期化中...")
        # エントリーシグナルの初期化（軽量設定）
        entry_signal = HyperFRAMAChannelBreakoutEntrySignal(
            band_lookback=1,
            # 短期FRAMA設定（軽量）
            short_frama_period=8,
            short_frama_src_type='close',  # シンプルなソース
            short_frama_fc=1,
            short_frama_sc=100,
            short_frama_alpha_multiplier=0.5,
            short_frama_enable_indicator_adaptation=False,  # 動的適応無効
            short_frama_smoothing_mode='none',  # 平滑化なし
            # 長期FRAMA設定（軽量）
            long_frama_period=22,
            long_frama_src_type='close',  # シンプルなソース
            long_frama_fc=1,
            long_frama_sc=200,
            long_frama_alpha_multiplier=0.5,
            long_frama_enable_indicator_adaptation=False,  # 動的適応無効
            long_frama_smoothing_mode='none',  # 平滑化なし
            # チャネル設定（軽量）
            channel_period=14,
            channel_multiplier_mode="fixed",  # 固定乗数で軽量化
            channel_fixed_multiplier=2.0,
            channel_src_type="close",
            # チャネルHyperFRAMA設定（軽量）
            channel_hyper_frama_period=16,
            channel_hyper_frama_src_type='close',  # シンプルなソース
            channel_hyper_frama_fc=1,
            channel_hyper_frama_sc=198,
            channel_hyper_frama_enable_indicator_adaptation=False,  # 動的適応無効で軽量化
            # チャネルATR設定（軽量）
            channel_x_atr_period=12.0,
            channel_x_atr_smoother_type='sma',  # SMAで軽量化
            channel_x_atr_period_mode='fixed',  # 固定期間で軽量化
            channel_x_atr_enable_kalman=False,  # カルマンフィルター無効
            # チャネルHyperER設定（軽量）
            channel_hyper_er_period=8,
            channel_hyper_er_midline_period=50,  # 期間短縮
            channel_hyper_er_use_kalman_filter=False,  # カルマンフィルター無効
            channel_hyper_er_use_roofing_filter=False,  # ルーフィングフィルター無効
            channel_hyper_er_use_smoothing=False,  # 平滑化無効
            channel_hyper_er_use_dynamic_period=False  # 動的期間無効
        )
        print("✓ エントリーシグナル（軽量版）初期化完了")
        
        print("\n2. 軽量版エグジットシグナル初期化中...")
        # エグジットシグナルの初期化（モード1: FRAMA交差、軽量）
        exit_signal_mode1 = HyperFRAMAChannelBreakoutExitSignal(
            exit_mode=1,
            src_type='close',
            # 短期FRAMA設定（軽量）
            short_frama_period=8,
            short_frama_src_type='close',
            short_frama_fc=1,
            short_frama_sc=100,
            short_frama_alpha_multiplier=0.5,
            short_frama_enable_indicator_adaptation=False,
            short_frama_smoothing_mode='none',
            # 長期FRAMA設定（軽量）
            long_frama_period=22,
            long_frama_src_type='close',
            long_frama_fc=1,
            long_frama_sc=200,
            long_frama_alpha_multiplier=0.5,
            long_frama_enable_indicator_adaptation=False,
            long_frama_smoothing_mode='none'
        )
        print("✓ エグジットシグナル（軽量版）初期化完了")
        
        print("\n3. シグナル計算中...")
        entry_signals = entry_signal.generate(data)
        exit_signals_mode1 = exit_signal_mode1.generate(data)
        
        print(f"✓ エントリーシグナル計算完了: {len(entry_signals)}個")
        print(f"✓ エグジットシグナル計算完了: {len(exit_signals_mode1)}個")
        
        print("\n4. 付加情報取得中...")
        midline, upper_band, lower_band = entry_signal.get_channel_values(data)
        short_frama, long_frama = entry_signal.get_frama_values(data)
        
        print(f"✓ チャネル値取得完了: {len(upper_band)}個")
        print(f"✓ FRAMA値取得完了: {len(short_frama)}個")
        
        # シグナル統計
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        long_exits = np.sum(exit_signals_mode1 == 1)
        short_exits = np.sum(exit_signals_mode1 == -1)
        
        print(f"\n=== シグナル統計 ===")
        print(f"エントリー - ロング: {long_entries}回, ショート: {short_entries}回")
        print(f"エグジット - ロング: {long_exits}回, ショート: {short_exits}回")
        
        # データ品質チェック
        print(f"\n=== データ品質チェック ===")
        print(f"チャネルミッドライン - 有効値: {np.sum(~np.isnan(midline))}, NaN: {np.sum(np.isnan(midline))}")
        print(f"チャネル上限バンド - 有効値: {np.sum(~np.isnan(upper_band))}, NaN: {np.sum(np.isnan(upper_band))}")
        print(f"チャネル下限バンド - 有効値: {np.sum(~np.isnan(lower_band))}, NaN: {np.sum(np.isnan(lower_band))}")
        print(f"短期FRAMA - 有効値: {np.sum(~np.isnan(short_frama))}, NaN: {np.sum(np.isnan(short_frama))}")
        print(f"長期FRAMA - 有効値: {np.sum(~np.isnan(long_frama))}, NaN: {np.sum(np.isnan(long_frama))}")
        
        # パラメータ完全設定テスト
        print(f"\n=== パラメータ設定テスト ===")
        print(f"エントリーシグナル名: {entry_signal.name}")
        print(f"エグジットシグナル名: {exit_signal_mode1.name}")
        print(f"エントリーシグナルパラメータ数: {len(entry_signal._params)}")
        print(f"エグジットシグナルパラメータ数: {len(exit_signal_mode1._params)}")
        
        print("\n✓ 軽量版テスト完了!")
        
    except Exception as e:
        print(f"✗ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hyper_frama_channel_signals_simple()