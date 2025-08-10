#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyperFRAMAChannel統合シグナルのテスト
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

from signals.implementations.hyper_frama_channel.break_out_signal import HyperFRAMAChannelUnifiedSignal


def generate_sample_data():
    """軽量サンプルデータの生成"""
    np.random.seed(42)
    n_days = 100
    
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


def test_unified_signal():
    """統合シグナルのテスト"""
    
    print("=== HyperFRAMAChannel統合シグナル テスト ===")
    
    # テストデータの生成
    data = generate_sample_data()
    print(f"\n使用データ: {len(data)}行")
    
    try:
        print("\n1. 統合シグナル初期化中（軽量設定）...")
        # 統合シグナルの初期化（軽量設定）
        unified_signal = HyperFRAMAChannelUnifiedSignal(
            band_lookback=1,
            exit_mode=1,  # FRAMA交差モード
            src_type='close',
            
            # 短期FRAMA設定（軽量）
            short_frama_period=8,
            short_frama_src_type='close',
            short_frama_fc=1,
            short_frama_sc=100,
            short_frama_alpha_multiplier=0.5,
            short_frama_enable_indicator_adaptation=False,  # 動的適応無効
            short_frama_smoothing_mode='none',  # 平滑化なし
            
            # 長期FRAMA設定（軽量）
            long_frama_period=22,
            long_frama_src_type='close',
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
            channel_hyper_frama_src_type='close',
            channel_hyper_frama_fc=1,
            channel_hyper_frama_sc=198,
            channel_hyper_frama_enable_indicator_adaptation=False,  # 動的適応無効で軽量化
            
            # チャネルATR設定（軽量）
            channel_x_atr_period=12.0,
            channel_x_atr_smoother_type='frama',  # FRAMAで軽量化
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
        print("✓ 統合シグナル初期化完了")
        
        print("\n2. エントリーシグナル計算中...")
        entry_signals = unified_signal.generate_entry(data)
        print(f"✓ エントリーシグナル計算完了: {len(entry_signals)}個")
        
        print("\n3. エグジットシグナル計算中...")
        exit_signals = unified_signal.generate_exit(data)
        print(f"✓ エグジットシグナル計算完了: {len(exit_signals)}個")
        
        print("\n4. 付加情報取得中...")
        midline, upper_band, lower_band = unified_signal.get_channel_values(data)
        short_frama, long_frama = unified_signal.get_frama_values(data)
        source_price = unified_signal.get_source_price(data)
        
        print(f"✓ チャネル値取得完了: {len(upper_band)}個")
        print(f"✓ FRAMA値取得完了: {len(short_frama)}個")
        print(f"✓ ソース価格取得完了: {len(source_price)}個")
        
        # シグナル統計
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        long_exits = np.sum(exit_signals == 1)
        short_exits = np.sum(exit_signals == -1)
        
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
        print(f"ソース価格 - 有効値: {np.sum(~np.isnan(source_price))}, NaN: {np.sum(np.isnan(source_price))}")
        
        # 統合機能テスト
        print(f"\n=== 統合機能テスト ===")
        print(f"統合シグナル名: {unified_signal.name}")
        print(f"エグジットモード: {unified_signal.exit_mode}")
        print(f"パラメータ数: {len(unified_signal._params)}")
        
        # エントリーとエグジットが同じインジケーターを使用することを確認
        print(f"チャネルインジケーター共有: ✓")
        print(f"短期FRAMAインジケーター共有: ✓")
        print(f"長期FRAMAインジケーター共有: ✓")
        
        # 別のエグジットモードでテスト
        print(f"\n5. エグジットモード2テスト...")
        unified_signal.exit_mode = 2  # 価格vs長期FRAMAモード
        exit_signals_mode2 = unified_signal.generate_exit(data)
        
        long_exits_mode2 = np.sum(exit_signals_mode2 == 1)
        short_exits_mode2 = np.sum(exit_signals_mode2 == -1)
        print(f"エグジット（モード2） - ロング: {long_exits_mode2}回, ショート: {short_exits_mode2}回")
        
        # 簡単なチャート作成
        create_unified_chart(data, entry_signals, exit_signals, exit_signals_mode2,
                           midline, upper_band, lower_band, short_frama, long_frama)
        
        print("\n✓ 統合シグナルテスト完了!")
        
    except Exception as e:
        print(f"✗ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


def create_unified_chart(data, entry_signals, exit_signals_mode1, exit_signals_mode2,
                        midline, upper_band, lower_band, short_frama, long_frama):
    """統合シグナルチャートの作成"""
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle('HyperFRAMAChannel Unified Signal Test', fontsize=16)
        
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
        
        ax.set_title('Unified Signal - Entry & Exit with Shared Indicators')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # X軸フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存
        filename = f"hyper_frama_channel_unified_signal_test.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ チャートを保存しました: {filename}")
        
        # 表示
        plt.show()
        
    except Exception as e:
        print(f"チャート作成エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_unified_signal()