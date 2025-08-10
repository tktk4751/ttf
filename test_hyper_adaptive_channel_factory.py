#!/usr/bin/env python3
"""
Hyper Adaptive Channel シグナルファクトリーテスト

シグナルファクトリー関数の動作テスト
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')

def test_signal_factory():
    """シグナルファクトリーテスト"""
    
    print("=== Hyper Adaptive Channel シグナルファクトリーテスト ===")
    
    try:
        from signals.implementations.hyper_adaptive_channel.signal_factory import (
            create_hyper_frama_breakout_signal,
            create_ultimate_ma_breakout_signal,
            create_laguerre_breakout_signal,
            create_super_smoother_breakout_signal,
            create_z_adaptive_breakout_signal,
            create_custom_atr_breakout_signal,
            create_high_sensitivity_signal,
            create_low_sensitivity_signal,
            create_signal_by_preset
        )
        
        # テストデータ作成
        np.random.seed(42)
        n = 150
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='2H'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.02),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.02),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.02),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.02),
            'volume': np.random.randint(1000, 3000, n)
        })
        
        print(f"テストデータ作成: {len(data)}件")
        
        # 各ファクトリー関数をテスト
        factory_functions = [
            ("HyperFRAMA", lambda: create_hyper_frama_breakout_signal(
                period=12,
                alpha_multiplier=0.7,
                fc_range=(1.0, 10.0)
            )),
            ("UltimateMA", lambda: create_ultimate_ma_breakout_signal(
                period=15,
                ultimate_smoother_period=6.0,
                dynamic_periods=True
            )),
            ("Laguerre", lambda: create_laguerre_breakout_signal(
                period=14,
                gamma=0.8,
                order=6
            )),
            ("SuperSmoother", lambda: create_super_smoother_breakout_signal(
                period=16,
                length=12,
                num_poles=3
            )),
            ("ZAdaptive", lambda: create_z_adaptive_breakout_signal(
                period=14,
                fast_period=3,
                slow_period=100
            )),
            ("CustomATR", lambda: create_custom_atr_breakout_signal(
                atr_period=15.0,
                tr_method="atr",
                enable_kalman=True
            )),
            ("HighSensitivity", lambda: create_high_sensitivity_signal(
                period=8
            )),
            ("LowSensitivity", lambda: create_low_sensitivity_signal(
                period=25
            ))
        ]
        
        results = {}
        
        for name, factory_func in factory_functions:
            print(f"\n--- {name} ファクトリーテスト ---")
            
            try:
                # シグナル作成
                signal = factory_func()
                
                # シグナル生成
                signals = signal.generate(data)
                
                # 統計
                long_count = np.sum(signals == 1)
                short_count = np.sum(signals == -1)
                total_signals = long_count + short_count
                signal_rate = total_signals / len(data) * 100
                
                print(f"✓ {name} ファクトリー作成成功")
                print(f"  - ロングシグナル: {long_count}回")
                print(f"  - ショートシグナル: {short_count}回")
                print(f"  - シグナル率: {signal_rate:.1f}%")
                
                # バンド情報
                midline, upper, lower = signal.get_band_values(data)
                band_width = np.nanmean(upper - lower)
                print(f"  - バンド幅平均: {band_width:.4f}")
                
                results[name] = {
                    'signal_count': total_signals,
                    'signal_rate': signal_rate,
                    'band_width': band_width
                }
                
            except Exception as e:
                print(f"✗ {name} ファクトリーエラー: {e}")
        
        print("\n--- プリセット関数テスト ---")
        
        # プリセット関数テスト
        presets = [
            "hyper_frama",
            "ultimate_ma", 
            "laguerre",
            "super_smoother",
            "high_sensitivity",
            "low_sensitivity"
        ]
        
        for preset in presets:
            try:
                signal = create_signal_by_preset(preset, period=12)
                signals = signal.generate(data)
                total_signals = np.sum((signals == 1) | (signals == -1))
                print(f"✓ プリセット '{preset}': {total_signals}シグナル")
                
            except Exception as e:
                print(f"✗ プリセット '{preset}' エラー: {e}")
        
        # 無効なプリセットテスト
        try:
            create_signal_by_preset("invalid_preset")
            print("✗ 無効プリセットチェック失敗")
        except ValueError as e:
            print(f"✓ 無効プリセットチェック成功: {e}")
        
        # 結果比較
        print("\n--- 結果比較 ---")
        if results:
            print("シグナル生成率順:")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['signal_rate'], reverse=True)
            for name, data in sorted_results:
                print(f"  {name}: {data['signal_rate']:.1f}% ({data['signal_count']}シグナル)")
        
        return True
        
    except Exception as e:
        print(f"✗ ファクトリーテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensitivity_comparison():
    """感度比較テスト"""
    
    print("\n=== 感度比較テスト ===")
    
    try:
        from signals.implementations.hyper_adaptive_channel.signal_factory import (
            create_high_sensitivity_signal,
            create_low_sensitivity_signal
        )
        
        # より変動の激しいテストデータ
        np.random.seed(42)
        n = 200
        
        # トレンド + ノイズ + スパイク
        base_trend = np.linspace(100, 110, n)
        noise = np.cumsum(np.random.randn(n) * 0.3)
        
        # ランダムスパイク追加
        spikes = np.zeros(n)
        spike_indices = np.random.choice(range(20, n-20), 8, replace=False)
        for idx in spike_indices:
            spikes[idx:idx+2] = np.random.choice([2, -2])
        
        close_prices = base_trend + noise + spikes
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': np.roll(close_prices, 1),
            'high': close_prices + np.abs(np.random.randn(n)) * 0.2,
            'low': close_prices - np.abs(np.random.randn(n)) * 0.2,
            'close': close_prices,
            'volume': np.random.randint(1000, 4000, n)
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        print(f"変動テストデータ作成: {len(data)}件")
        
        # 高感度シグナル
        high_sens_signal = create_high_sensitivity_signal(period=8)
        high_sens_signals = high_sens_signal.generate(data)
        
        # 低感度シグナル
        low_sens_signal = create_low_sensitivity_signal(period=20)
        low_sens_signals = low_sens_signal.generate(data)
        
        # 結果比較
        high_long = np.sum(high_sens_signals == 1)
        high_short = np.sum(high_sens_signals == -1)
        high_total = high_long + high_short
        
        low_long = np.sum(low_sens_signals == 1)
        low_short = np.sum(low_sens_signals == -1)
        low_total = low_long + low_short
        
        print(f"\n高感度シグナル:")
        print(f"  - ロング: {high_long}回, ショート: {high_short}回")
        print(f"  - 総シグナル: {high_total}回 ({high_total/len(data)*100:.1f}%)")
        
        print(f"\n低感度シグナル:")
        print(f"  - ロング: {low_long}回, ショート: {low_short}回")
        print(f"  - 総シグナル: {low_total}回 ({low_total/len(data)*100:.1f}%)")
        
        print(f"\n感度比: {high_total/max(low_total, 1):.1f}x")
        
        # バンド幅比較
        _, high_upper, high_lower = high_sens_signal.get_band_values(data)
        _, low_upper, low_lower = low_sens_signal.get_band_values(data)
        
        high_width = np.nanmean(high_upper - high_lower)
        low_width = np.nanmean(low_upper - low_lower)
        
        print(f"\nバンド幅:")
        print(f"  - 高感度: {high_width:.4f}")
        print(f"  - 低感度: {low_width:.4f}")
        print(f"  - 幅比: {low_width/high_width:.1f}x")
        
        print("✓ 感度比較テスト完了")
        
        return True
        
    except Exception as e:
        print(f"✗ 感度比較テストエラー: {e}")
        return False


def main():
    """メイン実行"""
    
    print("Hyper Adaptive Channel シグナルファクトリー テスト開始")
    print("="*70)
    
    # ファクトリーテスト
    test_signal_factory()
    
    # 感度比較テスト
    test_sensitivity_comparison()
    
    print("\n" + "="*70)
    print("ファクトリーテスト完了")


if __name__ == "__main__":
    main()