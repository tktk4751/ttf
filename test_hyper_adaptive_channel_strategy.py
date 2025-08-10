#!/usr/bin/env python3
"""
Hyper Adaptive Channel ストラテジーテスト

ハイパーアダプティブチャネルストラテジーの動作テスト
"""

import numpy as np
import pandas as pd
import sys
import time

sys.path.append('.')

def test_strategy_basic():
    """基本ストラテジーテスト"""
    
    print("=== Hyper Adaptive Channel ストラテジー基本テスト ===")
    
    try:
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelStrategy, HyperAdaptiveChannelSignalGenerator
        
        # テストデータ作成
        np.random.seed(42)
        n = 200
        
        # トレンドとノイズのある価格データ
        base_trend = np.linspace(100, 120, n)
        noise = np.cumsum(np.random.randn(n) * 0.5)
        close_prices = base_trend + noise
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': np.roll(close_prices, 1),
            'high': close_prices + np.abs(np.random.randn(n)) * 0.3,
            'low': close_prices - np.abs(np.random.randn(n)) * 0.3,
            'close': close_prices,
            'volume': np.random.randint(1000, 5000, n)
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        print(f"テストデータ作成: {len(data)}件")
        
        # ストラテジーテスト
        smoothers = ['hyper_frama', 'ultimate_ma', 'laguerre_filter', 'super_smoother', 'z_adaptive_ma']
        
        for smoother in smoothers:
            print(f"\n--- {smoother} スムーザーテスト ---")
            
            try:
                # ストラテジー作成
                strategy = HyperAdaptiveChannelStrategy(
                    period=14,
                    midline_smoother=smoother,
                    multiplier_mode='dynamic',
                    fixed_multiplier=2.0
                )
                
                # エントリーシグナル生成
                start_time = time.time()
                entry_signals = strategy.generate_entry(data)
                entry_time = time.time() - start_time
                
                # 統計
                long_count = np.sum(entry_signals == 1)
                short_count = np.sum(entry_signals == -1)
                total_signals = long_count + short_count
                signal_rate = total_signals / len(data) * 100
                
                print(f"✓ {smoother} ストラテジー作成成功")
                print(f"  - エントリー計算時間: {entry_time:.3f}秒")
                print(f"  - ロングエントリー: {long_count}回")
                print(f"  - ショートエントリー: {short_count}回")
                print(f"  - エントリー率: {signal_rate:.1f}%")
                
                # エグジットテスト（最後の10ポジションで）
                exit_tests = 0
                for i in range(max(0, len(data)-10), len(data)):
                    if entry_signals[i] != 0:
                        exit_signal = strategy.generate_exit(data, entry_signals[i], i)
                        if exit_signal:
                            exit_tests += 1
                
                print(f"  - エグジットテスト: {exit_tests}/10回")
                
                # バンド値取得
                midline, upper, lower = strategy.signal_generator.get_band_values(data)
                if len(midline) > 0:
                    band_width = np.nanmean(upper - lower)
                    print(f"  - バンド幅平均: {band_width:.4f}")
                
            except Exception as e:
                print(f"✗ {smoother} ストラテジーエラー: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ ストラテジーテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_generator_performance():
    """シグナルジェネレーター性能テスト"""
    
    print("\n=== シグナルジェネレーター性能テスト ===")
    
    try:
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelSignalGenerator
        
        # 大きなテストデータ
        np.random.seed(42)
        n = 1000
        
        base_trend = np.linspace(100, 150, n)
        noise = np.cumsum(np.random.randn(n) * 0.3)
        close_prices = base_trend + noise
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='30T'),
            'open': np.roll(close_prices, 1),
            'high': close_prices + np.abs(np.random.randn(n)) * 0.2,
            'low': close_prices - np.abs(np.random.randn(n)) * 0.2,
            'close': close_prices,
            'volume': np.random.randint(1000, 3000, n)
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        print(f"大規模テストデータ: {len(data)}件")
        
        # シグナルジェネレーター作成
        signal_gen = HyperAdaptiveChannelSignalGenerator(
            period=12,
            midline_smoother='hyper_frama',
            multiplier_mode='dynamic'
        )
        
        # 初回計算
        start_time = time.time()
        signals1 = signal_gen.get_entry_signals(data)
        first_time = time.time() - start_time
        
        # キャッシュされた計算
        start_time = time.time()
        signals2 = signal_gen.get_entry_signals(data)
        cached_time = time.time() - start_time
        
        # 結果比較
        are_same = np.array_equal(signals1, signals2)
        speedup = first_time / max(cached_time, 0.001)
        
        print(f"初回計算時間: {first_time:.3f}秒")
        print(f"キャッシュ計算時間: {cached_time:.4f}秒")
        print(f"高速化率: {speedup:.1f}x")
        print(f"結果一致: {'✓' if are_same else '✗'}")
        
        # 統計
        long_count = np.sum(signals1 == 1)
        short_count = np.sum(signals1 == -1)
        total_signals = long_count + short_count
        
        print(f"総シグナル数: {total_signals} ({total_signals/len(data)*100:.1f}%)")
        print(f"ロング: {long_count}, ショート: {short_count}")
        
        # 追加データ取得テスト
        print("\n--- 追加データ取得テスト ---")
        
        try:
            # HyperER
            hyper_er = signal_gen.get_hyper_er(data)
            print(f"✓ HyperER取得: {len(hyper_er)}件")
            
            # 動的乗数
            dynamic_mult = signal_gen.get_dynamic_multiplier(data)
            print(f"✓ 動的乗数取得: {len(dynamic_mult)}件")
            
            # X_ATR
            x_atr = signal_gen.get_x_atr(data)
            print(f"✓ X_ATR取得: {len(x_atr)}件")
            
        except Exception as e:
            print(f"✗ 追加データ取得エラー: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能テストエラー: {e}")
        return False


def test_multiplier_modes():
    """乗数モード比較テスト"""
    
    print("\n=== 乗数モード比較テスト ===")
    
    try:
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelStrategy
        
        # テストデータ
        np.random.seed(42)
        n = 300
        
        # ボラティリティの変動するデータ
        base_prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        volatility = 0.2 + 0.3 * np.sin(np.linspace(0, 4*np.pi, n))
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='2H'),
            'open': np.roll(base_prices, 1),
            'high': base_prices + volatility * np.abs(np.random.randn(n)),
            'low': base_prices - volatility * np.abs(np.random.randn(n)),
            'close': base_prices,
            'volume': np.random.randint(1000, 4000, n)
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        print(f"変動ボラティリティデータ: {len(data)}件")
        
        # 固定乗数 vs 動的乗数
        modes = [
            ('fixed', 2.0),
            ('fixed', 3.0),
            ('dynamic', 2.5)
        ]
        
        results = {}
        
        for mode, mult in modes:
            mode_name = f"{mode}_{mult}"
            print(f"\n--- {mode_name} モードテスト ---")
            
            try:
                strategy = HyperAdaptiveChannelStrategy(
                    period=15,
                    midline_smoother='super_smoother',
                    multiplier_mode=mode,
                    fixed_multiplier=mult
                )
                
                # シグナル生成
                signals = strategy.generate_entry(data)
                
                # 統計
                long_count = np.sum(signals == 1)
                short_count = np.sum(signals == -1)
                total_signals = long_count + short_count
                signal_rate = total_signals / len(data) * 100
                
                print(f"✓ {mode_name} モード完了")
                print(f"  - 総シグナル: {total_signals} ({signal_rate:.1f}%)")
                print(f"  - ロング: {long_count}, ショート: {short_count}")
                
                # バンド分析
                midline, upper, lower = strategy.signal_generator.get_band_values(data)
                if len(midline) > 0:
                    band_width = upper - lower
                    avg_width = np.nanmean(band_width)
                    width_std = np.nanstd(band_width)
                    print(f"  - バンド幅: 平均={avg_width:.4f}, 標準偏差={width_std:.4f}")
                
                results[mode_name] = {
                    'signals': total_signals,
                    'rate': signal_rate,
                    'avg_width': avg_width if len(midline) > 0 else 0
                }
                
            except Exception as e:
                print(f"✗ {mode_name} モードエラー: {e}")
        
        # 結果比較
        print("\n--- モード比較結果 ---")
        if results:
            print("シグナル生成率順:")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['rate'], reverse=True)
            for name, data in sorted_results:
                print(f"  {name}: {data['rate']:.1f}% ({data['signals']}シグナル, 幅={data['avg_width']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ 乗数モード比較エラー: {e}")
        return False


def test_optimization_params():
    """最適化パラメータテスト"""
    
    print("\n=== 最適化パラメータテスト ===")
    
    try:
        import optuna
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelStrategy
        
        # モックトライアル作成
        class MockTrial:
            def __init__(self):
                self.params = {}
                
            def suggest_int(self, name, low, high):
                val = np.random.randint(low, high + 1)
                self.params[name] = val
                return val
                
            def suggest_float(self, name, low, high, step=None):
                if step:
                    n_steps = int((high - low) / step) + 1
                    val = low + step * np.random.randint(0, n_steps)
                else:
                    val = np.random.uniform(low, high)
                self.params[name] = val
                return val
                
            def suggest_categorical(self, name, choices):
                val = np.random.choice(choices)
                self.params[name] = val
                return val
        
        # 最適化パラメータ生成テスト
        trial = MockTrial()
        
        print("最適化パラメータ生成中...")
        params = HyperAdaptiveChannelStrategy.create_optimization_params(trial)
        
        print(f"✓ パラメータ生成完了: {len(params)}個")
        
        # 主要パラメータ表示
        key_params = [
            'period', 'midline_smoother', 'multiplier_mode', 'fixed_multiplier',
            'hyper_frama_period', 'ultimate_ma_ultimate_smoother_period',
            'laguerre_gamma', 'super_smoother_length', 'z_adaptive_fast_period',
            'x_atr_period', 'hyper_er_period'
        ]
        
        print("\n主要パラメータ:")
        for key in key_params:
            if key in params:
                print(f"  {key}: {params[key]}")
        
        # ストラテジーフォーマット変換テスト
        strategy_params = HyperAdaptiveChannelStrategy.convert_params_to_strategy_format(params)
        
        print(f"\n✓ ストラテジーフォーマット変換完了: {len(strategy_params)}個")
        
        # ストラテジー作成テスト
        try:
            strategy = HyperAdaptiveChannelStrategy(**strategy_params)
            print("✓ ストラテジー作成成功")
        except Exception as e:
            print(f"✗ ストラテジー作成失敗: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 最適化パラメータテストエラー: {e}")
        return False


def main():
    """メイン実行"""
    
    print("Hyper Adaptive Channel ストラテジー テスト開始")
    print("="*70)
    
    # 基本テスト
    test_strategy_basic()
    
    # 性能テスト
    test_signal_generator_performance()
    
    # 乗数モード比較
    test_multiplier_modes()
    
    # 最適化パラメータテスト
    test_optimization_params()
    
    print("\n" + "="*70)
    print("ストラテジーテスト完了")


if __name__ == "__main__":
    main()