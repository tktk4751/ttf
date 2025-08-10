#!/usr/bin/env python3
"""
Hyper Adaptive Channel シグナルテスト

ハイパーアダプティブチャネルシグナルの動作テスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('.')

def test_basic_signal_functionality():
    """基本シグナル機能テスト"""
    
    print("=== Hyper Adaptive Channel シグナル基本テスト ===")
    
    try:
        from signals.implementations.hyper_adaptive_channel import HyperAdaptiveChannelBreakoutEntrySignal
        
        # テストデータ作成
        np.random.seed(42)
        n = 300
        
        # トレンドのある価格データ
        base_price = 100
        trend = np.linspace(0, 15, n)
        volatility = 1 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 50)
        
        noise = np.random.randn(n) * 0.8 * volatility
        close_prices = base_price + trend + np.cumsum(noise * 0.5)
        
        # ランダムなスパイクを追加（ブレイクアウト模擬）
        spike_indices = np.random.choice(range(50, n-50), 5, replace=False)
        for idx in spike_indices:
            close_prices[idx:idx+3] += np.random.choice([3, -3]) * volatility[idx]
        
        high_prices = close_prices + np.abs(np.random.randn(n)) * 0.3 * volatility
        low_prices = close_prices - np.abs(np.random.randn(n)) * 0.3 * volatility
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.random.randint(1000, 5000, n) * volatility
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='4H'),
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        print(f"テストデータ作成: {len(data)}件")
        
        # 各スムーザーでシグナルテスト
        smoothers = ["hyper_frama", "super_smoother", "laguerre_filter"]
        
        results = {}
        
        for smoother in smoothers:
            print(f"\n--- {smoother} シグナルテスト ---")
            
            try:
                # デフォルトパラメータでテスト
                signal = HyperAdaptiveChannelBreakoutEntrySignal(
                    band_lookback=1,
                    period=14,
                    midline_smoother=smoother,
                    multiplier_mode="dynamic"
                )
                
                # シグナル生成
                signals = signal.generate(data)
                
                # シグナル統計
                long_signals = np.sum(signals == 1)
                short_signals = np.sum(signals == -1)
                no_signals = np.sum(signals == 0)
                
                print(f"✓ {smoother} シグナル生成成功")
                print(f"  - ロングシグナル: {long_signals}回")
                print(f"  - ショートシグナル: {short_signals}回")
                print(f"  - シグナルなし: {no_signals}回")
                print(f"  - 総シグナル率: {(long_signals + short_signals) / len(data) * 100:.1f}%")
                
                # バンド値取得テスト
                midline, upper, lower = signal.get_band_values(data)
                print(f"  - ミッドライン有効値: {np.sum(~np.isnan(midline))}/{len(data)}")
                print(f"  - アッパーバンド有効値: {np.sum(~np.isnan(upper))}/{len(data)}")
                print(f"  - ロワーバンド有効値: {np.sum(~np.isnan(lower))}/{len(data)}")
                
                # ATR・乗数値取得テスト
                atr_values = signal.get_atr_values(data)
                multiplier_values = signal.get_multiplier_values(data)
                print(f"  - ATR範囲: {np.nanmin(atr_values):.4f} - {np.nanmax(atr_values):.4f}")
                print(f"  - 乗数範囲: {np.nanmin(multiplier_values):.2f} - {np.nanmax(multiplier_values):.2f}")
                
                # ER値取得テスト（動的モード時）
                er_values = signal.get_er_values(data)
                if er_values is not None:
                    print(f"  - ER範囲: {np.nanmin(er_values):.3f} - {np.nanmax(er_values):.3f}")
                
                results[smoother] = {
                    'signals': signals,
                    'long_count': long_signals,
                    'short_count': short_signals,
                    'midline': midline,
                    'upper': upper,
                    'lower': lower,
                    'atr': atr_values,
                    'multiplier': multiplier_values
                }
                
            except Exception as e:
                print(f"✗ {smoother} シグナルエラー: {e}")
        
        # 結果を返してチャート作成用に使用
        return data, results
        
    except Exception as e:
        print(f"✗ インポートまたは全体エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_custom_parameters():
    """カスタムパラメーターテスト"""
    
    print("\n=== カスタムパラメーターテスト ===")
    
    try:
        from signals.implementations.hyper_adaptive_channel import HyperAdaptiveChannelBreakoutEntrySignal
        
        # シンプルなテストデータ
        np.random.seed(42)
        n = 200
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='2H'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.01),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.01),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.01),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.01),
            'volume': np.random.randint(1000, 3000, n)
        })
        
        # カスタマイズされたパラメーターでテスト
        signal = HyperAdaptiveChannelBreakoutEntrySignal(
            band_lookback=2,  # 2バー前参照
            period=20,
            midline_smoother="super_smoother",
            multiplier_mode="fixed",
            fixed_multiplier=3.0,
            
            # SuperSmoother カスタマイズ
            super_smoother_length=16,
            super_smoother_num_poles=3,
            super_smoother_src_type="hlc3",
            
            # X_ATR カスタマイズ
            x_atr_period=18.0,
            x_atr_tr_method="atr",
            x_atr_enable_kalman=True,
            
            # シグナル設定
            enable_signals=True,
            enable_percentile=True
        )
        
        # シグナル生成
        signals = signal.generate(data)
        
        # 結果
        long_count = np.sum(signals == 1)
        short_count = np.sum(signals == -1)
        
        print("✓ カスタムパラメーターシグナル生成成功")
        print(f"  - ロングシグナル: {long_count}回")
        print(f"  - ショートシグナル: {short_count}回")
        print(f"  - バンド参照期間: 2バー前")
        print(f"  - 固定乗数: 3.0")
        print(f"  - SuperSmoother 3極モード")
        
        # バンド値テスト
        midline, upper, lower = signal.get_band_values(data)
        print(f"  - バンド幅平均: {np.nanmean(upper - lower):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ カスタムパラメーターテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """パフォーマンステスト"""
    
    print("\n=== パフォーマンステスト ===")
    
    try:
        from signals.implementations.hyper_adaptive_channel import HyperAdaptiveChannelBreakoutEntrySignal
        import time
        
        # 大量データ生成
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.005),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.005),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.005),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.005),
            'volume': np.random.randint(1000, 5000, n)
        })
        
        signal = HyperAdaptiveChannelBreakoutEntrySignal(
            period=14,
            midline_smoother="super_smoother",
            multiplier_mode="dynamic"
        )
        
        # 初回計算時間
        start_time = time.time()
        signals1 = signal.generate(data)
        first_time = time.time() - start_time
        
        # キャッシュからの計算時間
        start_time = time.time()
        signals2 = signal.generate(data)
        cached_time = time.time() - start_time
        
        print(f"✓ パフォーマンステスト完了")
        print(f"  - データ数: {n}件")
        print(f"  - 初回計算時間: {first_time:.3f}秒 ({n/first_time:.0f} データ/秒)")
        print(f"  - キャッシュ計算時間: {cached_time:.6f}秒")
        print(f"  - キャッシュ高速化率: {first_time/cached_time:.0f}x")
        print(f"  - 結果一致: {np.array_equal(signals1, signals2)}")
        
        return True
        
    except Exception as e:
        print(f"✗ パフォーマンステストエラー: {e}")
        return False


def create_signal_chart(data, results, filename="hyper_adaptive_channel_signals_test.png"):
    """シグナルチャート作成"""
    
    if data is None or not results:
        print("チャート作成用データがありません")
        return
    
    try:
        print(f"\n--- シグナルチャート作成中 ---")
        
        # 最初のスムーザー結果を使用
        smoother_name = list(results.keys())[0]
        result = results[smoother_name]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 価格とチャネル
        ax1.plot(data['close'].values, label='Close Price', color='black', linewidth=1.5)
        ax1.plot(result['midline'], label=f'Midline ({smoother_name})', color='blue', linewidth=1.5)
        ax1.plot(result['upper'], label='Upper Band', color='red', linewidth=1, alpha=0.8)
        ax1.plot(result['lower'], label='Lower Band', color='green', linewidth=1, alpha=0.8)
        
        # チャネル塗りつぶし
        valid_mask = ~(np.isnan(result['upper']) | np.isnan(result['lower']))
        if np.any(valid_mask):
            ax1.fill_between(range(len(result['upper'])), result['upper'], result['lower'],
                           alpha=0.1, color='gray')
        
        # シグナルプロット
        signals = result['signals']
        long_signals = signals == 1
        short_signals = signals == -1
        
        if np.any(long_signals):
            ax1.scatter(np.where(long_signals)[0], data['close'].values[long_signals],
                       color='green', marker='^', s=100, alpha=0.8, label='ロングシグナル', zorder=5)
        
        if np.any(short_signals):
            ax1.scatter(np.where(short_signals)[0], data['close'].values[short_signals],
                       color='red', marker='v', s=100, alpha=0.8, label='ショートシグナル', zorder=5)
        
        ax1.set_title(f'Hyper Adaptive Channel Signals - {smoother_name.title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ATRと乗数
        ax2_twin = ax2.twinx()
        ax2.plot(result['atr'], label='X-ATR', color='orange', linewidth=1.5)
        ax2_twin.plot(result['multiplier'], label='Dynamic Multiplier', color='purple', linewidth=1.5)
        
        ax2.set_ylabel('ATR', color='orange')
        ax2_twin.set_ylabel('Multiplier', color='purple')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ シグナルチャート保存: {filename}")
        
    except Exception as e:
        print(f"✗ チャート作成エラー: {e}")


def main():
    """メイン実行"""
    
    print("Hyper Adaptive Channel シグナル テスト開始")
    print("="*60)
    
    # 基本機能テスト
    data, results = test_basic_signal_functionality()
    
    # カスタムパラメーターテスト
    test_custom_parameters()
    
    # パフォーマンステスト
    test_performance()
    
    # チャート作成
    if data is not None and results:
        create_signal_chart(data, results)
    
    print("\n" + "="*60)
    print("シグナルテスト完了")


if __name__ == "__main__":
    main()