#!/usr/bin/env python3
"""
最小限のハイパーアダプティブチャネルストラテジーテスト
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')

def test_minimal_strategy():
    """最小限のストラテジーテスト"""
    
    print("=== 最小限のストラテジーテスト ===")
    
    try:
        # 直接HyperAdaptiveChannelBreakoutEntrySignalをテスト
        from signals.implementations.hyper_adaptive_channel.breakout_entry import HyperAdaptiveChannelBreakoutEntrySignal
        
        # テストデータ作成
        np.random.seed(42)
        n = 50
        
        base_prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        
        data = pd.DataFrame({
            'open': np.roll(base_prices, 1),
            'high': base_prices + np.abs(np.random.randn(n)) * 0.1,
            'low': base_prices - np.abs(np.random.randn(n)) * 0.1,
            'close': base_prices,
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        print(f"テストデータ作成: {len(data)}件")
        
        # 最小限のパラメータでシグナル作成
        signal = HyperAdaptiveChannelBreakoutEntrySignal(
            band_lookback=1,
            period=14,
            midline_smoother='hyper_frama',
            multiplier_mode='fixed',
            fixed_multiplier=2.0
        )
        
        print("✓ シグナル作成成功")
        
        # シグナル生成
        signals = signal.generate(data)
        
        # 統計
        long_count = np.sum(signals == 1)
        short_count = np.sum(signals == -1)
        total_signals = long_count + short_count
        
        print(f"✓ シグナル生成成功")
        print(f"  - ロング: {long_count}回")
        print(f"  - ショート: {short_count}回")
        print(f"  - 総シグナル: {total_signals}回")
        
        # バンド値取得
        midline, upper, lower = signal.get_band_values()
        if len(midline) > 0:
            print(f"  - バンド計算成功: {len(midline)}件")
        
        return True
        
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_with_minimal_signal():
    """最小限のシグナルジェネレーターでストラテジーテスト"""
    
    print("\n=== 最小限ストラテジーテスト ===")
    
    try:
        # 最小限のシグナルジェネレーター作成
        from ...base.signal_generator import BaseSignalGenerator
        from signals.implementations.hyper_adaptive_channel.breakout_entry import HyperAdaptiveChannelBreakoutEntrySignal
        
        class MinimalSignalGenerator(BaseSignalGenerator):
            def __init__(self):
                super().__init__("MinimalSignalGenerator")
                self.signal = HyperAdaptiveChannelBreakoutEntrySignal()
                
            def get_entry_signals(self, data):
                return self.signal.generate(data)
                
            def get_exit_signals(self, data, position, index=-1):
                signals = self.signal.generate(data)
                if index == -1:
                    index = len(data) - 1
                if position == 1:
                    return bool(signals[index] == -1)
                elif position == -1:
                    return bool(signals[index] == 1)
                return False
        
        # 最小限のストラテジー作成
        from ...base.strategy import BaseStrategy
        
        class MinimalStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("MinimalStrategy")
                self.signal_generator = MinimalSignalGenerator()
                
            def generate_entry(self, data):
                return self.signal_generator.get_entry_signals(data)
                
            def generate_exit(self, data, position, index=-1):
                return self.signal_generator.get_exit_signals(data, position, index)
        
        # テストデータ
        np.random.seed(42)
        n = 30
        
        base_prices = 100 + np.cumsum(np.random.randn(n) * 0.05)
        
        data = pd.DataFrame({
            'open': np.roll(base_prices, 1),
            'high': base_prices + np.abs(np.random.randn(n)) * 0.1,
            'low': base_prices - np.abs(np.random.randn(n)) * 0.1,
            'close': base_prices,
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        # ストラテジーテスト
        strategy = MinimalStrategy()
        entry_signals = strategy.generate_entry(data)
        
        long_count = np.sum(entry_signals == 1)
        short_count = np.sum(entry_signals == -1)
        
        print(f"✓ 最小限ストラテジー成功")
        print(f"  - エントリー: ロング{long_count}回, ショート{short_count}回")
        
        return True
        
    except Exception as e:
        print(f"✗ 最小限ストラテジーエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行"""
    
    print("最小限ハイパーアダプティブチャネルテスト開始")
    print("="*50)
    
    # シグナル直接テスト
    success1 = test_minimal_strategy()
    
    # 実装パス確認用のダミーテスト
    print("\n=== 実装確認 ===")
    try:
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelStrategy
        print("✓ HyperAdaptiveChannelStrategy インポート成功")
        
        # デフォルト値でインスタンス作成をテスト
        try:
            strategy = HyperAdaptiveChannelStrategy()
            print("✓ デフォルト値でのストラテジー作成成功")
        except Exception as e:
            print(f"✗ ストラテジー作成失敗: {e}")
            
    except Exception as e:
        print(f"✗ インポートエラー: {e}")
    
    print("\n" + "="*50)
    print("テスト完了")


if __name__ == "__main__":
    main()