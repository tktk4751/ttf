#!/usr/bin/env python3
"""
Hyper Adaptive Channel ストラテジー簡単テスト

ハイパーアダプティブチャネルストラテジーの基本動作確認
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')

def test_basic_functionality():
    """基本機能テスト"""
    
    print("=== Hyper Adaptive Channel ストラテジー基本テスト ===")
    
    try:
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelStrategy
        
        # テストデータ作成
        np.random.seed(42)
        n = 100
        
        # シンプルな価格データ
        base_prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': np.roll(base_prices, 1),
            'high': base_prices + np.abs(np.random.randn(n)) * 0.2,
            'low': base_prices - np.abs(np.random.randn(n)) * 0.2,
            'close': base_prices,
            'volume': np.random.randint(1000, 3000, n)
        })
        
        data['open'].iloc[0] = data['close'].iloc[0]
        
        print(f"テストデータ作成: {len(data)}件")
        
        # ストラテジー作成
        print("ストラテジー作成中...")
        strategy = HyperAdaptiveChannelStrategy(
            period=12,
            midline_smoother='hyper_frama',
            multiplier_mode='dynamic'
        )
        
        print("✓ ストラテジー作成成功")
        
        # エントリーシグナル生成
        print("エントリーシグナル生成中...")
        entry_signals = strategy.generate_entry(data)
        
        # 統計
        long_count = np.sum(entry_signals == 1)
        short_count = np.sum(entry_signals == -1)
        total_signals = long_count + short_count
        
        print(f"✓ エントリーシグナル生成成功")
        print(f"  - ロング: {long_count}回")
        print(f"  - ショート: {short_count}回")
        print(f"  - 総シグナル: {total_signals}回")
        print(f"  - シグナル率: {total_signals/len(data)*100:.1f}%")
        
        # エグジットテスト
        exit_count = 0
        if total_signals > 0:
            for i in range(len(data)):
                if entry_signals[i] != 0:
                    exit_signal = strategy.generate_exit(data, entry_signals[i], i)
                    if exit_signal:
                        exit_count += 1
        
        print(f"  - エグジットテスト: {exit_count}回")
        
        return True
        
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_params():
    """最適化パラメータテスト"""
    
    print("\n=== 最適化パラメータテスト ===")
    
    try:
        from strategies.implementations.hyper_adaptive_channel import HyperAdaptiveChannelStrategy
        
        # モックトライアル
        class MockTrial:
            def suggest_int(self, name, low, high):
                return np.random.randint(low, high + 1)
                
            def suggest_float(self, name, low, high, step=None):
                if step:
                    n_steps = int((high - low) / step) + 1
                    return low + step * np.random.randint(0, n_steps)
                else:
                    return np.random.uniform(low, high)
                
            def suggest_categorical(self, name, choices):
                return np.random.choice(choices)
        
        trial = MockTrial()
        
        # 各最適化レベルをテスト
        levels = ['basic', 'balanced', 'comprehensive']
        
        for level in levels:
            print(f"\n--- {level} レベルテスト ---")
            
            try:
                params = HyperAdaptiveChannelStrategy.create_optimization_params(trial, level)
                print(f"✓ {level}: {len(params)}個のパラメータ生成")
                
                # 変換テスト
                strategy_params = HyperAdaptiveChannelStrategy.convert_params_to_strategy_format(params)
                print(f"✓ パラメータ変換: {len(strategy_params)}個")
                
                # ストラテジー作成テスト
                strategy = HyperAdaptiveChannelStrategy(**strategy_params)
                print(f"✓ ストラテジー作成成功")
                
            except Exception as e:
                print(f"✗ {level} エラー: {e}")
        
        # 段階的最適化テスト
        print(f"\n--- 段階的最適化テスト ---")
        
        for stage in [1, 2, 3, 4]:
            try:
                params = HyperAdaptiveChannelStrategy.create_staged_optimization_params(trial, stage)
                print(f"✓ 段階{stage}: {len(params)}個のパラメータ")
            except Exception as e:
                print(f"✗ 段階{stage} エラー: {e}")
        
        # パラメータグループテスト
        groups = HyperAdaptiveChannelStrategy.get_optimization_groups()
        total_group_params = sum(len(group) for group in groups.values())
        print(f"\n✓ パラメータグループ: {len(groups)}グループ, 総{total_group_params}パラメータ")
        
        return True
        
    except Exception as e:
        print(f"✗ 最適化テストエラー: {e}")
        return False


def main():
    """メイン実行"""
    
    print("Hyper Adaptive Channel ストラテジー簡単テスト開始")
    print("="*60)
    
    # 基本機能テスト
    success1 = test_basic_functionality()
    
    # 最適化パラメータテスト
    success2 = test_optimization_params()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("✓ 全テスト成功")
    else:
        print("✗ 一部テスト失敗")
    
    print("テスト完了")


if __name__ == "__main__":
    main()