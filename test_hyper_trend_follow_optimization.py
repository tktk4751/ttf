#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_trend_follow.strategy import HyperTrendFollowStrategy

# Optunaをモックして最適化パラメータをテスト
class MockTrial:
    """Optunaトライアルのモック"""
    
    def suggest_int(self, name, low, high, step=1):
        """整数値のサジェスト"""
        if step == 1:
            return low + ((high - low) // 2)
        else:
            # stepを考慮した中間値
            mid_steps = ((high - low) // step) // 2
            return low + (mid_steps * step)
    
    def suggest_float(self, name, low, high, step=None, log=False):
        """浮動小数点値のサジェスト"""
        if log:
            # 対数スケールの場合
            return 10 ** ((np.log10(low) + np.log10(high)) / 2)
        else:
            # 線形スケールの場合
            if step is not None:
                mid_steps = int(((high - low) / step) / 2)
                return low + (mid_steps * step)
            else:
                return (low + high) / 2
    
    def suggest_categorical(self, name, choices):
        """カテゴリ値のサジェスト"""
        # 最初の選択肢を返す
        return choices[0]

def test_optimization_params():
    """最適化パラメータのテスト"""
    print("=== HyperTrendFollow最適化パラメータテスト ===")
    
    try:
        # モックトライアルを作成
        mock_trial = MockTrial()
        
        # 最適化パラメータを生成
        opt_params = HyperTrendFollowStrategy.create_optimization_params(mock_trial)
        
        print(f"✓ 最適化パラメータ生成成功")
        print(f"  生成されたパラメータ数: {len(opt_params)}")
        
        # パラメータの一部を表示
        print(f"\n主要パラメータ:")
        key_params = [
            'hyper_frama_period', 'hyper_frama_src_type', 'hyper_frama_alpha_multiplier',
            'channel_period', 'channel_fixed_multiplier', 'channel_multiplier_mode'
        ]
        
        for param in key_params:
            if param in opt_params:
                print(f"  {param}: {opt_params[param]}")
        
        # 戦略パラメータへの変換をテスト
        strategy_params = HyperTrendFollowStrategy.convert_params_to_strategy_format(opt_params)
        
        print(f"\n✓ 戦略パラメータ変換成功")
        print(f"  変換されたパラメータ数: {len(strategy_params)}")
        
        # 戦略の初期化をテスト
        strategy = HyperTrendFollowStrategy(**strategy_params)
        
        print(f"✓ 最適化パラメータでの戦略初期化成功")
        print(f"  戦略名: {strategy.name}")
        
        # 戦略情報の取得
        strategy_info = strategy.get_strategy_info()
        print(f"\n戦略情報:")
        print(f"  名前: {strategy_info['name']}")
        print(f"  説明: {strategy_info['description']}")
        print(f"  機能数: {len(strategy_info['features'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ 最適化パラメータテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_coverage():
    """パラメータの網羅性をテスト"""
    print("\n=== パラメータ網羅性テスト ===")
    
    try:
        # デフォルトの戦略インスタンスを作成
        default_strategy = HyperTrendFollowStrategy()
        default_params = default_strategy._parameters
        
        # 最適化パラメータを生成
        mock_trial = MockTrial()
        opt_params = HyperTrendFollowStrategy.create_optimization_params(mock_trial)
        strategy_params = HyperTrendFollowStrategy.convert_params_to_strategy_format(opt_params)
        
        # パラメータの比較
        missing_from_opt = set(default_params.keys()) - set(strategy_params.keys())
        extra_in_opt = set(strategy_params.keys()) - set(default_params.keys())
        
        print(f"デフォルトパラメータ数: {len(default_params)}")
        print(f"最適化パラメータ数: {len(strategy_params)}")
        
        if missing_from_opt:
            print(f"\n最適化に含まれていないパラメータ: {len(missing_from_opt)}")
            for param in sorted(missing_from_opt):
                print(f"  - {param}")
        else:
            print(f"\n✓ 全てのパラメータが最適化に含まれています")
        
        if extra_in_opt:
            print(f"\n最適化で追加されたパラメータ: {len(extra_in_opt)}")
            for param in sorted(extra_in_opt):
                print(f"  + {param}")
        
        # 網羅率の計算
        coverage_rate = (len(default_params) - len(missing_from_opt)) / len(default_params) * 100
        print(f"\nパラメータ網羅率: {coverage_rate:.1f}%")
        
        return len(missing_from_opt) == 0
        
    except Exception as e:
        print(f"✗ パラメータ網羅性テストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("HyperTrendFollow戦略最適化テスト開始")
    print("=" * 60)
    
    results = []
    
    # テストの実行
    results.append(test_optimization_params())
    results.append(test_parameter_coverage())
    
    # 結果のまとめ
    print(f"\n{'='*60}")
    print(f"テスト結果まとめ:")
    print(f"  実行済みテスト: {len(results)}")
    print(f"  成功: {sum(results)}")
    print(f"  失敗: {len(results) - sum(results)}")
    
    if all(results):
        print(f"✓ 全ての最適化テストが成功しました！")
        print(f"\nHyperTrendFollow戦略の最適化機能が正常に動作しています。")
    else:
        print(f"✗ 一部のテストが失敗しました。")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()