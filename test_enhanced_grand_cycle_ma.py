#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_grand_cycle_ma():
    """拡張されたグランドサイクルMAのテスト（カルマンフィルター + スムーサー統合）"""
    print("=== 拡張グランドサイクルMA テスト開始 ===")
    
    try:
        # 直接インポートを試行
        from indicators.grand_cycle_ma import GrandCycleMA
        print("✓ GrandCycleMA インポート成功")
        
        # テストデータの作成
        print("\n1. テストデータ作成中...")
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        np.random.seed(42)
        
        # 複雑な価格データを生成（トレンド + 複数サイクル + ノイズ）
        t = np.linspace(0, 6*np.pi, 200)
        trend = np.linspace(100, 140, 200)
        
        # 複数のサイクル成分
        long_cycle = 12 * np.sin(t * 0.3)    # 長期サイクル
        medium_cycle = 6 * np.sin(t * 1.2)   # 中期サイクル
        short_cycle = 3 * np.sin(t * 3.0)    # 短期サイクル
        noise = np.random.normal(0, 2, 200)  # ノイズ
        
        close_prices = trend + long_cycle + medium_cycle + short_cycle + noise
        
        data = pd.DataFrame({
            'open': close_prices * 0.998,
            'high': close_prices * 1.004,
            'low': close_prices * 0.996,
            'close': close_prices,
            'volume': np.random.uniform(100000, 500000, 200)
        }, index=dates)
        
        print(f"✓ テストデータ作成完了: {len(data)}件")
        print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
        
        # 2. 様々な設定でのテスト
        test_configs = [
            {
                'name': 'ベーシック（フィルター無し）',
                'params': {
                    'detector_type': 'hody',
                    'use_kalman_filter': False,
                    'use_smoother': False,
                    'src_type': 'close'
                }
            },
            {
                'name': 'FRAMAスムーサーのみ',
                'params': {
                    'detector_type': 'hody',
                    'use_kalman_filter': False,
                    'use_smoother': True,
                    'smoother_type': 'frama',
                    'smoother_params': {'period': 14},
                    'src_type': 'close'
                }
            },
            {
                'name': 'カルマンフィルターのみ',
                'params': {
                    'detector_type': 'hody',
                    'use_kalman_filter': True,
                    'kalman_filter_type': 'adaptive',
                    'use_smoother': False,
                    'src_type': 'close'
                }
            },
            {
                'name': 'カルマンフィルター + FRAMAスムーサー',
                'params': {
                    'detector_type': 'hody',
                    'use_kalman_filter': True,
                    'kalman_filter_type': 'adaptive',
                    'use_smoother': True,
                    'smoother_type': 'frama',
                    'smoother_params': {'period': 16},
                    'src_type': 'close'
                }
            },
            {
                'name': '究極スムーサー + 量子カルマン',
                'params': {
                    'detector_type': 'cycle_period',
                    'use_kalman_filter': True,
                    'kalman_filter_type': 'quantum_adaptive',
                    'use_smoother': True,
                    'smoother_type': 'ultimate_smoother',
                    'smoother_params': {'period': 20},
                    'src_type': 'hlc3'
                }
            }
        ]
        
        print(f"\n2. {len(test_configs)}種類の設定でテスト:")
        
        results = {}
        
        for config in test_configs:
            try:
                print(f"\n--- {config['name']} ---")
                
                # グランドサイクルMAの作成
                grand_cycle_ma = GrandCycleMA(**config['params'])
                
                # 計算実行
                result = grand_cycle_ma.calculate(data)
                
                # 結果の保存
                results[config['name']] = {
                    'result': result,
                    'config': config
                }
                
                # 統計情報
                valid_mama = result.grand_mama_values[~np.isnan(result.grand_mama_values)]
                valid_fama = result.grand_fama_values[~np.isnan(result.grand_fama_values)]
                valid_alpha = result.alpha_values[~np.isnan(result.alpha_values)]
                
                print(f"  ✓ 計算成功")
                print(f"  グランドMAMA: {len(valid_mama)}/{len(result.grand_mama_values)} 有効データ")
                if len(valid_mama) > 0:
                    print(f"    平均値: {np.mean(valid_mama):.4f}")
                    print(f"    標準偏差: {np.std(valid_mama):.4f}")
                
                print(f"  Alpha値範囲: {np.min(valid_alpha):.4f} - {np.max(valid_alpha):.4f}")
                
            except Exception as e:
                print(f"  ✗ エラー: {e}")
                continue
        
        # 3. 結果の比較分析
        if results:
            print("\n3. 結果の比較分析:")
            
            original_price = data['close'].values
            
            print("\n設定別のパフォーマンス比較:")
            for name, result_data in results.items():
                result = result_data['result']
                valid_mama = result.grand_mama_values[~np.isnan(result.grand_mama_values)]
                
                if len(valid_mama) > 10:  # 十分なデータがある場合のみ
                    # 遅延の測定（簡易版）
                    price_subset = original_price[-len(valid_mama):]
                    correlation = np.corrcoef(price_subset, valid_mama)[0, 1]
                    
                    # ボラティリティの測定
                    mama_volatility = np.std(np.diff(valid_mama))
                    price_volatility = np.std(np.diff(price_subset))
                    volatility_ratio = mama_volatility / price_volatility
                    
                    print(f"  {name}:")
                    print(f"    価格相関: {correlation:.4f}")
                    print(f"    ボラティリティ比: {volatility_ratio:.4f} (低いほど滑らか)")
            
            # 4. 視覚化
            print("\n4. 結果の視覚化中...")
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # 価格とグランドサイクルMAMAの比較
            axes[0].plot(data.index, data['close'], label='Original Price', alpha=0.7, color='black', linewidth=1)
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (name, result_data) in enumerate(results.items()):
                result = result_data['result']
                color = colors[i % len(colors)]
                
                valid_indices = ~np.isnan(result.grand_mama_values)
                if np.any(valid_indices):
                    axes[0].plot(
                        data.index[valid_indices], 
                        result.grand_mama_values[valid_indices],
                        label=f'Grand MAMA ({name})',
                        alpha=0.8,
                        color=color,
                        linewidth=1.5
                    )
            
            axes[0].set_title('Price vs Enhanced Grand Cycle MAMA (設定比較)')
            axes[0].set_ylabel('Price')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # Alpha値の比較
            for i, (name, result_data) in enumerate(results.items()):
                result = result_data['result']
                color = colors[i % len(colors)]
                
                valid_indices = ~np.isnan(result.alpha_values)
                if np.any(valid_indices):
                    axes[1].plot(
                        data.index[valid_indices],
                        result.alpha_values[valid_indices],
                        label=f'Alpha ({name})',
                        alpha=0.8,
                        color=color,
                        linewidth=1.5
                    )
            
            axes[1].set_title('Alpha Values Comparison (適応速度比較)')
            axes[1].set_ylabel('Alpha')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            
            # サイクル周期の比較
            for i, (name, result_data) in enumerate(results.items()):
                result = result_data['result']
                color = colors[i % len(colors)]
                
                valid_indices = ~np.isnan(result.cycle_period)
                if np.any(valid_indices):
                    axes[2].plot(
                        data.index[valid_indices],
                        result.cycle_period[valid_indices],
                        label=f'Cycle Period ({name})',
                        alpha=0.8,
                        color=color,
                        linewidth=1.5
                    )
            
            axes[2].set_title('Cycle Period Comparison (サイクル周期比較)')
            axes[2].set_ylabel('Period')
            axes[2].set_xlabel('Date')
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像を保存
            output_file = 'enhanced_grand_cycle_ma_test.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ グラフを保存しました: {output_file}")
            
            plt.show()
        
        # 5. パフォーマンステスト
        if results:
            print("\n5. パフォーマンステスト:")
            import time
            
            # 最初の設定でパフォーマンステスト
            first_config = list(results.keys())[0]
            config_params = results[first_config]['config']['params']
            
            grand_cycle_ma = GrandCycleMA(**config_params)
            
            # 5回実行して平均時間を測定
            times = []
            for i in range(5):
                start_time = time.time()
                grand_cycle_ma.calculate(data)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            print(f"  平均計算時間 ({first_config}): {avg_time:.4f}秒")
            print(f"  データ長: {len(data)}件")
            print(f"  処理速度: {len(data)/avg_time:.0f}件/秒")
        
        print("\n=== 拡張グランドサイクルMA テスト完了 ===")
        
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        print("必要なモジュールが見つかりません")
        return False
    
    except Exception as e:
        import traceback
        print(f"✗ テストエラー: {e}")
        print(f"詳細: {traceback.format_exc()}")
        return False
    
    return True


def test_smoother_kalman_combinations():
    """スムーサーとカルマンフィルターの組み合わせテスト"""
    print("\n=== スムーサー・カルマンフィルター組み合わせテスト ===")
    
    try:
        from indicators.grand_cycle_ma import GrandCycleMA
        
        # 簡単なテストデータ
        np.random.seed(123)
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, 50))
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(10000, 50000, 50)
        }, index=dates)
        
        # 組み合わせテスト
        combinations = [
            {'kalman': None, 'smoother': None},
            {'kalman': 'adaptive', 'smoother': None},
            {'kalman': None, 'smoother': 'frama'},
            {'kalman': 'adaptive', 'smoother': 'frama'},
            {'kalman': 'quantum_adaptive', 'smoother': 'ultimate_smoother'},
        ]
        
        for combo in combinations:
            try:
                kalman_type = combo['kalman']
                smoother_type = combo['smoother']
                
                params = {
                    'detector_type': 'hody',
                    'use_kalman_filter': kalman_type is not None,
                    'use_smoother': smoother_type is not None,
                    'src_type': 'close'
                }
                
                if kalman_type:
                    params['kalman_filter_type'] = kalman_type
                if smoother_type:
                    params['smoother_type'] = smoother_type
                
                grand_cycle_ma = GrandCycleMA(**params)
                result = grand_cycle_ma.calculate(data)
                
                valid_count = np.sum(~np.isnan(result.grand_mama_values))
                
                combo_name = f"Kalman:{kalman_type or 'None'}, Smoother:{smoother_type or 'None'}"
                print(f"  {combo_name}: {valid_count}/{len(data)} 有効データ ✓")
                
            except Exception as e:
                print(f"  {combo_name}: エラー - {e} ✗")
        
        print("✓ 組み合わせテスト完了")
        
    except Exception as e:
        print(f"✗ 組み合わせテストエラー: {e}")


if __name__ == "__main__":
    # メインテスト
    success = test_enhanced_grand_cycle_ma()
    
    if success:
        # 追加テスト
        test_smoother_kalman_combinations()