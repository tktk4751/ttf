#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **UKF比較テスト実行スクリプト** 🎯

修正版無香料カルマンフィルターと元のUKFを実際の相場データで比較テスト：
- 設定ファイルからのデータ読み込み
- 両方のUKFの計算と比較
- チャート描画と統計表示
- パフォーマンス分析
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization.ukf_comparison_chart import UKFComparisonChart


def run_ukf_comparison_test():
    """UKF比較テストを実行"""
    
    print("🚀 UKF比較テストを開始します")
    print("="*60)
    
    # 設定ファイルのパス
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        print("   config.yamlファイルを作成してください")
        return
    
    try:
        # UKF比較チャートを作成
        chart = UKFComparisonChart()
        
        # 1. データの読み込み
        print("\n📊 ステップ1: データ読み込み")
        chart.load_data_from_config(str(config_path))
        
        # 2. UKF比較計算（複数のパラメータセットでテスト）
        print("\n🔬 ステップ2: UKF比較計算")
        
        # テストパラメータセット
        test_params = [
            {
                'name': '標準パラメータ',
                'alpha': 0.001,
                'process_noise_scale': 0.001,
                'volatility_window': 10
            },
            {
                'name': '高感度パラメータ',
                'alpha': 0.01,
                'process_noise_scale': 0.01,
                'volatility_window': 5
            },
            {
                'name': '低ノイズパラメータ',
                'alpha': 0.0001,
                'process_noise_scale': 0.0001,
                'volatility_window': 20
            }
        ]
        
        # 各パラメータセットでテスト
        for i, params in enumerate(test_params, 1):
            print(f"\n--- テスト{i}: {params['name']} ---")
            
            # UKF計算（改善された不確実性制御付き）
            chart.calculate_ukf_comparison(
                src_type='close',
                alpha=params['alpha'],
                process_noise_scale=params['process_noise_scale'],
                volatility_window=params['volatility_window'],
                adaptive_noise=True,
                            conservative_uncertainty=True,
            max_uncertainty_ratio=1.5  # さらに厳しい制限
            )
            
            # 3. チャート描画と保存
            print(f"\n📈 ステップ3: チャート描画 - {params['name']}")
            
            # 出力ファイル名
            output_file = project_root / "output" / f"ukf_comparison_{params['name'].replace(' ', '_')}.png"
            output_file.parent.mkdir(exist_ok=True)
            
            # チャート描画
            chart.plot_comparison(
                title=f"UKF比較: {params['name']}",
                show_volume=True,
                figsize=(16, 14),
                savefig=str(output_file)
            )
            
            print(f"✅ チャートを保存しました: {output_file}")
            
            # 統計表示はplot_comparisonで自動実行される
            
        print("\n" + "="*60)
        print("🎉 UKF比較テストが完了しました！")
        print(f"📁 結果は {project_root / 'output'} フォルダに保存されています")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def run_simple_comparison():
    """簡単な比較テスト"""
    
    print("🔥 簡単UKF比較テスト")
    print("="*40)
    
    # 設定ファイルのパス
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    try:
        chart = UKFComparisonChart()
        
        # データ読み込み
        chart.load_data_from_config(str(config_path))
        
        # UKF計算（改善された不確実性制御付き）
        chart.calculate_ukf_comparison(
            src_type='close',
            alpha=0.001,
            process_noise_scale=0.001,
            volatility_window=10,
            conservative_uncertainty=True,
            max_uncertainty_ratio=1.0  # 極めて保守的な制限
        )
        
        # 統計のみを表示
        chart._print_comparison_stats()
        
        print("\n📊 チャート表示をスキップしました（統計のみ表示）")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


def run_stats_only():
    """統計のみのテスト"""
    
    print("📊 UKF統計比較テスト")
    print("="*40)
    
    # 設定ファイルのパス
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    try:
        chart = UKFComparisonChart()
        
        # データ読み込み
        chart.load_data_from_config(str(config_path))
        
        # UKF計算（改善された不確実性制御付き）
        chart.calculate_ukf_comparison(
            src_type='close',
            alpha=0.001,
            process_noise_scale=0.001,
            volatility_window=10,
            conservative_uncertainty=True,
            max_uncertainty_ratio=1.0  # 極めて保守的な制限
        )
        
        # 統計のみを表示
        chart._print_comparison_stats()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='UKF比較テスト')
    parser.add_argument('--simple', '-s', action='store_true', 
                       help='簡単なテストを実行（統計のみ表示）')
    parser.add_argument('--full', '-f', action='store_true', 
                       help='完全なテストを実行（複数パラメータ・保存あり）')
    parser.add_argument('--stats', action='store_true',
                       help='統計のみを表示（チャート表示なし）')
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_comparison()
    elif args.full:
        run_ukf_comparison_test()
    elif args.stats:
        run_stats_only()
    else:
        # デフォルトは統計のみ
        print("使用法:")
        print("  python ukf_comparison_test.py --simple  # 簡単テスト（統計のみ）")
        print("  python ukf_comparison_test.py --full    # 完全テスト（チャート保存）")
        print("  python ukf_comparison_test.py --stats   # 統計のみ")
        print("\nデフォルトで統計のみテストを実行します...")
        run_stats_only() 