#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.ultimate_ma_v3_chart import UltimateMAV3Chart


def run_comprehensive_demo():
    """
    🚀 UltimateMA V3 包括的デモ実行
    複数のシンボルと時間足でテストを実行
    """
    print("🚀 UltimateMA V3 包括的デモ開始")
    print("=" * 60)
    
    # テスト設定
    test_configs = [
        {
            'symbol': 'BTC',
            'market_type': 'spot',
            'timeframe': '4h',
            'title': 'Bitcoin 4時間足',
            'output': 'ultimate_ma_v3_btc_4h.png'
        },
        {
            'symbol': 'ETH',
            'market_type': 'spot', 
            'timeframe': '4h',
            'title': 'Ethereum 4時間足',
            'output': 'ultimate_ma_v3_eth_4h.png'
        },
        {
            'symbol': 'BTC',
            'market_type': 'spot',
            'timeframe': '1d',
            'title': 'Bitcoin 日足',
            'output': 'ultimate_ma_v3_btc_1d.png'
        }
    ]
    
    # パラメータ設定（最適化済み）
    optimized_params = {
        'super_smooth_period': 6,
        'zero_lag_period': 12,
        'realtime_window': 21,
        'quantum_window': 13,
        'fractal_window': 13,
        'entropy_window': 13,
        'base_threshold': 0.0015,
        'min_confidence': 0.12
    }
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n📊 テスト {i}/{len(test_configs)}: {config['title']}")
        print("-" * 40)
        
        try:
            # チャートインスタンス作成
            chart = UltimateMAV3Chart()
            
            # データ読み込み
            data = chart.load_binance_data_direct(
                symbol=config['symbol'],
                market_type=config['market_type'],
                timeframe=config['timeframe']
            )
            
            if data is None:
                print(f"❌ {config['symbol']} {config['timeframe']} データの読み込みに失敗")
                continue
            
            # インジケーター計算
            chart.calculate_indicators(**optimized_params)
            
            # 統計情報表示
            chart.print_statistics()
            
            # チャート描画（最新1年分）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            chart.plot(
                title=f"UltimateMA V3 Analysis - {config['title']}",
                start_date=start_date.strftime('%Y-%m-%d'),
                show_volume=True,
                show_signals=True,
                show_filters=True,
                figsize=(24, 18),
                savefig=config['output'],
                max_data_points=1500
            )
            
            # 結果を記録
            result = chart.ultimate_ma_v3._result
            if result:
                results.append({
                    'symbol': config['symbol'],
                    'timeframe': config['timeframe'],
                    'current_trend': result.current_trend,
                    'current_confidence': result.current_confidence,
                    'avg_confidence': np.nanmean(result.trend_confidence),
                    'up_signals': np.sum(result.trend_signals == 1),
                    'down_signals': np.sum(result.trend_signals == -1),
                    'range_signals': np.sum(result.trend_signals == 0),
                    'total_signals': len(result.trend_signals),
                    'quantum_avg': np.nanmean(result.quantum_state),
                    'mtf_consensus_avg': np.nanmean(result.multi_timeframe_consensus),
                    'fractal_avg': np.nanmean(result.fractal_dimension),
                    'entropy_avg': np.nanmean(result.entropy_level)
                })
            
            print(f"✅ {config['title']} 完了 - チャート保存: {config['output']}")
            
        except Exception as e:
            print(f"❌ {config['title']} でエラー発生: {str(e)}")
            continue
    
    # 総合結果表示
    print("\n" + "=" * 60)
    print("📈 総合結果サマリー")
    print("=" * 60)
    
    if results:
        # 結果をDataFrameに変換
        df_results = pd.DataFrame(results)
        
        print("\n🎯 現在のトレンド状況:")
        for _, row in df_results.iterrows():
            trend_emoji = "🟢" if row['current_trend'] == 'up' else "🔴" if row['current_trend'] == 'down' else "🟡"
            print(f"  {trend_emoji} {row['symbol']} {row['timeframe']}: {row['current_trend']} "
                  f"(信頼度: {row['current_confidence']:.3f})")
        
        print(f"\n📊 シグナル分布統計:")
        total_up = df_results['up_signals'].sum()
        total_down = df_results['down_signals'].sum()
        total_range = df_results['range_signals'].sum()
        total_all = total_up + total_down + total_range
        
        print(f"  🟢 上昇シグナル: {total_up:,}回 ({total_up/total_all*100:.1f}%)")
        print(f"  🔴 下降シグナル: {total_down:,}回 ({total_down/total_all*100:.1f}%)")
        print(f"  🟡 レンジシグナル: {total_range:,}回 ({total_range/total_all*100:.1f}%)")
        
        print(f"\n🔬 量子分析統計:")
        print(f"  平均信頼度: {df_results['avg_confidence'].mean():.3f}")
        print(f"  平均量子状態: {df_results['quantum_avg'].mean():.3f}")
        print(f"  平均MTF合意度: {df_results['mtf_consensus_avg'].mean():.3f}")
        print(f"  平均フラクタル次元: {df_results['fractal_avg'].mean():.3f}")
        print(f"  平均エントロピー: {df_results['entropy_avg'].mean():.3f}")
        
        # 最も信頼度の高いシグナル
        max_confidence_idx = df_results['current_confidence'].idxmax()
        best_signal = df_results.iloc[max_confidence_idx]
        print(f"\n🏆 最高信頼度シグナル:")
        print(f"  {best_signal['symbol']} {best_signal['timeframe']}: "
              f"{best_signal['current_trend']} (信頼度: {best_signal['current_confidence']:.3f})")
        
        # 詳細統計テーブル
        print(f"\n📋 詳細統計テーブル:")
        print(df_results[['symbol', 'timeframe', 'current_trend', 'current_confidence', 
                         'avg_confidence', 'quantum_avg', 'mtf_consensus_avg']].round(3).to_string(index=False))
    
    else:
        print("❌ 有効な結果がありませんでした。")
    
    print(f"\n🎉 UltimateMA V3 包括的デモ完了!")
    print(f"📁 生成されたチャート: {len([c['output'] for c in test_configs])}枚")
    print("=" * 60)


def run_parameter_optimization_demo():
    """
    🔧 パラメータ最適化デモ
    異なるパラメータ設定での性能比較
    """
    print("\n🔧 UltimateMA V3 パラメータ最適化デモ")
    print("=" * 60)
    
    # パラメータセット
    param_sets = [
        {
            'name': '高感度設定',
            'params': {
                'super_smooth_period': 4,
                'zero_lag_period': 8,
                'realtime_window': 13,
                'quantum_window': 8,
                'base_threshold': 0.001,
                'min_confidence': 0.08
            }
        },
        {
            'name': 'バランス設定',
            'params': {
                'super_smooth_period': 8,
                'zero_lag_period': 16,
                'realtime_window': 34,
                'quantum_window': 16,
                'base_threshold': 0.002,
                'min_confidence': 0.15
            }
        },
        {
            'name': '安定性重視設定',
            'params': {
                'super_smooth_period': 12,
                'zero_lag_period': 24,
                'realtime_window': 55,
                'quantum_window': 21,
                'base_threshold': 0.004,
                'min_confidence': 0.25
            }
        }
    ]
    
    # BTCデータでテスト
    chart = UltimateMAV3Chart()
    data = chart.load_binance_data_direct(symbol='BTC', timeframe='4h')
    
    if data is None:
        print("❌ BTCデータの読み込みに失敗")
        return
    
    optimization_results = []
    
    for i, param_set in enumerate(param_sets, 1):
        print(f"\n🧪 テスト {i}/{len(param_sets)}: {param_set['name']}")
        print("-" * 30)
        
        try:
            # 新しいチャートインスタンス
            test_chart = UltimateMAV3Chart()
            test_chart.data = data.copy()
            
            # パラメータで計算
            test_chart.calculate_indicators(**param_set['params'])
            
            # 結果取得
            result = test_chart.ultimate_ma_v3._result
            if result:
                # 性能指標計算
                up_signals = np.sum(result.trend_signals == 1)
                down_signals = np.sum(result.trend_signals == -1)
                total_signals = len(result.trend_signals)
                signal_ratio = (up_signals + down_signals) / total_signals
                avg_confidence = np.nanmean(result.trend_confidence)
                high_confidence_ratio = np.sum(result.trend_confidence > 0.5) / len(result.trend_confidence)
                
                optimization_results.append({
                    'name': param_set['name'],
                    'signal_ratio': signal_ratio,
                    'avg_confidence': avg_confidence,
                    'high_confidence_ratio': high_confidence_ratio,
                    'current_trend': result.current_trend,
                    'current_confidence': result.current_confidence,
                    'quantum_range': np.nanmax(result.quantum_state) - np.nanmin(result.quantum_state),
                    'mtf_consensus': np.nanmean(result.multi_timeframe_consensus)
                })
                
                print(f"  シグナル率: {signal_ratio:.1%}")
                print(f"  平均信頼度: {avg_confidence:.3f}")
                print(f"  高信頼度率: {high_confidence_ratio:.1%}")
                print(f"  現在のトレンド: {result.current_trend} ({result.current_confidence:.3f})")
        
        except Exception as e:
            print(f"❌ {param_set['name']} でエラー: {str(e)}")
    
    # 最適化結果表示
    if optimization_results:
        print(f"\n🏆 パラメータ最適化結果:")
        opt_df = pd.DataFrame(optimization_results)
        print(opt_df[['name', 'signal_ratio', 'avg_confidence', 'high_confidence_ratio', 
                     'current_confidence']].round(3).to_string(index=False))
        
        # 最適パラメータ推奨
        best_idx = opt_df['avg_confidence'].idxmax()
        best_params = optimization_results[best_idx]
        print(f"\n🎯 推奨パラメータ: {best_params['name']}")
        print(f"   理由: 最高平均信頼度 {best_params['avg_confidence']:.3f}")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='UltimateMA V3 包括的デモ')
    parser.add_argument('--comprehensive', action='store_true', help='包括的デモを実行')
    parser.add_argument('--optimization', action='store_true', help='パラメータ最適化デモを実行')
    parser.add_argument('--all', action='store_true', help='全てのデモを実行')
    args = parser.parse_args()
    
    if args.all or args.comprehensive:
        run_comprehensive_demo()
    
    if args.all or args.optimization:
        run_parameter_optimization_demo()
    
    if not any([args.comprehensive, args.optimization, args.all]):
        print("使用方法:")
        print("  --comprehensive: 包括的デモ実行")
        print("  --optimization: パラメータ最適化デモ実行")
        print("  --all: 全デモ実行")
        print("\n例: python ultimate_ma_v3_demo_comprehensive.py --all")


if __name__ == "__main__":
    main() 