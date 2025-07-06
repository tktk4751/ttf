#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Ultimate Efficiency Ratio Chart 使用例 🚀

このスクリプトは、Ultimate Efficiency Ratioインジケーターを使用した
高度なチャート分析の使用例を示します。

実行方法:
1. 基本実行:
   python examples/ultimate_efficiency_ratio_chart_example.py

2. パラメータ指定実行:
   python examples/ultimate_efficiency_ratio_chart_example.py --config config.yaml --period 21 --hilbert-window 16

3. 特定期間のチャート表示:
   python examples/ultimate_efficiency_ratio_chart_example.py --start 2024-01-01 --end 2024-12-31

4. 量子解析なしで表示:
   python examples/ultimate_efficiency_ratio_chart_example.py --no-quantum --no-hilbert

5. チャートをファイルに保存:
   python examples/ultimate_efficiency_ratio_chart_example.py --output ultimate_er_chart.png
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.ultimate_efficiency_ratio_chart import UltimateEfficiencyRatioChart
import argparse
from pathlib import Path


def basic_example():
    """基本的な使用例"""
    print("🚀 Ultimate Efficiency Ratio Chart - 基本例")
    print("=" * 60)
    
    # チャートオブジェクトの作成
    chart = UltimateEfficiencyRatioChart()
    
    # デフォルトの設定ファイルを使用してデータを読み込み
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        print("設定ファイルを作成するか、--configオプションで指定してください。")
        return
    
    try:
        # データ読み込み
        chart.load_data_from_config(config_path)
        
        # インジケーター計算（デフォルトパラメータ）
        chart.calculate_indicators()
        
        # チャート描画
        chart.plot(title="Ultimate Efficiency Ratio V3.0 - 基本例")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return
    
    print("✅ 基本例の実行が完了しました")


def advanced_example():
    """高度な使用例"""
    print("🚀 Ultimate Efficiency Ratio Chart - 高度な例")
    print("=" * 60)
    
    # チャートオブジェクトの作成
    chart = UltimateEfficiencyRatioChart()
    
    # 設定ファイルパス
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    try:
        # データ読み込み
        print("📊 データを読み込み中...")
        chart.load_data_from_config(config_path)
        
        # カスタムパラメータでインジケーター計算
        print("⚙️ カスタムパラメータでインジケーターを計算中...")
        chart.calculate_indicators(
            period=21,                    # 基本期間を21に設定
            src_type='hlc3',              # HLC3価格を使用
            hilbert_window=16,            # ヒルベルト変換ウィンドウを16に
            her_window=20,                # ハイパー効率率ウィンドウを20に
            slope_index=5,                # トレンド判定期間を5に
            range_threshold=0.002         # レンジしきい値を0.002に
        )
        
        # 高度なチャート描画
        print("📈 高度なチャートを描画中...")
        chart.plot(
            title="Ultimate Efficiency Ratio V3.0 - カスタムパラメータ",
            show_volume=True,             # 出来高表示
            show_quantum=True,            # 量子解析表示
            show_hilbert=True,            # ヒルベルト変換表示
            figsize=(18, 16),             # 大きなサイズ
            style='charles'               # Charlesスタイル
        )
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return
    
    print("✅ 高度な例の実行が完了しました")


def comparison_example():
    """比較表示の例"""
    print("🚀 Ultimate Efficiency Ratio Chart - 比較例")
    print("=" * 60)
    
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    # 短期パラメータの設定
    print("📊 短期パラメータでの分析...")
    chart_short = UltimateEfficiencyRatioChart()
    chart_short.load_data_from_config(config_path)
    chart_short.calculate_indicators(
        period=7,
        hilbert_window=6,
        her_window=10,
        slope_index=2,
        range_threshold=0.005
    )
    chart_short.plot(
        title="Ultimate ER - 短期設定 (敏感)",
        savefig="ultimate_er_short_term.png",
        show_quantum=False,
        show_hilbert=False
    )
    
    # 長期パラメータの設定
    print("📊 長期パラメータでの分析...")
    chart_long = UltimateEfficiencyRatioChart()
    chart_long.load_data_from_config(config_path)
    chart_long.calculate_indicators(
        period=30,
        hilbert_window=20,
        her_window=25,
        slope_index=7,
        range_threshold=0.001
    )
    chart_long.plot(
        title="Ultimate ER - 長期設定 (安定)",
        savefig="ultimate_er_long_term.png",
        show_quantum=False,
        show_hilbert=False
    )
    
    print("✅ 比較例の実行が完了しました")
    print("📁 ファイルが保存されました:")
    print("  - ultimate_er_short_term.png")
    print("  - ultimate_er_long_term.png")


def quantum_analysis_example():
    """量子解析重点の例"""
    print("🚀 Ultimate Efficiency Ratio Chart - 量子解析例")
    print("=" * 60)
    
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return
    
    # チャートオブジェクトの作成
    chart = UltimateEfficiencyRatioChart()
    
    try:
        # データ読み込み
        chart.load_data_from_config(config_path)
        
        # 量子効果を重視したパラメータ設定
        chart.calculate_indicators(
            period=14,
            hilbert_window=12,            # ヒルベルト変換に重点
            her_window=16,
            slope_index=3,
            range_threshold=0.003
        )
        
        # 量子解析に特化したチャート
        chart.plot(
            title="Ultimate ER - 量子解析重点",
            show_volume=True,
            show_quantum=True,            # 量子解析を表示
            show_hilbert=True,            # ヒルベルト変換を表示
            figsize=(20, 18),             # 大きなサイズで詳細表示
            savefig="ultimate_er_quantum_analysis.png"
        )
        
        print("✅ 量子解析例の実行が完了しました")
        print("📁 ファイルが保存されました: ultimate_er_quantum_analysis.png")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Ultimate Efficiency Ratio Chart の使用例')
    parser.add_argument('--example', '-e', choices=['basic', 'advanced', 'comparison', 'quantum'], 
                       default='basic', help='実行する例を選択')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-end', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=14, help='基本期間')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--hilbert-window', type=int, default=12, help='ヒルベルト変換ウィンドウ')
    parser.add_argument('--her-window', type=int, default=16, help='ハイパー効率率ウィンドウ')
    parser.add_argument('--slope-index', type=int, default=3, help='トレンド判定期間')
    parser.add_argument('--range-threshold', type=float, default=0.003, help='レンジ判定しきい値')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-quantum', action='store_true', help='量子解析を非表示')
    parser.add_argument('--no-hilbert', action='store_true', help='ヒルベルト変換を非表示')
    
    args = parser.parse_args()
    
    print("🚀 Ultimate Efficiency Ratio Chart Examples")
    print("=" * 60)
    
    # 指定された例を実行
    if args.example == 'basic':
        if len(sys.argv) == 1:  # 引数なしの場合は基本例
            basic_example()
        else:
            # カスタムパラメータでの実行
            chart = UltimateEfficiencyRatioChart()
            chart.load_data_from_config(args.config)
            chart.calculate_indicators(
                period=args.period,
                src_type=args.src_type,
                hilbert_window=args.hilbert_window,
                her_window=args.her_window,
                slope_index=args.slope_index,
                range_threshold=args.range_threshold
            )
            chart.plot(
                start_date=args.start,
                end_date=args.end,
                show_volume=not args.no_volume,
                show_quantum=not args.no_quantum,
                show_hilbert=not args.no_hilbert,
                savefig=args.output
            )
    elif args.example == 'advanced':
        advanced_example()
    elif args.example == 'comparison':
        comparison_example()
    elif args.example == 'quantum':
        quantum_analysis_example()
    
    print("\n✅ 全ての例の実行が完了しました!")
    print("\n📊 Ultimate Efficiency Ratio V3.0 の特徴:")
    print("  🔬 量子強化ヒルベルト変換による瞬時解析")
    print("  🎯 量子適応カルマンフィルターによるノイズ除去")
    print("  🚀 5次元ハイパー効率率による超精密測定")
    print("  💡 量子もつれ効果による相関関係の完全捕捉")
    print("  ⚡ 超低遅延・超高精度・実用性重視の設計")


if __name__ == "__main__":
    main() 