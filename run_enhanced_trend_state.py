#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from enhanced_trend_state_chart import EnhancedTrendStateChart


def main():
    """
    Enhanced Trend State 分析の実行スクリプト
    """
    parser = argparse.ArgumentParser(
        description='Enhanced Trend State インジケーターによる市場分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python run_enhanced_trend_state.py
  
  # カスタム設定ファイルを使用
  python run_enhanced_trend_state.py --config my_config.yaml
  
  # チャートを表示せずに保存のみ
  python run_enhanced_trend_state.py --no-show
  
  # カスタムパラメータで実行
  python run_enhanced_trend_state.py --period 25 --threshold 0.65
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='設定ファイルのパス (デフォルト: config.yaml)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='チャートを表示せずに保存のみ実行'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        help='基本期間をオーバーライド'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='トレンド判定閾値をオーバーライド (0.5-0.8推奨)'
    )
    
    parser.add_argument(
        '--no-dynamic',
        action='store_true',
        help='動的期間調整を無効化'
    )
    
    parser.add_argument(
        '--no-volatility-adj',
        action='store_true',
        help='ボラティリティ調整を無効化'
    )
    
    parser.add_argument(
        '--no-atr-smoothing',
        action='store_true',
        help='ATRスムージングを無効化'
    )
    
    parser.add_argument(
        '--er-weight',
        type=float,
        help='効率比の重みをオーバーライド (0.0-1.0)'
    )
    
    parser.add_argument(
        '--chop-weight',
        type=float,
        help='チョピネス指数の重みをオーバーライド (0.0-1.0)'
    )
    
    parser.add_argument(
        '--no-components',
        action='store_true',
        help='コンポーネント表示を無効化'
    )
    
    args = parser.parse_args()
    
    try:
        # チャート分析器の初期化
        print("Enhanced Trend State 分析を開始します...")
        chart_analyzer = EnhancedTrendStateChart(args.config)
        
        # パラメータのオーバーライド
        if args.period:
            chart_analyzer.ets_indicator.base_period = args.period
            print(f"基本期間を {args.period} にオーバーライドしました")
        
        if args.threshold:
            if not 0.3 <= args.threshold <= 0.9:
                print("エラー: 閾値は0.3-0.9の範囲で指定してください")
                sys.exit(1)
            chart_analyzer.ets_indicator.threshold = args.threshold
            print(f"閾値を {args.threshold} にオーバーライドしました")
        
        if args.no_dynamic:
            chart_analyzer.ets_indicator.use_dynamic_period = False
            print("動的期間調整を無効化しました")
        
        if args.no_volatility_adj:
            chart_analyzer.ets_indicator.volatility_adjustment = False
            print("ボラティリティ調整を無効化しました")
        
        if args.no_atr_smoothing:
            chart_analyzer.ets_indicator.atr_smoothing = False
            print("ATRスムージングを無効化しました")
        
        if args.er_weight:
            if not 0.0 <= args.er_weight <= 1.0:
                print("エラー: 効率比の重みは0.0-1.0の範囲で指定してください")
                sys.exit(1)
            chart_analyzer.ets_indicator.er_weight = args.er_weight
            # チョピネスの重みも自動調整
            chart_analyzer.ets_indicator.chop_weight = 1.0 - args.er_weight
            print(f"効率比の重みを {args.er_weight} にオーバーライドしました")
            print(f"チョピネス指数の重みを {1.0 - args.er_weight} に自動調整しました")
        
        if args.chop_weight:
            if not 0.0 <= args.chop_weight <= 1.0:
                print("エラー: チョピネス指数の重みは0.0-1.0の範囲で指定してください")
                sys.exit(1)
            chart_analyzer.ets_indicator.chop_weight = args.chop_weight
            # 効率比の重みも自動調整
            chart_analyzer.ets_indicator.er_weight = 1.0 - args.chop_weight
            print(f"チョピネス指数の重みを {args.chop_weight} にオーバーライドしました")
            print(f"効率比の重みを {1.0 - args.chop_weight} に自動調整しました")
        
        # 設定の更新
        if args.no_show:
            chart_analyzer.config['enhanced_trend_state']['save_chart'] = True
            print("チャート表示を無効化し、保存のみ実行します")
        
        if args.no_components:
            chart_analyzer.config['enhanced_trend_state']['show_components'] = False
            print("コンポーネント表示を無効化しました")
        
        # 分析の実行
        chart_analyzer.run_analysis_optimized(show_chart=not args.no_show)
        
        print("\n✓ 分析が完了しました")
        
        # インジケーター設定の表示
        print(f"\n=== インジケーター設定 ===")
        print(f"基本期間: {chart_analyzer.ets_indicator.base_period}")
        print(f"閾値: {chart_analyzer.ets_indicator.threshold}")
        print(f"動的期間: {'有効' if chart_analyzer.ets_indicator.use_dynamic_period else '無効'}")
        print(f"ボラティリティ調整: {'有効' if chart_analyzer.ets_indicator.volatility_adjustment else '無効'}")
        print(f"ATRスムージング: {'有効' if chart_analyzer.ets_indicator.atr_smoothing else '無効'}")
        print(f"効率比重み: {chart_analyzer.ets_indicator.er_weight:.2f}")
        print(f"チョピネス重み: {chart_analyzer.ets_indicator.chop_weight:.2f}")
        print(f"サイクル検出器: {chart_analyzer.ets_indicator.detector_type}")
        print(f"サイクル範囲: {chart_analyzer.ets_indicator.min_cycle}-{chart_analyzer.ets_indicator.max_cycle}")
        
    except KeyboardInterrupt:
        print("\n中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()