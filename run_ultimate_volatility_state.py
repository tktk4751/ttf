#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from ultimate_volatility_state_chart import UltimateVolatilityStateChart


def main():
    """
    Ultimate Volatility State 分析の実行スクリプト
    """
    parser = argparse.ArgumentParser(
        description='Ultimate Volatility State インジケーターによる市場分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python run_ultimate_volatility_state.py
  
  # カスタム設定ファイルを使用
  python run_ultimate_volatility_state.py --config my_config.yaml
  
  # チャートを表示せずに保存のみ
  python run_ultimate_volatility_state.py --no-show
  
  # カスタムパラメータで実行
  python run_ultimate_volatility_state.py --period 30 --threshold 0.6
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
        help='ボラティリティ閾値をオーバーライド (0.0-1.0)'
    )
    
    parser.add_argument(
        '--zscore-period',
        type=int,
        help='Z-Score期間をオーバーライド'
    )
    
    parser.add_argument(
        '--adaptive-threshold',
        action='store_true',
        help='適応的閾値調整を有効化'
    )
    
    parser.add_argument(
        '--no-components',
        action='store_true',
        help='コンポーネント表示を無効化'
    )
    
    args = parser.parse_args()
    
    try:
        # チャート分析器の初期化
        print("Ultimate Volatility State 分析を開始します...")
        chart_analyzer = UltimateVolatilityStateChart(args.config)
        
        # パラメータのオーバーライド
        if args.period:
            chart_analyzer.uvs_indicator.period = args.period
            print(f"期間を {args.period} にオーバーライドしました")
        
        if args.threshold:
            if not 0.0 <= args.threshold <= 1.0:
                print("エラー: 閾値は0.0-1.0の範囲で指定してください")
                sys.exit(1)
            chart_analyzer.uvs_indicator.threshold = args.threshold
            print(f"閾値を {args.threshold} にオーバーライドしました")
        
        if args.zscore_period:
            chart_analyzer.uvs_indicator.zscore_period = args.zscore_period
            print(f"Z-Score期間を {args.zscore_period} にオーバーライドしました")
        
        if args.adaptive_threshold:
            chart_analyzer.uvs_indicator.adaptive_threshold = True
            print("適応的閾値調整を有効化しました")
        
        # 設定の更新
        if args.no_show:
            chart_analyzer.config['ultimate_volatility_state']['save_chart'] = True
            print("チャート表示を無効化し、保存のみ実行します")
        
        if args.no_components:
            chart_analyzer.config['ultimate_volatility_state']['show_components'] = False
            print("コンポーネント表示を無効化しました")
        
        # 分析の実行
        chart_analyzer.run_analysis_optimized(show_chart=not args.no_show)
        
        print("\n✓ 分析が完了しました")
        
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