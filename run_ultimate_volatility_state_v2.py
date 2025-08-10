#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
from ultimate_volatility_state_chart_v2 import UltimateVolatilityStateChartV2


def main():
    """
    Ultimate Volatility State V2 分析の実行スクリプト
    """
    parser = argparse.ArgumentParser(
        description='Ultimate Volatility State V2 インジケーターによる超高精度市場分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V2 の新機能:
  - ウェーブレット分解による多時間軸ボラティリティ分析
  - スペクトルエントロピーによる周波数領域分析  
  - Hurst指数によるフラクタル特性分析
  - GARCH風ボラティリティモデリング
  - 信頼度ベースの適応的判定
  - 14の多角的指標による総合判定

使用例:
  # V2デフォルト設定で実行
  python run_ultimate_volatility_state_v2.py
  
  # チャートを表示せずに保存のみ
  python run_ultimate_volatility_state_v2.py --no-show
  
  # 高精度モード（厳格な信頼度設定）
  python run_ultimate_volatility_state_v2.py --high-precision
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
        '--confidence-threshold',
        type=float,
        help='信頼度閾値をオーバーライド (0.0-1.0)'
    )
    
    parser.add_argument(
        '--high-precision',
        action='store_true',
        help='高精度モード（厳格な設定）を有効化'
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='高速モード（短縮された計算期間）'
    )
    
    args = parser.parse_args()
    
    try:
        # V2チャート分析器の初期化
        print("Ultimate Volatility State V2 分析を開始します...")
        chart_analyzer = UltimateVolatilityStateChartV2(args.config)
        
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
        
        if args.confidence_threshold:
            if not 0.0 <= args.confidence_threshold <= 1.0:
                print("エラー: 信頼度閾値は0.0-1.0の範囲で指定してください")
                sys.exit(1)
            chart_analyzer.uvs_indicator.confidence_threshold = args.confidence_threshold
            print(f"信頼度閾値を {args.confidence_threshold} にオーバーライドしました")
        
        # 高精度モード
        if args.high_precision:
            chart_analyzer.uvs_indicator.confidence_threshold = 0.8
            chart_analyzer.uvs_indicator.threshold = 0.45  # より保守的
            chart_analyzer.uvs_indicator.smoother_period = 2  # より少ないスムージング
            print("高精度モードを有効化しました（信頼度閾値: 0.8, 判定閾値: 0.45）")
        
        # 高速モード
        if args.fast_mode:
            chart_analyzer.uvs_indicator.period = 15
            chart_analyzer.uvs_indicator.zscore_period = 30
            print("高速モードを有効化しました（短縮された計算期間）")
        
        # 設定の更新
        if args.no_show:
            chart_analyzer.config['ultimate_volatility_state']['save_chart'] = True
            print("チャート表示を無効化し、保存のみ実行します")
        
        # V2分析の実行
        chart_analyzer.run_analysis_v2(show_chart=not args.no_show)
        
        print("\n✓ V2分析が完了しました")
        
        # 最終的な精度評価
        if chart_analyzer.uvs_indicator._result_cache:
            result = list(chart_analyzer.uvs_indicator._result_cache.values())[-1]
            high_confidence_ratio = np.sum(result.confidence > 0.7) / len(result.confidence)
            print(f"📊 分析精度評価: 高信頼度判定率 {high_confidence_ratio:.1%}")
            
            if high_confidence_ratio > 0.8:
                print("🟢 優秀な分析精度です")
            elif high_confidence_ratio > 0.6:
                print("🟡 良好な分析精度です")
            else:
                print("🔴 分析精度に改善の余地があります")
        else:
            print("⚠️ 結果キャッシュが空のため、精度評価をスキップしました")
        
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