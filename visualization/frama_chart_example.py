#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRAMA Chart Example
実際の相場データを使ったFRAMAチャートの例
"""

from visualization.frama_chart import FRAMAChart

def example_frama_chart():
    """FRAMA Chart の実行例"""
    
    # FRAMAChartインスタンスを作成
    chart = FRAMAChart()
    
    # config.yamlからデータを読み込み
    try:
        chart.load_data_from_config('config.yaml')
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        print("config.yamlが存在し、適切な設定があることを確認してください。")
        return
    
    # 複数のFRAMA設定（PineScript版対応）
    frama_configs = [
        # 短期：高速適応（トレンドフォロー向け）
        {'period': 16, 'src_type': 'hl2', 'fc': 1, 'sc': 100, 'name': 'FRAMA_Fast'},
        
        # 中期：バランス型（一般的な用途、PineScriptデフォルト）
        {'period': 32, 'src_type': 'hlc3', 'fc': 1, 'sc': 198, 'name': 'FRAMA_Medium'},
        
        # 長期：低速適応（ノイズ除去重視）
        {'period': 48, 'src_type': 'close', 'fc': 1, 'sc': 300, 'name': 'FRAMA_Slow'}
    ]
    
    # インジケーターを計算
    chart.calculate_indicators(frama_configs=frama_configs)
    
    # チャートを描画（画像保存とmatplotlib表示の両方）
    chart.plot(
        title="FRAMA - Fractal Adaptive Moving Average Analysis",
        start_date=None,  # 全期間表示
        end_date=None,
        show_volume=True,
        show_fractal_dimension=True,
        show_alpha=True,
        savefig="frama_analysis.png"  # 画像保存もする場合
    )
    
    print("\n=== FRAMA分析完了 ===")
    print("チャートがmatplotlibで表示され、'frama_analysis.png' に保存されました")
    
    # 分析のポイント
    print("\n=== FRAMA分析のポイント ===")
    print("1. FRAMA値（青・赤・緑線）:")
    print("   - 価格に追従する適応的移動平均")
    print("   - トレンド時は高速、レンジ時は低速で反応")
    print("   - 期間が短いほど敏感、長いほど安定")
    
    print("\n2. フラクタル次元（中段パネル）:")
    print("   - 1.0に近い：価格変動が単調（トレンド）")
    print("   - 2.0に近い：価格変動が複雑（レンジ）")
    print("   - 市場の複雑さを数値化")
    
    print("\n3. アルファ値（下段パネル）:")
    print("   - 1.0に近い：高速適応（トレンド追従）")
    print("   - 0.01に近い：低速適応（ノイズ除去）")
    print("   - FRAMAの適応係数")
    
    print("\n4. 使用方法:")
    print("   - クロスオーバー：FRAMAと価格のクロスで売買シグナル")
    print("   - 傾き：FRAMAの傾きでトレンド方向を判断")
    print("   - 複数期間比較：短期と長期FRAMAの位置関係で強度判断")

if __name__ == "__main__":
    example_frama_chart()