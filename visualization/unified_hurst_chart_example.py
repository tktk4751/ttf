#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合ハースト指数チャートの使用例

実際の相場データを使ってハースト指数の3つの手法を比較・評価します：
1. R/S法 (Rescaled Range Statistics)
2. DFA法 (Detrended Fluctuation Analysis)
3. ウェーブレット法 (Daubechies Wavelet Method)
"""

from visualization.unified_hurst_chart import UnifiedHurstChart

def example_unified_hurst_chart():
    """統合ハースト指数チャートの実行例"""
    
    # UnifiedHurstChartインスタンスを作成
    chart = UnifiedHurstChart()
    
    # config.yamlからデータを読み込み
    try:
        chart.load_data_from_config('config.yaml')
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        print("config.yamlが存在し、適切な設定があることを確認してください。")
        return
    
    # 複数の統合ハースト指数設定（異なるウィンドウサイズと価格ソース）
    hurst_configs = [
        # 短期：高頻度変動の捉え方
        {'window_size': 50, 'src_type': 'close', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_Short_Close'},
        
        # 中期：バランス型（一般的な分析）
        {'window_size': 80, 'src_type': 'hlc3', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_Medium_HLC3'},
        
        # 長期：長期トレンドの持続性
        {'window_size': 120, 'src_type': 'hl2', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_Long_HL2'}
    ]
    
    # インジケーターを計算
    chart.calculate_indicators(hurst_configs=hurst_configs)
    
    # チャートを描画（画像保存とmatplotlib表示の両方）
    chart.plot(
        title="統合ハースト指数分析 - 3手法比較",
        start_date=None,  # 全期間表示
        end_date=None,
        show_volume=True,
        show_individual_methods=True,  # 個別手法も表示
        savefig="unified_hurst_analysis.png"  # 画像保存もする場合
    )
    
    print("\n=== 統合ハースト指数分析完了 ===")
    print("チャートがmatplotlibで表示され、'unified_hurst_analysis.png' に保存されました")
    
    # 分析のポイント
    print("\n=== ハースト指数分析のポイント ===")
    print("1. コンセンサスハースト指数（上段パネル）:")
    print("   - 3つの手法を統合した最終的なハースト指数")
    print("   - H < 0.5: 反持続性（平均回帰傾向）")
    print("   - H = 0.5: ランダムウォーク（記憶なし）")
    print("   - H > 0.5: 持続性（トレンド継続傾向）")
    
    print("\n2. 個別手法比較（中段パネル）:")
    print("   - R/S法（青線）: 古典的手法、長期記憶検出に強い")
    print("   - DFA法（緑破線）: トレンド除去、金融時系列に適している")
    print("   - ウェーブレット法（赤一点鎖線）: 周波数領域分析、複雑な構造検出")
    
    print("\n3. 信頼度スコア（下段パネル）:")
    print("   - 0.7以上: 高信頼度（結果を信頼できる）")
    print("   - 0.5-0.7: 中信頼度（注意深く解釈）")
    print("   - 0.5以下: 低信頼度（慎重に扱う）")
    
    print("\n4. 持続性レジーム（最下段パネル）:")
    print("   - +1: 強い持続性（強いトレンド継続）")
    print("   - +0.5: 弱い持続性（弱いトレンド）")
    print("   - 0: ランダムウォーク（方向性なし）")
    print("   - -0.5: 弱い反持続性（弱い平均回帰）")
    print("   - -1: 強い反持続性（強い平均回帰）")
    
    print("\n5. 実用的な使い方:")
    print("   - トレンドフォロー戦略: H > 0.55の時に有効")
    print("   - 平均回帰戦略: H < 0.45の時に有効")
    print("   - 複数時間軸の組み合わせで戦略強度を判断")
    print("   - 信頼度スコアが高い期間の結果を重視")

def compare_methods_only():
    """手法比較に特化した分析例"""
    print("\n=== 手法比較特化分析 ===")
    
    chart = UnifiedHurstChart()
    
    try:
        chart.load_data_from_config('config.yaml')
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        return
    
    # 単一設定で手法比較に特化
    single_config = [
        {'window_size': 80, 'src_type': 'hlc3', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Method_Comparison'}
    ]
    
    chart.calculate_indicators(hurst_configs=single_config)
    
    # 手法比較に特化したチャート表示
    chart.plot(
        title="ハースト指数手法比較 - R/S vs DFA vs ウェーブレット",
        show_volume=False,  # 出来高非表示でスペースを節約
        show_individual_methods=True,
        savefig="hurst_methods_comparison.png"
    )
    
    print("手法比較チャートが 'hurst_methods_comparison.png' に保存されました")

if __name__ == "__main__":
    # 基本的な統合ハースト指数分析
    example_unified_hurst_chart()
    
    # 手法比較に特化した分析
    compare_methods_only()