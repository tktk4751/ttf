#!/usr/bin/env python3
"""
HyperFRAMAChannelフィルタリング機能のテストスクリプト
"""

import numpy as np
import pandas as pd
from strategies.implementations.hyper_frama_channel.strategy import HyperFRAMAChannelStrategy
from strategies.implementations.hyper_frama_channel.signal_generator import FilterType

def create_test_data(length: int = 1000) -> pd.DataFrame:
    """テスト用のダミーデータを生成"""
    np.random.seed(42)
    
    # トレンドのあるデータを生成
    trend = np.linspace(100, 200, length)
    noise = np.random.normal(0, 5, length)
    
    close = trend + noise
    high = close + np.random.uniform(0, 3, length)
    low = close - np.random.uniform(0, 3, length)
    open_price = close + np.random.uniform(-2, 2, length)
    volume = np.random.uniform(1000, 5000, length)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

def test_filter_types():
    """各フィルタータイプのテスト"""
    print("=== HyperFRAMAChannelフィルタリング機能のテスト ===\n")
    
    # テストデータ生成
    print("ダミーデータ生成中...")
    data = create_test_data(500)
    print(f"データ生成完了: {len(data)} データポイント\n")
    
    # 各フィルタータイプのテスト
    filter_types = [
        FilterType.NONE,
        FilterType.HYPER_ER,
        FilterType.HYPER_TREND_INDEX,
        FilterType.HYPER_ADX,
        FilterType.CONSENSUS
    ]
    
    results = {}
    
    for filter_type in filter_types:
        print(f"=== {filter_type.value.upper()} フィルターテスト ===")
        
        try:
            # ストラテジー初期化
            strategy = HyperFRAMAChannelStrategy(
                filter_type=filter_type,
                # フィルター固有パラメータ（テスト用に小さい値）
                filter_hyper_er_period=8,
                filter_hyper_er_midline_period=50,
                filter_hyper_trend_index_period=8,
                filter_hyper_trend_index_midline_period=50,
                filter_hyper_adx_period=8,
                filter_hyper_adx_midline_period=50
            )
            
            # シグナル計算
            print("シグナル計算中...")
            entry_signals = strategy.generate_entry(data)
            channel_signals = strategy.get_channel_signals(data)
            filter_signals = strategy.get_filter_signals(data)
            
            # 統計情報
            long_signals = np.sum(entry_signals == 1)
            short_signals = np.sum(entry_signals == -1)
            channel_long = np.sum(channel_signals == 1)
            channel_short = np.sum(channel_signals == -1)
            filter_positive = np.sum(filter_signals == 1)
            filter_negative = np.sum(filter_signals == -1)
            
            results[filter_type.value] = {
                'entry_long': long_signals,
                'entry_short': short_signals,
                'channel_long': channel_long,
                'channel_short': channel_short,
                'filter_positive': filter_positive,
                'filter_negative': filter_negative
            }
            
            print(f"エントリーシグナル - ロング: {long_signals}, ショート: {short_signals}")
            print(f"チャネルシグナル - ロング: {channel_long}, ショート: {channel_short}")
            print(f"フィルターシグナル - ポジティブ: {filter_positive}, ネガティブ: {filter_negative}")
            
            # エグジットシグナルテスト
            exit_long = strategy.generate_exit(data, 1, -1)
            exit_short = strategy.generate_exit(data, -1, -1)
            print(f"エグジットシグナルテスト - ロング: {exit_long}, ショート: {exit_short}")
            
            # フィルター詳細情報（該当する場合）
            if filter_type != FilterType.NONE:
                filter_details = strategy.get_filter_details(data)
                print(f"フィルター詳細キー: {list(filter_details.keys())}")
            
            print(f"{filter_type.value} フィルターテスト完了✓\n")
            
        except Exception as e:
            print(f"{filter_type.value} フィルターテスト中にエラー: {str(e)}")
            print(f"{filter_type.value} フィルターテスト失敗✗\n")
            results[filter_type.value] = None
    
    return results

def test_optimization_params():
    """最適化パラメータのテスト"""
    print("=== 最適化パラメータのテスト ===")
    
    try:
        import optuna
        
        # ダミートライアルの作成
        study = optuna.create_study()
        trial = study.ask()
        
        # 最適化パラメータ生成
        params = HyperFRAMAChannelStrategy.create_optimization_params(trial)
        print(f"最適化パラメータ数: {len(params)}")
        
        # フィルター関連パラメータの確認
        filter_params = {k: v for k, v in params.items() if 'filter' in k}
        print(f"フィルター関連パラメータ数: {len(filter_params)}")
        print(f"フィルタータイプ: {params.get('filter_type')}")
        
        # パラメータ変換テスト
        strategy_params = HyperFRAMAChannelStrategy.convert_params_to_strategy_format(params)
        print(f"変換されたパラメータ数: {len(strategy_params)}")
        
        # フィルター付きストラテジー初期化テスト
        strategy = HyperFRAMAChannelStrategy(**strategy_params)
        print(f"フィルター付きストラテジー初期化成功✓")
        print(f"使用フィルター: {strategy._parameters['filter_type'].value}")
        
        print("最適化パラメータテスト完了✓\n")
        return True
        
    except Exception as e:
        print(f"最適化パラメータテスト中にエラー: {str(e)}")
        print("最適化パラメータテスト失敗✗\n")
        return False

def test_advanced_metrics():
    """高度なメトリクステスト"""
    print("=== 高度なメトリクステスト ===")
    
    try:
        # コンセンサスフィルター付きストラテジー
        strategy = HyperFRAMAChannelStrategy(
            filter_type=FilterType.CONSENSUS,
            filter_hyper_er_period=8,
            filter_hyper_trend_index_period=8,
            filter_hyper_adx_period=8
        )
        
        data = create_test_data(200)
        
        # 高度なメトリクス取得
        metrics = strategy.get_advanced_metrics(data)
        print(f"メトリクス項目数: {len(metrics)}")
        print(f"メトリクス項目: {list(metrics.keys())}")
        
        # フィルター詳細確認
        if 'hyper_er_signals' in metrics:
            print(f"HyperERシグナル数: {len(metrics['hyper_er_signals'])}")
        if 'hyper_trend_index_signals' in metrics:
            print(f"HyperTrendIndexシグナル数: {len(metrics['hyper_trend_index_signals'])}")
        if 'hyper_adx_signals' in metrics:
            print(f"HyperADXシグナル数: {len(metrics['hyper_adx_signals'])}")
        
        print("高度なメトリクステスト完了✓\n")
        return True
        
    except Exception as e:
        print(f"高度なメトリクステスト中にエラー: {str(e)}")
        print("高度なメトリクステスト失敗✗\n")
        return False

def main():
    """メイン関数"""
    print("HyperFRAMAChannelフィルタリング機能のテストを開始します\n")
    
    # フィルタータイプテスト
    filter_results = test_filter_types()
    
    # 最適化パラメータテスト
    opt_result = test_optimization_params()
    
    # 高度なメトリクステスト
    metrics_result = test_advanced_metrics()
    
    # 結果まとめ
    print("==================================================")
    print("テスト結果まとめ")
    print("==================================================")
    
    success_count = 0
    total_count = len(filter_results) + 2
    
    for filter_type, result in filter_results.items():
        if result is not None:
            print(f"{filter_type.upper()} フィルター        : 成功✓")
            success_count += 1
        else:
            print(f"{filter_type.upper()} フィルター        : 失敗✗")
    
    if opt_result:
        print(f"最適化パラメータ          : 成功✓")
        success_count += 1
    else:
        print(f"最適化パラメータ          : 失敗✗")
    
    if metrics_result:
        print(f"高度なメトリクス          : 成功✓")
        success_count += 1
    else:
        print(f"高度なメトリクス          : 失敗✗")
    
    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\nすべてのテストが成功しました！🎉")
        print("\nHyperFRAMAChannelフィルタリング機能は正常に動作しています。")
        print("最適化とバックテストの準備ができました。")
    else:
        print(f"\n{total_count - success_count}個のテストが失敗しました。")
        print("エラーメッセージを確認して修正してください。")
    
    # フィルター効果の比較
    if all(result is not None for result in filter_results.values()):
        print("\n==================================================")
        print("フィルター効果の比較")
        print("==================================================")
        
        for filter_type, result in filter_results.items():
            ratio = (result['entry_long'] + result['entry_short']) / (result['channel_long'] + result['channel_short'] + 1e-8) * 100
            print(f"{filter_type.upper():20}: チャネル信号から{ratio:.1f}%が最終エントリーに")

if __name__ == "__main__":
    main()