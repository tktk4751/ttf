#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperFRAMAChannelストラテジーのテスト用スクリプト
"""

import numpy as np
import pandas as pd
import yaml
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.binance_data_source import BinanceDataSource
from strategies.implementations.hyper_frama_channel.strategy import HyperFRAMAChannelStrategy
from strategies.implementations.hyper_frama_channel.signal_generator import HyperFRAMAChannelSignalGenerator


def load_config():
    """設定ファイルを読み込み"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        return None


def load_test_data(config):
    """テスト用データを読み込み"""
    try:
        # Binanceデータソースからデータを取得
        data_source = BinanceDataSource()
        
        # 設定からBinanceデータ設定を使用
        if 'binance_data' in config:
            data_config = config['binance_data']
            symbol = data_config['symbol'] + 'USDT'  # SOL -> SOLUSDT
            timeframe = data_config['timeframe']
        else:
            # フォールバック - dataセクションを使用
            data_config = config['data']
            symbol = data_config['symbol']
            timeframe = data_config['timeframe']
        
        print(f"データを読み込み中: {symbol} ({timeframe})")
        
        # データを取得
        data = data_source.load_data(symbol, timeframe)
        
        if data is None or len(data) == 0:
            print("データの読み込みに失敗しました")
            return None
        
        print(f"データ読み込み完了: {len(data)} データポイント")
        print(f"データ範囲: {data.index[0]} - {data.index[-1]}")
        
        return data
    
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None


def test_signal_generator():
    """シグナルジェネレーターのテスト"""
    print("\n=== HyperFRAMAChannelSignalGeneratorのテスト ===")
    
    try:
        # 設定読み込み
        config = load_config()
        if config is None:
            return False
        
        # テストデータ読み込み
        data = load_test_data(config)
        if data is None:
            print("実データを使用できないため、ダミーデータでテストします")
            # ダミーデータ生成
            import numpy as np
            np.random.seed(42)
            n_points = 1000
            data = pd.DataFrame({
                'open': 100 + np.random.randn(n_points).cumsum() * 0.5,
                'high': None,
                'low': None,
                'close': None
            })
            data['close'] = data['open'] + np.random.randn(n_points) * 0.1
            data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(n_points) * 0.05)
            data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(n_points) * 0.05)
            print(f"ダミーデータ生成完了: {len(data)} データポイント")
        
        # シグナルジェネレーターの初期化（デフォルトパラメータ）
        signal_generator = HyperFRAMAChannelSignalGenerator()
        
        print("シグナル計算中...")
        
        # エントリーシグナルのテスト
        entry_signals = signal_generator.get_entry_signals(data)
        print(f"エントリーシグナル: 配列サイズ={len(entry_signals)}")
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        no_signals = np.sum(entry_signals == 0)
        
        print(f"- ロングシグナル: {long_signals}")
        print(f"- ショートシグナル: {short_signals}")
        print(f"- シグナルなし: {no_signals}")
        
        # エグジットシグナルのテスト
        exit_test_result = signal_generator.get_exit_signals(data, position=1, index=-1)
        print(f"エグジットシグナルテスト (ロング): {exit_test_result}")
        
        # チャネル値の取得テスト
        midline, upper_band, lower_band = signal_generator.get_channel_values(data)
        print(f"チャネル値: 中心線={len(midline)}, 上限={len(upper_band)}, 下限={len(lower_band)}")
        
        # シンプル版ではFRAMA値の直接取得は非対応のため、この部分は削除
        print("シンプル版：FRAMA値の直接取得は非対応")
        
        # ソース価格の取得テスト
        source_price = signal_generator.get_source_price(data)
        print(f"ソース価格: {len(source_price)}")
        
        print("シグナルジェネレーターテスト完了✓")
        return True
        
    except Exception as e:
        print(f"シグナルジェネレーターテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy():
    """ストラテジーのテスト"""
    print("\n=== HyperFRAMAChannelStrategyのテスト ===")
    
    try:
        # 設定読み込み
        config = load_config()
        if config is None:
            return False
        
        # テストデータ読み込み
        data = load_test_data(config)
        if data is None:
            print("実データを使用できないため、ダミーデータでテストします")
            # ダミーデータ生成
            import numpy as np
            np.random.seed(42)
            n_points = 1000
            data = pd.DataFrame({
                'open': 100 + np.random.randn(n_points).cumsum() * 0.5,
                'high': None,
                'low': None,
                'close': None
            })
            data['close'] = data['open'] + np.random.randn(n_points) * 0.1
            data['high'] = np.maximum(data['open'], data['close']) + np.abs(np.random.randn(n_points) * 0.05)
            data['low'] = np.minimum(data['open'], data['close']) - np.abs(np.random.randn(n_points) * 0.05)
            print(f"ダミーデータ生成完了: {len(data)} データポイント")
        
        # ストラテジーの初期化（デフォルトパラメータ）
        strategy = HyperFRAMAChannelStrategy()
        
        print("ストラテジー計算中...")
        
        # エントリーシグナルのテスト
        entry_signals = strategy.generate_entry(data)
        print(f"エントリーシグナル: 配列サイズ={len(entry_signals)}")
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        no_signals = np.sum(entry_signals == 0)
        
        print(f"- ロングシグナル: {long_signals}")
        print(f"- ショートシグナル: {short_signals}")
        print(f"- シグナルなし: {no_signals}")
        
        # エグジットシグナルのテスト
        exit_test_long = strategy.generate_exit(data, position=1, index=-1)
        exit_test_short = strategy.generate_exit(data, position=-1, index=-1)
        print(f"エグジットシグナルテスト - ロング: {exit_test_long}, ショート: {exit_test_short}")
        
        print("ストラテジーテスト完了✓")
        return True
        
    except Exception as e:
        print(f"ストラテジーテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_params():
    """最適化パラメータのテスト"""
    print("\n=== 最適化パラメータのテスト ===")
    
    try:
        import optuna
        
        # ダミートライアルの作成
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        
        # 最適化パラメータの生成テスト
        params = HyperFRAMAChannelStrategy.create_optimization_params(trial)
        print(f"最適化パラメータ数: {len(params)}")
        
        # パラメータ例の表示
        param_sample = {k: v for i, (k, v) in enumerate(params.items()) if i < 5}
        print(f"パラメータ例: {param_sample}")
        
        # ストラテジー形式への変換テスト
        strategy_params = HyperFRAMAChannelStrategy.convert_params_to_strategy_format(params)
        print(f"変換されたパラメータ数: {len(strategy_params)}")
        
        # 変換されたパラメータでストラテジーの初期化テスト
        strategy = HyperFRAMAChannelStrategy(**strategy_params)
        print("パラメータ変換によるストラテジー初期化成功✓")
        
        print("最適化パラメータテスト完了✓")
        return True
        
    except Exception as e:
        print(f"最適化パラメータテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト関数"""
    print("HyperFRAMAChannelストラテジーのテストを開始します")
    
    results = []
    
    # 各テストを実行
    results.append(("シグナルジェネレーター", test_signal_generator()))
    results.append(("ストラテジー", test_strategy()))
    results.append(("最適化パラメータ", test_optimization_params()))
    
    # 結果まとめ
    print("\n" + "="*50)
    print("テスト結果まとめ")
    print("="*50)
    
    success_count = 0
    for test_name, result in results:
        status = "成功✓" if result else "失敗✗"
        print(f"{test_name:<20}: {status}")
        if result:
            success_count += 1
    
    print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("\nすべてのテストが成功しました！🎉")
        print("\nHyperFRAMAChannelストラテジーは正常に動作しています。")
        print("最適化を実行する準備ができました。")
    else:
        print("\nいくつかのテストが失敗しました。")
        print("エラーメッセージを確認して修正してください。")


if __name__ == "__main__":
    main()