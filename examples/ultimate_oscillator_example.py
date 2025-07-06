#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ehlers' Ultimate Oscillator の使用例
John Ehlers' "Ultimate Oscillator" (Traders Tips 4/2025) に基づく実装のテスト
実際の相場データを使用したチャート描画
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import sys
import os
from datetime import datetime
import yaml

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.ultimate_oscillator import UltimateOscillator
from data.data_loader import DataLoader, CSVDataSource
from data.binance_data_source import BinanceDataSource
from data.data_processor import DataProcessor


def load_market_data(symbol: str = 'ETH/spot/4h', start_date: str = '2024-01-01', end_date: str = '2025-03-31') -> pd.DataFrame:
    """
    設定ファイルから実際の相場データを読み込む
    
    Args:
        symbol: シンボル（例: 'ETH/spot/4h'）
        start_date: 開始日
        end_date: 終了日
        
    Returns:
        pd.DataFrame: OHLC価格データ
    """
    try:
        print(f"相場データを読み込み中: {symbol}")
        
        # 設定ファイルの読み込み
        config_path = 'config.yaml'
        if not os.path.exists(config_path):
            print(f"設定ファイルが見つかりません: {config_path}")
            return None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        print("データを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 指定されたシンボルのデータを取得
        if symbol in processed_data:
            data = processed_data[symbol]
        else:
            # 指定されたシンボルが見つからない場合は最初のシンボルを使用
            first_symbol = next(iter(processed_data))
            data = processed_data[first_symbol]
            print(f"指定されたシンボル '{symbol}' が見つかりません。'{first_symbol}' を使用します。")
        
        if data is not None and len(data) > 0:
            print(f"データ読み込み成功: {len(data)}行")
            print(f"期間: {data.index[0]} - {data.index[-1]}")
            return data
        else:
            print("データ読み込みに失敗しました")
            return None
            
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ultimate_oscillator_real_data():
    """実際の相場データでのUltimate Oscillatorテスト"""
    print("=== 実際の相場データでのUltimate Oscillatorテスト ===")
    
    # 実際のデータを読み込み
    data = load_market_data('ETH/spot/4h', '2024-01-01', '2025-03-31')
    if data is None:
        print("データ読み込みに失敗しました")
        return None, None
    
    print(f"テストデータ: {len(data)}点")
    
    # 異なるパラメータでのテスト
    test_configs = [
        {'edge': 20, 'width': 2, 'rms_period': 50, 'name': '短期'},
        {'edge': 30, 'width': 2, 'rms_period': 100, 'name': '中期'},  # デフォルト
        {'edge': 40, 'width': 3, 'rms_period': 150, 'name': '長期'}
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- {config['name']}設定: edge={config['edge']}, width={config['width']}, rms={config['rms_period']} ---")
        
        # UltimateOscillator作成
        oscillator = UltimateOscillator(
            edge=config['edge'], 
            width=config['width'], 
            rms_period=config['rms_period'],
            src_type='hlc3'  # 実際のデータではhlc3を使用
        )
        
        # 計算実行
        result = oscillator.calculate(data)
        results[config['name']] = result
        
        print(f"計算結果: {len(result.values)}点")
        print(f"有効な値: {np.sum(~np.isnan(result.values))}点")
        
        if len(result.values) > 0:
            print(f"最終値: {result.values[-1]:.4f}")
            print(f"最大値: {np.nanmax(result.values):.4f}")
            print(f"最小値: {np.nanmin(result.values):.4f}")
            print(f"標準偏差: {np.nanstd(result.values):.4f}")
            
            # ゼロクロス回数
            valid_values = result.values[~np.isnan(result.values)]
            if len(valid_values) > 1:
                zero_crosses = np.sum(np.diff(np.sign(valid_values)) != 0)
                print(f"ゼロクロス回数: {zero_crosses}")
    
    return data, results


def plot_ultimate_oscillator_real_data(data: pd.DataFrame, results: dict):
    """実際の相場データでのUltimate Oscillatorチャート作成"""
    print("\n=== Ultimate Oscillator チャート作成（実際の相場データ） ===")
    
    try:
        # プロット設定
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        
        # 価格チャート（ローソク足）
        ax1 = axes[0]
        
        # データをmplfinance用に整形
        plot_data = data.copy()
        plot_data.index = pd.to_datetime(plot_data.index)
        
        # ローソク足チャート
        mpf.plot(plot_data, type='candle', style='charles', 
                ax=ax1, volume=False,
                datetime_format='%Y-%m-%d', show_nontrading=False)
        ax1.set_title('SOL/USDT 4H ローソク足チャート', fontsize=14, fontweight='bold')
        
        # Ultimate Oscillator（複数設定）
        ax2 = axes[1]
        colors = ['red', 'blue', 'green']
        
        for i, (name, result) in enumerate(results.items()):
            ax2.plot(result.values, label=f'{name}設定', color=colors[i], linewidth=1.5, alpha=0.8)
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Overbought (+2)')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Oversold (-2)')
        ax2.set_title('Ultimate Oscillator（複数設定比較）', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('Oscillator Value')
        
        # 信号分析
        ax3 = axes[2]
        
        # 中期設定の結果を使用
        mid_result = results.get('中期')
        if mid_result is not None:
            # ゼロクロス信号
            values = mid_result.values
            buy_signals = []
            sell_signals = []
            
            for i in range(1, len(values)):
                if not (np.isnan(values[i]) or np.isnan(values[i-1])):
                    # 買いシグナル（負から正へのクロス）
                    if values[i-1] < 0 and values[i] > 0:
                        buy_signals.append(i)
                    # 売りシグナル（正から負へのクロス）
                    elif values[i-1] > 0 and values[i] < 0:
                        sell_signals.append(i)
            
            # オシレーター値とシグナル
            ax3.plot(values, label='Ultimate Oscillator', color='purple', linewidth=2)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # シグナルポイントをプロット
            if buy_signals:
                ax3.scatter(buy_signals, values[buy_signals], color='green', s=50, marker='^', 
                           label=f'Buy Signal ({len(buy_signals)})', zorder=5)
            if sell_signals:
                ax3.scatter(sell_signals, values[sell_signals], color='red', s=50, marker='v', 
                           label=f'Sell Signal ({len(sell_signals)})', zorder=5)
            
            ax3.set_title('Ultimate Oscillator 信号分析', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylabel('Oscillator Value')
        
        # ハイパスフィルター成分
        ax4 = axes[3]
        
        # 中期設定のオシレーターを使用
        if mid_result is not None:
            oscillator = UltimateOscillator(edge=30, width=2, rms_period=100, src_type='hlc3')
            oscillator.calculate(data)  # 再計算してハイパス成分を取得
            
            hp_short, hp_long = oscillator.get_highpass_components()
            if hp_short is not None and hp_long is not None:
                ax4.plot(hp_short, label='短期ハイパスフィルター', color='blue', alpha=0.7, linewidth=1)
                ax4.plot(hp_long, label='長期ハイパスフィルター', color='orange', alpha=0.7, linewidth=1)
                ax4.plot(mid_result.signals, label='信号（差分）', color='purple', linewidth=2)
                ax4.set_title('ハイパスフィルター成分', fontsize=14, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylabel('Filter Value')
        
        plt.tight_layout()
        
        # ファイル保存
        output_path = os.path.join(os.path.dirname(__file__), 'output', 'ultimate_oscillator_real_data.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"チャートを保存しました: {output_path}")
        
        # 表示
        plt.show()
        
    except Exception as e:
        print(f"チャート作成でエラー: {e}")
        import traceback
        traceback.print_exc()


def analyze_oscillator_performance(data: pd.DataFrame, results: dict):
    """オシレーターの性能分析"""
    print("\n=== Ultimate Oscillator 性能分析 ===")
    
    # 中期設定の結果を使用
    mid_result = results.get('中期')
    if mid_result is None:
        print("中期設定の結果が見つかりません")
        return
    
    values = mid_result.values
    prices = data['close'].values
    
    # 信号検出
    buy_signals = []
    sell_signals = []
    
    for i in range(1, len(values)):
        if not (np.isnan(values[i]) or np.isnan(values[i-1])):
            # 買いシグナル（負から正へのクロス）
            if values[i-1] < 0 and values[i] > 0:
                buy_signals.append(i)
            # 売りシグナル（正から負へのクロス）
            elif values[i-1] > 0 and values[i] < 0:
                sell_signals.append(i)
    
    print(f"買いシグナル数: {len(buy_signals)}")
    print(f"売りシグナル数: {len(sell_signals)}")
    
    # シグナル後の価格変化分析
    if len(buy_signals) > 0:
        print("\n買いシグナル後の価格変化分析:")
        for i, signal_idx in enumerate(buy_signals[:5]):  # 最初の5つのシグナル
            if signal_idx + 10 < len(prices):
                price_change = (prices[signal_idx + 10] - prices[signal_idx]) / prices[signal_idx] * 100
                print(f"  シグナル{i+1}: {price_change:.2f}% (10期間後)")
    
    if len(sell_signals) > 0:
        print("\n売りシグナル後の価格変化分析:")
        for i, signal_idx in enumerate(sell_signals[:5]):  # 最初の5つのシグナル
            if signal_idx + 10 < len(prices):
                price_change = (prices[signal_idx + 10] - prices[signal_idx]) / prices[signal_idx] * 100
                print(f"  シグナル{i+1}: {price_change:.2f}% (10期間後)")


def test_parameter_optimization(data: pd.DataFrame):
    """パラメータ最適化テスト"""
    print("\n=== パラメータ最適化テスト ===")
    
    # テストするパラメータ範囲
    edge_values = [15, 20, 25, 30, 35, 40]
    width_values = [1.5, 2, 2.5, 3]
    rms_values = [50, 80, 100, 120, 150]
    
    best_config = None
    best_score = -np.inf
    
    print("パラメータ最適化中...")
    
    for edge in edge_values:
        for width in width_values:
            for rms in rms_values:
                try:
                    oscillator = UltimateOscillator(
                        edge=edge, 
                        width=width, 
                        rms_period=rms, 
                        src_type='hlc3'
                    )
                    result = oscillator.calculate(data)
                    
                    # スコア計算（ゼロクロス回数と標準偏差の組み合わせ）
                    valid_values = result.values[~np.isnan(result.values)]
                    if len(valid_values) > 10:
                        std_dev = np.std(valid_values)
                        zero_crosses = np.sum(np.abs(np.diff(np.sign(valid_values))) > 0)
                        
                        # スコア: ゼロクロス回数が多いほど良く、標準偏差が適度なほど良い
                        score = zero_crosses / (1 + abs(std_dev - 1.0))
                        
                        if score > best_score:
                            best_score = score
                            best_config = {'edge': edge, 'width': width, 'rms': rms, 'score': score}
                            
                except Exception as e:
                    continue
    
    if best_config:
        print(f"最適パラメータ: edge={best_config['edge']}, width={best_config['width']}, rms={best_config['rms']}")
        print(f"スコア: {best_config['score']:.4f}")
    else:
        print("最適パラメータが見つかりませんでした")


def main():
    """メイン関数"""
    print("Ehlers' Ultimate Oscillator テスト開始（実際の相場データ）")
    print("=" * 80)
    
    # 実際の相場データでのテスト
    data, results = test_ultimate_oscillator_real_data()
    
    if data is not None and results:
        # チャート作成
        plot_ultimate_oscillator_real_data(data, results)
        
        # 性能分析
        analyze_oscillator_performance(data, results)
        
        # パラメータ最適化
        test_parameter_optimization(data)
    else:
        print("データ読み込みに失敗したため、テストをスキップします。")
        print("設定ファイル（config.yaml）が正しく設定されているか確認してください。")
    
    print("\n" + "=" * 80)
    print("テスト完了")


if __name__ == "__main__":
    main() 