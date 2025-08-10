#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
エーラーズ統合サイクル検出器の簡易テスト
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# データ取得
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

def load_data_from_config(config_path: str = 'config.yaml') -> pd.DataFrame:
    """設定ファイルからデータを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    print("データを読み込み・処理中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"データ読み込み完了: {first_symbol}")
    print(f"期間: {data.index.min()} → {data.index.max()}")
    print(f"データ数: {len(data)}")
    
    return data

def test_detector_import():
    """検出器のインポートテスト"""
    print("=== 検出器インポートテスト ===")
    
    try:
        # 統合検出器をインポート
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
        print("✓ EhlersUnifiedDC のインポート成功")
        
        # 利用可能な検出器を取得
        available_detectors = EhlersUnifiedDC.get_available_detectors()
        print(f"✓ 利用可能な検出器数: {len(available_detectors)}")
        
        for detector_name, description in available_detectors.items():
            print(f"  - {detector_name}: {description}")
        
        return EhlersUnifiedDC, available_detectors
        
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return None, {}

def test_all_detectors(data: pd.DataFrame, detector_class, detector_types: list):
    """全サイクル検出器のテスト"""
    print("\n=== 全サイクル検出器テスト ===")
    
    # テスト用にデータサイズを制限（計算時間短縮のため）
    test_data = data.tail(1000).copy()  # 最後の1000データポイントのみ使用
    print(f"テスト用データサイズ: {len(test_data)} ポイント")
    
    results = {}
    success_count = 0
    error_count = 0
    timeout_count = 0
    
    # 重い計算の検出器をスキップするリスト
    skip_detectors = ['supreme', 'quantum_adaptive', 'adaptive_ensemble']  # 非常に重い計算
    
    for detector_type in detector_types:
        try:
            # 重い計算の検出器をスキップ
            if detector_type in skip_detectors:
                print(f"検出器 '{detector_type}' をスキップ（計算時間短縮のため）")
                timeout_count += 1
                continue
                
            print(f"検出器 '{detector_type}' をテスト中...")
            
            # 検出器タイプに応じてパラメータを調整
            detector_params = {
                'detector_type': detector_type,
                'src_type': 'hlc3',
                'max_cycle': 50,
                'min_cycle': 6,
                'use_kalman_filter': False,  # カルマンフィルターを無効化
                'max_output': 34,
                'min_output': 1,
                'cycle_part': 0.5
            }
            
            # 特定の検出器に対する追加パラメータ
            if detector_type in ['cycle_period', 'cycle_period2']:
                detector_params['alpha'] = 0.07
            elif detector_type == 'bandpass_zero':
                detector_params['bandwidth'] = 0.6
                detector_params['center_period'] = 15.0
            elif detector_type == 'autocorr_perio':
                detector_params['avg_length'] = 3.0
            elif detector_type == 'dft_dominant':
                detector_params['window'] = 50
            elif detector_type in ['absolute_ultimate', 'ultra_supreme_stability']:
                detector_params['period_range'] = (5, 120)
            elif detector_type == 'adaptive_ensemble':
                detector_params['entropy_window'] = 20
                detector_params['period_range'] = (5, 120)
            elif detector_type == 'supreme':
                detector_params['dft_window'] = 50
                detector_params['use_ukf'] = True
                detector_params['ukf_alpha'] = 0.001
                detector_params['smoothing_factor'] = 0.1
                detector_params['weight_lookback'] = 20
                detector_params['adaptive_params'] = True
            elif detector_type == 'refined':
                detector_params['ultimate_smoother_period'] = 20.0
                detector_params['use_ultimate_smoother'] = True
                detector_params['period_range'] = (5.0, 120.0)
                detector_params['alpha'] = 0.07
            
            detector = detector_class(**detector_params)
            
            result = detector.calculate(test_data)
            
            valid_count = np.sum(~np.isnan(result))
            mean_value = np.nanmean(result) if valid_count > 0 else np.nan
            std_value = np.nanstd(result) if valid_count > 0 else np.nan
            min_value = np.nanmin(result) if valid_count > 0 else np.nan
            max_value = np.nanmax(result) if valid_count > 0 else np.nan
            
            results[detector_type] = result
            success_count += 1
            print(f"  ✓ 成功: 有効値数={valid_count}, 平均={mean_value:.2f}, "
                  f"標準偏差={std_value:.2f}, 範囲=[{min_value:.2f}, {max_value:.2f}]")
            
        except Exception as e:
            error_count += 1
            print(f"  ✗ エラー: {str(e)}")
            results[detector_type] = None
            continue
    
    print(f"\n=== テスト結果サマリー ===")
    print(f"成功: {success_count}/{len(detector_types)} 検出器")
    print(f"エラー: {error_count}/{len(detector_types)} 検出器")
    print(f"スキップ: {timeout_count}/{len(detector_types)} 検出器")
    print(f"テスト実行: {success_count + error_count}/{len(detector_types)} 検出器")
    
    return results

def plot_comprehensive_results(data: pd.DataFrame, results: dict):
    """全検出器結果の包括的なプロット"""
    print("\n=== 結果プロット ===")
    
    # 有効な結果のみフィルタ
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("プロット可能な結果がありません")
        return
    
    print(f"プロット対象: {len(valid_results)}個の検出器")
    
    # テスト用データサイズに合わせてプロットデータを調整
    plot_data = data.tail(1000).copy()  # テストデータと同じサイズ
    
    # 検出器を1つのパネルに複数描画（見やすくするため）
    detectors_per_panel = 4
    num_panels = (len(valid_results) + detectors_per_panel - 1) // detectors_per_panel
    
    fig, axes = plt.subplots(num_panels + 1, 1, figsize=(16, 4 * (num_panels + 1)), sharex=True)
    
    # axesが単一の場合はリストに変換
    if num_panels == 0:
        axes = [axes]
    
    # カラーパレット
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
              'olive', 'cyan', 'magenta', 'darkblue', 'darkred', 'darkgreen', 'darkorange']
    
    # 価格チャート
    axes[0].plot(plot_data.index, plot_data['close'], label='Close Price', color='black', linewidth=1.5)
    axes[0].set_title('価格チャート (SOL/USDT 4H)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('価格 (USDT)')
    
    # 検出器結果をパネルごとにグループ化
    detector_items = list(valid_results.items())
    for panel_idx in range(num_panels):
        start_idx = panel_idx * detectors_per_panel
        end_idx = min(start_idx + detectors_per_panel, len(detector_items))
        panel_detectors = detector_items[start_idx:end_idx]
        
        ax = axes[panel_idx + 1]
        
        for color_idx, (detector_name, result) in enumerate(panel_detectors):
            if result is not None:
                # テストデータのインデックスに合わせて結果をプロット
                result_series = pd.Series(result, index=plot_data.index)
                color = colors[color_idx % len(colors)]
                ax.plot(result_series.index, result_series.values, 
                       label=detector_name, color=color, linewidth=1.2, alpha=0.8)
        
        # パネルのタイトルと設定
        detector_names = [name for name, _ in panel_detectors]
        ax.set_title(f'パネル {panel_idx + 1}: {", ".join(detector_names)}', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('サイクル値')
        
        # Y軸の範囲を調整
        try:
            all_values = []
            for _, result in panel_detectors:
                if result is not None:
                    result_series = pd.Series(result, index=plot_data.index)
                    valid_vals = result_series.dropna()
                    if len(valid_vals) > 0:
                        all_values.extend(valid_vals.values)
            
            if all_values:
                y_min, y_max = np.percentile(all_values, [5, 95])
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        except:
            pass
    
    plt.tight_layout()
    
    # 保存
    filename = 'ehlers_unified_dc_all_detectors_test.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"結果を保存しました: {filename}")
    
    # 統計サマリーを出力
    print(f"\n=== 検出器性能サマリー ===")
    for detector_name, result in valid_results.items():
        if result is not None:
            valid_count = np.sum(~np.isnan(result))
            if valid_count > 0:
                mean_val = np.nanmean(result)
                std_val = np.nanstd(result)
                median_val = np.nanmedian(result)
                print(f"{detector_name:25}: 平均={mean_val:6.2f}, 中央値={median_val:6.2f}, 標準偏差={std_val:6.2f}")
    
    plt.show()

def main():
    """メイン関数"""
    print("エーラーズ統合サイクル検出器 - 簡易テスト開始\n")
    
    # インポートテスト
    detector_class, available_detectors = test_detector_import()
    if detector_class is None:
        print("インポートに失敗したため、テストを終了します")
        return
    
    # データ読み込み
    try:
        data = load_data_from_config()
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 検出器テスト
    detector_types = list(available_detectors.keys())
    results = test_all_detectors(data, detector_class, detector_types)
    
    # 結果プロット
    if results:
        plot_comprehensive_results(data, results)
    
    print("\nテスト完了")

if __name__ == "__main__":
    main()