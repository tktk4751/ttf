#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.er_adaptive_ukf import CycleERAdaptiveUKF


class CycleERAdaptiveUKFChart:
    """
    Cycle-ER-Adaptive UKFを表示するローソク足チャートクラス
    
    🌟 **表示内容:**
    - ローソク足と出来高
    - Stage1フィルタ済み価格（通常UKF）
    - 最終フィルタ済み価格（Cycle-ER-Adaptive UKF）
    - Absolute Ultimate Cycle値
    - Efficiency Ratio値
    - 動的適応パラメータ（α, β, κ）
    - 信頼度スコア
    - 不確実性推定
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_er_ukf = None
        self.result = None
        self.fig = None
        self.axes = None
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        print("\nデータを読み込み・処理中...")
        
        try:
            # 設定ファイルの読み込み（JSON/YAML対応）
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    import json
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # シンプルなCSVファイル読み込み（テスト用）
            if 'file_path' in config:
                file_path = config['file_path']
                if os.path.exists(file_path):
                    print(f"CSVファイルを読み込み中: {file_path}")
                    self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # 必要な列があるかチェック
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in self.data.columns]
                    if missing_cols:
                        raise ValueError(f"必要な列が不足しています: {missing_cols}")
                    
                    print(f"データ読み込み完了: {config.get('symbol', 'Unknown')}")
                    print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
                    print(f"データ数: {len(self.data)}")
                    
                    return self.data
                else:
                    raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
            
            # Binanceデータソースを使用（従来の方法）
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
            raw_data = data_loader.load_data_from_config(config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(processed_data))
            self.data = processed_data[first_symbol]
            
            print(f"データ読み込み完了: {first_symbol}")
            print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
            print(f"データ数: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            raise e

    def calculate_indicators(self,
                            # UKFパラメータ
                            ukf_alpha: float = 0.001,
                            ukf_beta: float = 2.0,
                            ukf_kappa: float = 0.0,
                            # ERパラメータ  
                            er_period: int = 14,
                            er_smoothing_method: str = 'hma',
                            er_slope_index: int = 1,
                            er_range_threshold: float = 0.005,
                            # サイクルパラメータ
                            cycle_part: float = 1.0,
                            cycle_max_output: int = 120,
                            cycle_min_output: int = 5,
                            cycle_period_range: Tuple[int, int] = (5, 120),
                            # 適応パラメータ範囲
                            alpha_min: float = 0.0001,
                            alpha_max: float = 0.01,
                            beta_min: float = 1.0,
                            beta_max: float = 4.0,
                            kappa_min: float = -1.0,
                            kappa_max: float = 3.0,
                            # サイクル閾値
                            cycle_threshold_ratio_high: float = 0.8,
                            cycle_threshold_ratio_low: float = 0.3,
                            # その他
                            volatility_window: int = 10
                           ) -> None:
        """
        Cycle-ER-Adaptive UKFを計算する
        
        Args:
            ukf_alpha: UKFアルファパラメータ
            ukf_beta: UKFベータパラメータ
            ukf_kappa: UKFカッパパラメータ
            er_period: ER計算期間
            er_smoothing_method: ERスムージング方法
            er_slope_index: ERトレンド判定期間
            er_range_threshold: ERレンジ判定閾値
            cycle_part: サイクル部分の倍率
            cycle_max_output: サイクル最大出力値
            cycle_min_output: サイクル最小出力値
            cycle_period_range: サイクル期間の範囲
            alpha_min/max: αパラメータの最小/最大値
            beta_min/max: βパラメータの最小/最大値
            kappa_min/max: κパラメータの最小/最大値
            cycle_threshold_ratio_high/low: サイクル閾値比率
            volatility_window: ボラティリティ計算ウィンドウ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nCycle-ER-Adaptive UKFを計算中...")
        
        # Cycle-ER-Adaptive UKFを計算
        self.cycle_er_ukf = CycleERAdaptiveUKF(
            ukf_alpha=ukf_alpha,
            ukf_beta=ukf_beta,
            ukf_kappa=ukf_kappa,
            er_period=er_period,
            er_smoothing_method=er_smoothing_method,
            er_slope_index=er_slope_index,
            er_range_threshold=er_range_threshold,
            cycle_part=cycle_part,
            cycle_max_output=cycle_max_output,
            cycle_min_output=cycle_min_output,
            cycle_period_range=cycle_period_range,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            beta_min=beta_min,
            beta_max=beta_max,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            cycle_threshold_ratio_high=cycle_threshold_ratio_high,
            cycle_threshold_ratio_low=cycle_threshold_ratio_low,
            volatility_window=volatility_window
        )
        
        # 計算実行
        print("計算を実行します...")
        self.result = self.cycle_er_ukf.calculate(self.data)
        
        # 結果の確認
        print(f"計算完了")
        print(f"最終フィルタ値数: {len(self.result.values)}")
        print(f"Stage1フィルタ値数: {len(self.result.stage1_filtered)}")
        print(f"サイクル値数: {len(self.result.cycle_values)}")
        print(f"ER値数: {len(self.result.er_values)}")
        
        # NaN値のチェック
        final_nan = np.isnan(self.result.values).sum()
        stage1_nan = np.isnan(self.result.stage1_filtered).sum()
        cycle_nan = np.isnan(self.result.cycle_values).sum()
        er_nan = np.isnan(self.result.er_values).sum()
        
        print(f"NaN値 - 最終: {final_nan}, Stage1: {stage1_nan}, サイクル: {cycle_nan}, ER: {er_nan}")
        
        # 統計情報
        valid_final = self.result.values[~np.isnan(self.result.values)]
        valid_cycle = self.result.cycle_values[~np.isnan(self.result.cycle_values)]
        valid_er = self.result.er_values[~np.isnan(self.result.er_values)]
        
        if len(valid_final) > 0:
            print(f"最終フィルタ値 - 平均: {np.mean(valid_final):.2f}, 範囲: {np.min(valid_final):.2f} - {np.max(valid_final):.2f}")
        if len(valid_cycle) > 0:
            print(f"サイクル値 - 平均: {np.mean(valid_cycle):.2f}, 範囲: {np.min(valid_cycle):.2f} - {np.max(valid_cycle):.2f}")
        if len(valid_er) > 0:
            print(f"ER値 - 平均: {np.mean(valid_er):.3f}, 範囲: {np.min(valid_er):.3f} - {np.max(valid_er):.3f}")
        
        print("Cycle-ER-Adaptive UKF計算完了")
            
    def plot(self, 
             start_date: Optional[str] = None, 
             end_date: Optional[str] = None,
             savefig: Optional[str] = None) -> None:
        """
        Cycle-ER-Adaptive UKFチャートを描画
        
        Args:
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（YYYY-MM-DD形式）
            savefig: 保存ファイル名（Noneの場合は表示のみ）
        """
        if self.data is None or self.result is None:
            raise ValueError("データまたは結果が設定されていません。")
        
        print("📊 チャート描画中...")
        
        # データの範囲を設定
        df = self.data.copy()
        
        # 日付範囲フィルタリング
        if start_date or end_date:
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
        
        print("結果データを取得中...")
        
        # 結果データを取得
        n_points = len(df)
        
        # 結果データの長さを調整
        if len(self.result.values) != n_points:
            # データ長が異なる場合は調整
            min_len = min(len(self.result.values), n_points)
            df = df.iloc[:min_len]
            n_points = min_len
        
        # 各値を取得し、長さを統一
        final_values = self.result.values[:n_points]
        stage1_values = self.result.stage1_filtered[:n_points]
        cycle_values = self.result.cycle_values[:n_points]
        er_values = self.result.er_values[:n_points]
        adaptive_alpha = self.result.adaptive_alpha[:n_points]
        confidence_scores = self.result.confidence_scores[:n_points]
        
        print(f"チャートデータ準備完了 - 行数: {n_points}")
        
        # NaN値のチェックと有効データの確認
        final_valid = np.sum(~np.isnan(final_values))
        stage1_valid = np.sum(~np.isnan(stage1_values))
        cycle_valid = np.sum(~np.isnan(cycle_values))
        er_valid = np.sum(~np.isnan(er_values))
        
        print(f"有効データ数 - 最終: {final_valid}, Stage1: {stage1_valid}, サイクル: {cycle_valid}, ER: {er_valid}")
        
        # 有効なデータが少なすぎる場合は警告またはエラー
        if final_valid < n_points * 0.1:  # 10%未満の場合
            print(f"⚠️ 警告: 有効データが少なすぎます ({final_valid}/{n_points})")
            if final_valid == 0:
                print("❌ 有効なデータがありません。基本チャートのみ表示します。")
                # 基本チャートのみ表示
                try:
                    simple_kwargs = {
                        'type': 'candle',
                        'style': 'charles',
                        'volume': True,
                        'figscale': 1.0,
                        'title': f'Basic Chart - No Valid Filter Data (Data Points: {n_points})',
                        'ylabel': 'Price'
                    }
                    
                    if savefig:
                        simple_kwargs['savefig'] = dict(fname=savefig, dpi=300, bbox_inches='tight')
                    
                    fig, axes = mpf.plot(df, **simple_kwargs)
                    
                    if not savefig:
                        plt.show()
                    else:
                        print(f"📁 基本チャートを保存しました: {savefig}")
                    
                    return
                    
                except Exception as e:
                    print(f"❌ 基本チャートもエラー: {e}")
                    raise e
        
        # HLC3を計算
        hlc3_values = (df['high'].values + df['low'].values + df['close'].values) / 3.0
        
        # 追加プロット用データ準備
        addplot_data = []
        
        # HLC3（元データ）- 有効値のみ表示
        hlc3_clean = hlc3_values.copy()
        hlc3_clean[np.isnan(hlc3_clean)] = np.nan
        addplot_data.append(mpf.make_addplot(hlc3_clean, color='gray', alpha=0.6, width=0.8, 
                                           secondary_y=False, panel=0))
        
        # Stage1 UKF（青色）- 有効値のみ表示
        stage1_clean = stage1_values.copy()
        stage1_clean[np.isnan(stage1_clean)] = np.nan
        if stage1_valid > 0:
            addplot_data.append(mpf.make_addplot(stage1_clean, color='blue', alpha=0.7, width=1.2,
                                               secondary_y=False, panel=0))
        
        # 最終適応UKF（赤色）- 有効値のみ表示
        final_clean = final_values.copy()
        final_clean[np.isnan(final_clean)] = np.nan
        if final_valid > 0:
            addplot_data.append(mpf.make_addplot(final_clean, color='red', alpha=0.8, width=1.5,
                                               secondary_y=False, panel=0))
        
        # 出来高パネルに指標を統合表示（secondary_yを使用してスケール分離）
        volume_panel_data = []
        
        # Absolute Ultimate Cycle値（緑色）- 出来高パネルに表示
        cycle_clean = cycle_values.copy()
        cycle_clean[np.isnan(cycle_clean)] = np.nan
        if cycle_valid > 0:
            volume_panel_data.append(mpf.make_addplot(cycle_clean, color='green', alpha=0.8, width=1.0,
                                                    secondary_y=True, panel=1))
        
        # ER値*100（オレンジ色）- スケール調整して出来高パネルに表示
        er_scaled = er_values * 100
        er_scaled[np.isnan(er_scaled)] = np.nan
        if er_valid > 0:
            volume_panel_data.append(mpf.make_addplot(er_scaled, color='orange', alpha=0.8, width=1.0,
                                                    secondary_y=True, panel=1))
        
        # 適応α*1000（紫色）- スケール調整して出来高パネルに表示
        alpha_scaled = adaptive_alpha * 1000
        alpha_scaled[np.isnan(alpha_scaled)] = np.nan
        if np.sum(~np.isnan(alpha_scaled)) > 0:
            volume_panel_data.append(mpf.make_addplot(alpha_scaled, color='purple', alpha=0.8, width=1.0,
                                                    secondary_y=True, panel=1))
        
        # 信頼度*50（青緑色）- スケール調整して出来高パネルに表示
        confidence_scaled = confidence_scores * 50
        confidence_scaled[np.isnan(confidence_scaled)] = np.nan
        if np.sum(~np.isnan(confidence_scaled)) > 0:
            volume_panel_data.append(mpf.make_addplot(confidence_scaled, color='teal', alpha=0.8, width=1.0,
                                                    secondary_y=True, panel=1))
        
        # 全ての追加プロットを統合
        addplot_data.extend(volume_panel_data)
        
        # 安全なデータ処理（NaN値対応）
        cleaned_addplot_data = []
        
        for i, ap in enumerate(addplot_data):
            try:
                # データを取得
                if hasattr(ap, 'data'):
                    data = ap.data
                elif hasattr(ap, '_data'):
                    data = ap._data  
                elif isinstance(ap, dict) and 'data' in ap:
                    data = ap['data']
                else:
                    print(f"⚠️ スキップ: データ属性が見つかりません (index: {i})")
                    continue
                
                # データが空でないかチェック
                if data is None or len(data) == 0:
                    print(f"⚠️ スキップ: 空のデータ (index: {i})")
                    continue
                
                # NaN値の処理
                data_clean = np.array(data, dtype=float)
                
                # 全てNaNの場合はスキップ
                if np.all(np.isnan(data_clean)):
                    print(f"⚠️ スキップ: 全てNaN値のデータ (index: {i})")
                    continue
                
                # NaN値を前方補完
                mask = ~np.isnan(data_clean)
                if np.any(mask):
                    # 最初の有効値で前を埋める
                    first_valid_idx = np.where(mask)[0][0]
                    if first_valid_idx > 0:
                        data_clean[:first_valid_idx] = data_clean[first_valid_idx]
                    
                    # 前方補完
                    for j in range(1, len(data_clean)):
                        if np.isnan(data_clean[j]):
                            data_clean[j] = data_clean[j-1]
                
                # まだNaNが残っている場合は平均値で埋める
                remaining_nan = np.isnan(data_clean)
                if np.any(remaining_nan):
                    valid_mean = np.nanmean(data_clean)
                    if not np.isnan(valid_mean):
                        data_clean[remaining_nan] = valid_mean
                    else:
                        data_clean[remaining_nan] = 0.0
                
                # 無限値の処理
                data_clean[np.isinf(data_clean)] = 0.0
                
                # 新しいプロットデータを作成（シンプルな方法）
                color = getattr(ap, 'color', 'blue')
                alpha = getattr(ap, 'alpha', 0.8)
                width = getattr(ap, 'width', 1.0)
                secondary_y = getattr(ap, 'secondary_y', False)
                panel = getattr(ap, 'panel', 0)
                
                # mplfinanceに安全なパラメータのみ渡す
                cleaned_ap = mpf.make_addplot(
                    data_clean,
                    color=color,
                    alpha=alpha,
                    width=width,
                    secondary_y=secondary_y,
                    panel=panel
                )
                
                cleaned_addplot_data.append(cleaned_ap)
                
            except Exception as e:
                print(f"⚠️ データ処理エラー (index: {i}): {e}")
                continue
        
        # クリーンなデータに更新
        addplot_data = cleaned_addplot_data
        
        # チャート設定
        kwargs = {
            'type': 'candle',
            'style': 'charles',
            'addplot': addplot_data,
            'volume': True,
            'panel_ratios': (3, 1),  # メイン:出来高 = 3:1
            'figscale': 1.2,
            'figratio': (12, 8),
            'title': f'Cycle-ER-Adaptive UKF Analysis\n'
                    f'Data Points: {n_points}, Valid Final: {final_valid}, Valid Cycle: {cycle_valid}, Valid ER: {er_valid}',
            'ylabel': 'Price',
            'ylabel_lower': 'Volume & Indicators',
            'tight_layout': True,
            'returnfig': True
        }
        
        if savefig:
            kwargs['savefig'] = dict(fname=savefig, dpi=300, bbox_inches='tight')
        
        try:
            # mplfinanceでプロット
            fig, axes = mpf.plot(df, **kwargs)
            
            if not savefig:
                plt.show()
            else:
                print(f"📁 チャートを保存しました: {savefig}")
                
        except Exception as e:
            print(f"❌ チャート描画エラー: {e}")
            import traceback
            traceback.print_exc()
            
            # フォールバック: 基本的なプロット
            try:
                print("🔄 基本プロットにフォールバック...")
                simple_kwargs = {
                    'type': 'candle',
                    'style': 'charles',
                    'volume': True,
                    'figscale': 1.0,
                    'title': f'Fallback Chart - Data Points: {n_points}',
                    'returnfig': True
                }
                
                if savefig:
                    simple_kwargs['savefig'] = dict(fname=savefig, dpi=300, bbox_inches='tight')
                
                fig, axes = mpf.plot(df, **simple_kwargs)
                
                if not savefig:
                    plt.show()
                else:
                    print(f"📁 基本チャートを保存しました: {savefig}")
                    
            except Exception as e2:
                print(f"❌ 基本プロットもエラー: {e2}")
                raise e2


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Cycle-ER-Adaptive UKFの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    
    # UKFパラメータ
    parser.add_argument('--ukf-alpha', type=float, default=0.001, help='UKFアルファパラメータ')
    parser.add_argument('--ukf-beta', type=float, default=2.0, help='UKFベータパラメータ')
    parser.add_argument('--ukf-kappa', type=float, default=0.0, help='UKFカッパパラメータ')
    
    # ERパラメータ
    parser.add_argument('--er-period', type=int, default=14, help='ER計算期間')
    parser.add_argument('--er-smoothing', type=str, default='hma', help='ERスムージング方法')
    
    # サイクルパラメータ
    parser.add_argument('--cycle-part', type=float, default=1.0, help='サイクル部分の倍率')
    parser.add_argument('--cycle-max', type=int, default=120, help='サイクル最大出力値')
    parser.add_argument('--cycle-min', type=int, default=5, help='サイクル最小出力値')
    
    # 適応パラメータ範囲
    parser.add_argument('--alpha-min', type=float, default=0.0001, help='αパラメータ最小値')
    parser.add_argument('--alpha-max', type=float, default=0.01, help='αパラメータ最大値')
    parser.add_argument('--beta-min', type=float, default=1.0, help='βパラメータ最小値')
    parser.add_argument('--beta-max', type=float, default=4.0, help='βパラメータ最大値')
    
    args = parser.parse_args()
    
    # チャートを作成
    chart = CycleERAdaptiveUKFChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        ukf_alpha=args.ukf_alpha,
        ukf_beta=args.ukf_beta,
        ukf_kappa=args.ukf_kappa,
        er_period=args.er_period,
        er_smoothing_method=args.er_smoothing,
        cycle_part=args.cycle_part,
        cycle_max_output=args.cycle_max,
        cycle_min_output=args.cycle_min,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        beta_min=args.beta_min,
        beta_max=args.beta_max
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 