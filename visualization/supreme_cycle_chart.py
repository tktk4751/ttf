#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supreme Cycle Detector チャート表示

設定ファイルから実際の相場データを取得し、
Supreme Cycle Detectorの解析結果を表示します。
"""

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
from indicators.cycle.supreme_cycle_detector import SupremeCycleDetector
from indicators.cycle.ehlers_hody_dce import EhlersHoDyDCE
from indicators.cycle.ehlers_dudi_dce import EhlersDuDiDCE
from indicators.cycle.ehlers_phac_dce import EhlersPhAcDCE
from indicators.cycle.ehlers_dft_dominant_cycle import EhlersDFTDominantCycle


class SupremeCycleChart:
    """
    Supreme Cycle Detectorを表示するチャートクラス
    
    - ローソク足と出来高
    - Supreme Cycleと各コンポーネントサイクル
    - 重み配分の時系列変化
    - 信頼度とボラティリティ状態
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.supreme_detector = None
        self.component_detectors = {}
        self.fig = None
        self.axes = None
        self.symbol = None
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        # 設定ファイルの読み込み
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
        print("\nデータを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        self.symbol = next(iter(processed_data))
        self.data = processed_data[self.symbol]
        
        print(f"データ読み込み完了: {self.symbol}")
        print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"データ数: {len(self.data)}")
        
        return self.data

    def calculate_indicators(self,
                            # Supreme Cycle Detector パラメータ
                            lp_period: int = 10,
                            hp_period: int = 48,
                            cycle_part: float = 0.5,
                            max_output: int = 120,
                            min_output: int = 5,
                            src_type: str = 'hlc3',
                            # DFT固有パラメータ
                            dft_window: int = 50,
                            # Supreme固有パラメータ
                            use_ukf: bool = False,
                            ukf_alpha: float = 0.001,
                            smoothing_factor: float = 0.1,
                            weight_lookback: int = 20,
                            adaptive_params: bool = True,
                            # コンポーネント計算フラグ
                            calculate_components: bool = True
                           ) -> None:
        """
        Supreme Cycle Detectorと各コンポーネントを計算する
        
        Args:
            各種パラメータ（SupremeCycleDetectorと同じ）
            calculate_components: 個別コンポーネントも計算するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nSupreme Cycle Detectorを計算中...")
        
        # Supreme Cycle Detectorを計算
        self.supreme_detector = SupremeCycleDetector(
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            dft_window=dft_window,
            use_ukf=use_ukf,
            ukf_alpha=ukf_alpha,
            smoothing_factor=smoothing_factor,
            weight_lookback=weight_lookback,
            adaptive_params=adaptive_params
        )
        
        # Supreme Cycleの計算
        print("Supreme Cycle計算を実行します...")
        supreme_cycles = self.supreme_detector.calculate(self.data)
        print(f"Supreme Cycle計算完了 - データ数: {len(supreme_cycles)}")
        
        # コンポーネント検出器の計算（オプション）
        if calculate_components:
            print("\n各コンポーネントサイクルを計算中...")
            
            # HoDy
            self.component_detectors['hody'] = EhlersHoDyDCE(
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type
            )
            
            # DuDi
            self.component_detectors['dudi'] = EhlersDuDiDCE(
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type
            )
            
            # PhAc
            self.component_detectors['phac'] = EhlersPhAcDCE(
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type
            )
            
            # DFT
            self.component_detectors['dft'] = EhlersDFTDominantCycle(
                window=dft_window,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type
            )
            
            # 各コンポーネントの計算
            for name, detector in self.component_detectors.items():
                cycles = detector.calculate(self.data)
                print(f"{name.upper()}: 計算完了 - データ数: {len(cycles)}")
        
        # 統計情報の表示
        if self.supreme_detector._result:
            info = self.supreme_detector.get_component_info()
            if info:
                print("\n=== Supreme Cycle Detector 統計 ===")
                print(f"平均信頼度: {info['average_confidence']:.2%}")
                print("\n平均コンポーネント重み:")
                for comp, weight in info['component_weights'].items():
                    print(f"  {comp.upper()}: {weight:.2%}")
                print("\nボラティリティ分布:")
                for state, ratio in info['volatility_distribution'].items():
                    print(f"  {state.capitalize()}: {ratio:.2%}")
            
    def plot(self, 
            title: Optional[str] = None, 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_components: bool = True,
            show_weights: bool = True,
            show_confidence: bool = True,
            show_volatility: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        チャートを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_components: コンポーネントサイクルを表示するか
            show_weights: 重み配分を表示するか
            show_confidence: 信頼度を表示するか
            show_volatility: ボラティリティ状態を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
            
        if self.supreme_detector is None:
            raise ValueError("インジケーターが計算されていません。")
        
        # タイトルの設定
        if title is None:
            title = f'Supreme Cycle Detector Analysis - {self.symbol}'
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Supreme Cycleの結果を取得
        result = self.supreme_detector._result
        supreme_cycles = result.values[:len(self.data)]
        
        # 全データの時系列データフレームを作成
        cycle_data = pd.DataFrame(
            index=self.data.index,
            data={'supreme': supreme_cycles}
        )
        
        # コンポーネントサイクルを追加
        if show_components and hasattr(result, 'component_cycles'):
            for name, cycles in result.component_cycles.items():
                cycle_data[name] = cycles[:len(self.data)]
        
        # 重み、信頼度、ボラティリティを追加
        if hasattr(result, 'weights'):
            for name, weights in result.weights.items():
                cycle_data[f'weight_{name}'] = weights[:len(self.data)]
        
        if hasattr(result, 'confidence'):
            cycle_data['confidence'] = result.confidence[:len(self.data)]
            
        if hasattr(result, 'volatility_state'):
            cycle_data['volatility'] = result.volatility_state[:len(self.data)]
        
        # 絞り込み後のデータに結合
        df = df.join(cycle_data)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # パネル数の計算とmplfinanceのパネル配置
        # mplfinanceのパネル配置:
        # panel=0: メインチャート（価格）
        # panel=1: 出来高（show_volume=Trueの場合のみ）
        # panel=2以降: カスタムパネル
        
        # mplfinanceでプロット用の設定
        all_plots = []
        
        if show_volume:
            # 出来高ありの場合: メイン(3), 出来高(1), Supreme(2)
            panel_ratios = [3, 1, 2]
            cycle_panel = 2
            next_panel = 3
        else:
            # 出来高なしの場合: メイン(3), Supreme(2)
            panel_ratios = [3, 2]
            cycle_panel = 1
            next_panel = 2
        
        # 1. Supreme Cycleパネル
        all_plots.append(mpf.make_addplot(df['supreme'], panel=cycle_panel, color='red', width=2, 
                                         ylabel='Supreme Cycle', label='Supreme'))
        
        if show_components and 'hody' in df.columns:
            all_plots.append(mpf.make_addplot(df['hody'], panel=cycle_panel, color='green', 
                                            width=1, alpha=0.7, label='HoDy'))
            all_plots.append(mpf.make_addplot(df['dudi'], panel=cycle_panel, color='blue', 
                                            width=1, alpha=0.7, label='DuDi'))
            all_plots.append(mpf.make_addplot(df['phac'], panel=cycle_panel, color='magenta', 
                                            width=1, alpha=0.7, label='PhAc'))
            all_plots.append(mpf.make_addplot(df['dft'], panel=cycle_panel, color='cyan', 
                                            width=1, alpha=0.7, label='DFT'))
        
        current_panel = next_panel
        
        # 3. 重み配分パネル
        if show_weights and 'weight_hody' in df.columns:
            panel_ratios.append(1)
            all_plots.append(mpf.make_addplot(df['weight_hody'], panel=current_panel, color='green', 
                                            width=1, ylabel='Weights', label='W_HoDy'))
            all_plots.append(mpf.make_addplot(df['weight_dudi'], panel=current_panel, color='blue', 
                                            width=1, label='W_DuDi'))
            all_plots.append(mpf.make_addplot(df['weight_phac'], panel=current_panel, color='magenta', 
                                            width=1, label='W_PhAc'))
            all_plots.append(mpf.make_addplot(df['weight_dft'], panel=current_panel, color='cyan', 
                                            width=1, label='W_DFT'))
            current_panel += 1
        
        # 4. 信頼度パネル
        if show_confidence and 'confidence' in df.columns:
            panel_ratios.append(1)
            all_plots.append(mpf.make_addplot(df['confidence'], panel=current_panel, color='orange', 
                                            width=2, ylabel='Confidence', label='Confidence'))
            current_panel += 1
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            volume=show_volume,
            panel_ratios=panel_ratios,
            addplot=all_plots
        )
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 参照線の追加
        # Supreme Cycleパネル
        cycle_ax_idx = 2 if show_volume else 1
        cycle_ax = axes[cycle_ax_idx]
        cycle_ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='20 Period')
        cycle_ax.axhline(y=35, color='gray', linestyle='--', alpha=0.3, label='35 Period')
        cycle_ax.legend(loc='upper right')
        
        # 動的にパネルのインデックスを計算
        ax_idx = 3 if show_volume else 2  # Supreme Cycleパネルの次から開始
        
        # 重みパネル
        if show_weights and 'weight_hody' in df.columns:
            if ax_idx < len(axes):
                weight_ax = axes[ax_idx]
                weight_ax.axhline(y=0.25, color='black', linestyle='--', alpha=0.3)
                weight_ax.set_ylim(0, 1)
                weight_ax.legend(loc='upper right')
                ax_idx += 1
        
        # 信頼度パネル
        if show_confidence and 'confidence' in df.columns:
            if ax_idx < len(axes):
                conf_ax = axes[ax_idx]
                conf_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
                conf_ax.set_ylim(0, 1)
                ax_idx += 1
        
        # ボラティリティ状態の表示（メインチャート上）
        if show_volatility and 'volatility' in df.columns:
            ax_main = axes[0]
            
            # ボラティリティ状態の背景色
            vol_colors = {0: 'green', 1: 'yellow', 2: 'red'}
            vol_alpha = {0: 0.1, 1: 0.15, 2: 0.2}
            
            # 連続する同じ状態をグループ化
            vol_state = df['volatility'].values
            i = 0
            while i < len(vol_state):
                current_state = vol_state[i]
                j = i
                while j < len(vol_state) and vol_state[j] == current_state:
                    j += 1
                
                # 背景を描画
                if current_state in vol_colors:
                    ax_main.axvspan(df.index[i], df.index[j-1], 
                                   color=vol_colors[current_state], 
                                   alpha=vol_alpha[current_state])
                i = j
        
        # 統計情報の表示
        print(f"\n=== サイクル統計 ===")
        print(f"Supreme Cycle - 平均: {df['supreme'].mean():.1f}, 標準偏差: {df['supreme'].std():.1f}")
        
        if show_components and 'hody' in df.columns:
            print(f"HoDy - 平均: {df['hody'].mean():.1f}, 標準偏差: {df['hody'].std():.1f}")
            print(f"DuDi - 平均: {df['dudi'].mean():.1f}, 標準偏差: {df['dudi'].std():.1f}")
            print(f"PhAc - 平均: {df['phac'].mean():.1f}, 標準偏差: {df['phac'].std():.1f}")
            print(f"DFT  - 平均: {df['dft'].mean():.1f}, 標準偏差: {df['dft'].std():.1f}")
        
        if 'confidence' in df.columns:
            print(f"\n信頼度 - 平均: {df['confidence'].mean():.2%}, 最小: {df['confidence'].min():.2%}, 最大: {df['confidence'].max():.2%}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Supreme Cycle Detectorの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--lp-period', type=int, default=10, help='ローパスフィルター期間')
    parser.add_argument('--hp-period', type=int, default=48, help='ハイパスフィルター期間')
    parser.add_argument('--use-ukf', action='store_true', help='UKFフィルタリングを使用')
    parser.add_argument('--adaptive', action='store_true', help='適応的パラメータ調整を使用')
    parser.add_argument('--no-components', action='store_true', help='コンポーネント表示を無効化')
    parser.add_argument('--no-weights', action='store_true', help='重み表示を無効化')
    parser.add_argument('--no-confidence', action='store_true', help='信頼度表示を無効化')
    parser.add_argument('--no-volatility', action='store_true', help='ボラティリティ表示を無効化')
    
    args = parser.parse_args()
    
    # チャートを作成
    chart = SupremeCycleChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        lp_period=args.lp_period,
        hp_period=args.hp_period,
        src_type=args.src_type,
        use_ukf=args.use_ukf,
        adaptive_params=args.adaptive,
        calculate_components=not args.no_components
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output,
        show_components=not args.no_components,
        show_weights=not args.no_weights,
        show_confidence=not args.no_confidence,
        show_volatility=not args.no_volatility
    )


if __name__ == "__main__":
    main()