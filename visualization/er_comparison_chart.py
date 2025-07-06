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
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.efficiency_ratio import EfficiencyRatio
from indicators.hyper_efficiency_ratio import HyperEfficiencyRatio

# フォント設定（警告を防ぐため）
plt.rcParams['font.family'] = ['DejaVu Sans']
import matplotlib
matplotlib.use('Agg')


class ERComparisonChart:
    """
    Efficiency Ratio比較チャートクラス
    
    - ローソク足と出来高
    - 従来Efficiency Ratio
    - Hyper Efficiency Ratio
    - 比較分析パネル
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.classic_er = None
        self.hyper_er = None
        self.classic_result = None
        self.hyper_result = None
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
        print("\n📊 Loading and processing market data...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ Data loaded: {first_symbol}")
        print(f"📅 Period: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📈 Data points: {len(self.data)}")
        
        return self.data

    def calculate_indicators(self,
                           classic_period: int = 14,
                           classic_src_type: str = 'hlc3',
                           hyper_period: int = 14,
                           hyper_src_type: str = 'hlc3',
                           hyper_slope_index: int = 3,
                           hyper_threshold: float = 0.3) -> None:
        """
        両方のEfficiency Ratioを計算する
        
        Args:
            classic_period: 従来ER期間
            classic_src_type: 従来ERソースタイプ
            hyper_period: ハイパーER期間
            hyper_src_type: ハイパーERソースタイプ
            hyper_slope_index: ハイパーERスロープインデックス
            hyper_threshold: ハイパーER閾値
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please run load_data_from_config() first.")
            
        print("\n🔬 Calculating Efficiency Ratios...")
        
        # 従来ERの計算
        print("📊 Calculating Classic ER...")
        self.classic_er = EfficiencyRatio(
            period=classic_period,
            src_type=classic_src_type
        )
        self.classic_result = self.classic_er.calculate(self.data)
        
        # ハイパーERの計算
        print("🚀 Calculating Hyper ER...")
        self.hyper_er = HyperEfficiencyRatio(
            window=hyper_period,
            src_type=hyper_src_type,
            slope_index=hyper_slope_index,
            threshold=hyper_threshold
        )
        self.hyper_result = self.hyper_er.calculate(self.data)
        
        # 結果の統計情報
        classic_valid = ~np.isnan(self.classic_result.values)
        hyper_valid = ~np.isnan(self.hyper_result.values)
        
        print(f"📈 Classic ER - Valid points: {classic_valid.sum()}, Range: {self.classic_result.values[classic_valid].min():.3f} - {self.classic_result.values[classic_valid].max():.3f}")
        print(f"🚀 Hyper ER - Valid points: {hyper_valid.sum()}, Range: {self.hyper_result.values[hyper_valid].min():.3f} - {self.hyper_result.values[hyper_valid].max():.3f}")
        
        # 相関係数の計算
        both_valid = classic_valid & hyper_valid
        if both_valid.sum() > 10:
            correlation = np.corrcoef(
                self.classic_result.values[both_valid],
                self.hyper_result.values[both_valid]
            )[0, 1]
            print(f"🔗 Correlation: {correlation:.4f}")
        
        print("✅ Efficiency Ratio calculation completed")
            
    def plot(self, 
            title: str = "Efficiency Ratio Comparison", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとEfficiency Ratioを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please run load_data_from_config() first.")
            
        if self.classic_result is None or self.hyper_result is None:
            raise ValueError("Indicators not calculated. Please run calculate_indicators() first.")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーター値を取得（フィルタリングされたインデックスに合わせる）
        print("📊 Preparing chart data...")
        
        # 全データのインジケーター値を取得
        classic_values = self.classic_result.values
        hyper_values = self.hyper_result.values
        hyper_linear_vol = self.hyper_result.linear_volatility
        hyper_nonlinear_vol = self.hyper_result.nonlinear_volatility
        hyper_adaptive_vol = self.hyper_result.adaptive_volatility
        
        # フィルタリング後のデータフレームの期間に対応するインジケーター値を抽出
        start_idx = self.data.index.get_loc(df.index[0])
        end_idx = self.data.index.get_loc(df.index[-1]) + 1
        
        df_classic_values = classic_values[start_idx:end_idx]
        df_hyper_values = hyper_values[start_idx:end_idx]
        df_hyper_linear_vol = hyper_linear_vol[start_idx:end_idx]
        df_hyper_nonlinear_vol = hyper_nonlinear_vol[start_idx:end_idx]
        df_hyper_adaptive_vol = hyper_adaptive_vol[start_idx:end_idx]
        
        # インジケーターデータをデータフレームに追加
        df = df.assign(
            classic_er=df_classic_values,
            hyper_er=df_hyper_values,
            hyper_linear_vol=df_hyper_linear_vol,
            hyper_nonlinear_vol=df_hyper_nonlinear_vol,
            hyper_adaptive_vol=df_hyper_adaptive_vol
        )
        
        # シグナル強度の計算（ERのトレンド方向色分け用）
        df['classic_er_strong'] = np.where(df['classic_er'] > 0.618, df['classic_er'], np.nan)
        df['classic_er_weak'] = np.where(df['classic_er'] <= 0.618, df['classic_er'], np.nan)
        df['hyper_er_strong'] = np.where(df['hyper_er'] > 0.618, df['hyper_er'], np.nan)
        df['hyper_er_weak'] = np.where(df['hyper_er'] <= 0.618, df['hyper_er'], np.nan)
        
        # ER差分の計算
        df['er_diff'] = df['hyper_er'] - df['classic_er']
        
        print(f"📈 Chart data prepared - Rows: {len(df)}")
        print(f"📊 Valid ER data - Classic: {~df['classic_er'].isna().sum()}, Hyper: {~df['hyper_er'].isna().sum()}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 1. メインチャート上のプロット（移動平均など必要に応じて）
        
        # 2. ERパネル（パネル1または2）
        er_panel_idx = 2 if show_volume else 1
        
        # Classic ER (強弱で色分け)
        main_plots.append(mpf.make_addplot(df['classic_er_strong'], panel=er_panel_idx, color='blue', width=1.5, 
                                          alpha=0.8, label='Classic ER (Strong)', secondary_y=False))
        main_plots.append(mpf.make_addplot(df['classic_er_weak'], panel=er_panel_idx, color='lightblue', width=1.0, 
                                          alpha=0.6, label='Classic ER (Weak)', secondary_y=False))
        
        # Hyper ER (強弱で色分け)
        main_plots.append(mpf.make_addplot(df['hyper_er_strong'], panel=er_panel_idx, color='red', width=1.5, 
                                          alpha=0.8, label='Hyper ER (Strong)', secondary_y=False))
        main_plots.append(mpf.make_addplot(df['hyper_er_weak'], panel=er_panel_idx, color='pink', width=1.0, 
                                          alpha=0.6, label='Hyper ER (Weak)', secondary_y=False))
        
        # 3. ER差分パネル
        diff_panel_idx = er_panel_idx + 1
        main_plots.append(mpf.make_addplot(df['er_diff'], panel=diff_panel_idx, color='green', width=1.2, 
                                          ylabel='ER Difference', secondary_y=False, label='Hyper - Classic'))
        
        # 4. ハイパーERボラティリティ成分パネル
        vol_panel_idx = diff_panel_idx + 1
        main_plots.append(mpf.make_addplot(df['hyper_linear_vol'], panel=vol_panel_idx, color='purple', width=1.0, 
                                          alpha=0.7, label='Linear Vol', secondary_y=False))
        main_plots.append(mpf.make_addplot(df['hyper_nonlinear_vol'], panel=vol_panel_idx, color='orange', width=1.0, 
                                          alpha=0.7, label='Nonlinear Vol', secondary_y=False))
        main_plots.append(mpf.make_addplot(df['hyper_adaptive_vol'], panel=vol_panel_idx, color='brown', width=1.0, 
                                          alpha=0.7, label='Adaptive Vol', secondary_y=False))
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        # パネル比率の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 2, 1, 1.5)  # メイン:出来高:ER:差分:ボラティリティ
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 2, 1, 1.5)  # メイン:ER:差分:ボラティリティ
        
        kwargs['addplot'] = main_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 各パネルに参照線とラベルを追加
        if show_volume:
            # ERパネル（パネル2）
            axes[2].axhline(y=0.618, color='gold', linestyle='--', alpha=0.8, linewidth=2, label='Golden Ratio')
            axes[2].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            axes[2].set_ylabel('Efficiency Ratio', fontsize=10)
            axes[2].legend(loc='upper left', fontsize=8)
            
            # ER差分パネル（パネル3）
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            axes[3].set_ylabel('ER Difference', fontsize=10)
            
            # ボラティリティ成分パネル（パネル4）
            axes[4].set_ylabel('Volatility Components', fontsize=10)
            axes[4].legend(loc='upper left', fontsize=8)
        else:
            # ERパネル（パネル1）
            axes[1].axhline(y=0.618, color='gold', linestyle='--', alpha=0.8, linewidth=2, label='Golden Ratio')
            axes[1].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            axes[1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            axes[1].set_ylabel('Efficiency Ratio', fontsize=10)
            axes[1].legend(loc='upper left', fontsize=8)
            
            # ER差分パネル（パネル2）
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            axes[2].set_ylabel('ER Difference', fontsize=10)
            
            # ボラティリティ成分パネル（パネル3）
            axes[3].set_ylabel('Volatility Components', fontsize=10)
            axes[3].legend(loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n=== Efficiency Ratio Statistics ===")
        
        # 有効データのマスク
        valid_mask = ~(np.isnan(df['classic_er']) | np.isnan(df['hyper_er']))
        valid_data = df[valid_mask]
        
        if len(valid_data) > 0:
            print(f"Valid data points: {len(valid_data)}")
            print(f"Classic ER - Mean: {valid_data['classic_er'].mean():.4f}, Std: {valid_data['classic_er'].std():.4f}")
            print(f"Hyper ER - Mean: {valid_data['hyper_er'].mean():.4f}, Std: {valid_data['hyper_er'].std():.4f}")
            print(f"ER Difference - Mean: {valid_data['er_diff'].mean():.4f}, Std: {valid_data['er_diff'].std():.4f}")
            
            # 相関係数
            correlation = valid_data['classic_er'].corr(valid_data['hyper_er'])
            print(f"Correlation: {correlation:.4f}")
            
            # 強いシグナルの割合
            classic_strong_pct = (valid_data['classic_er'] > 0.618).mean() * 100
            hyper_strong_pct = (valid_data['hyper_er'] > 0.618).mean() * 100
            print(f"Strong signals (>0.618) - Classic: {classic_strong_pct:.1f}%, Hyper: {hyper_strong_pct:.1f}%")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Efficiency Ratio Comparison Chart')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--start', '-s', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--classic-period', type=int, default=14, help='Classic ER period')
    parser.add_argument('--hyper-period', type=int, default=14, help='Hyper ER period')
    parser.add_argument('--classic-src', type=str, default='hlc3', help='Classic ER source type')
    parser.add_argument('--hyper-src', type=str, default='hlc3', help='Hyper ER source type')
    parser.add_argument('--hyper-slope', type=int, default=3, help='Hyper ER slope index')
    parser.add_argument('--hyper-threshold', type=float, default=0.3, help='Hyper ER threshold')
    parser.add_argument('--no-volume', action='store_true', help='Hide volume panel')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ERComparisonChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        classic_period=args.classic_period,
        classic_src_type=args.classic_src,
        hyper_period=args.hyper_period,
        hyper_src_type=args.hyper_src,
        hyper_slope_index=args.hyper_slope,
        hyper_threshold=args.hyper_threshold
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 