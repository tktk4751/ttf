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
from indicators.mama import MAMA


class MAMAChart:
    """
    MAMA/FAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - MAMAライン（適応型移動平均）
    - FAMAライン（フォロー適応型移動平均）
    - Period値（適応期間）
    - Alpha値（適応係数）
    - Phase値（位相）
    - InPhase/Quadrature成分
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.mama_indicator = None
        self.mama_result = None
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
        print("\nデータを読み込み・処理中...")
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

    def calculate_indicators(self,
                            fast_limit: float = 0.5,
                            slow_limit: float = 0.05,
                            src_type: str = 'ukf_hlc3'
                           ) -> None:
        """
        MAMA/FAMAインジケーターを計算する
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nMAMA/FAMAインジケーターを計算中...")
        
        # MAMAインジケーターを初期化
        self.mama_indicator = MAMA(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type
        )
        
        # MAMA/FAMAの計算
        print("計算を実行します...")
        self.mama_result = self.mama_indicator.calculate(self.data)
        
        # 結果の検証
        mama_values = self.mama_result.mama_values
        fama_values = self.mama_result.fama_values
        period_values = self.mama_result.period_values
        alpha_values = self.mama_result.alpha_values
        
        print(f"MAMA/FAMA計算完了 - データ長: {len(mama_values)}")
        
        # NaN値のチェック
        nan_count_mama = np.isnan(mama_values).sum()
        nan_count_fama = np.isnan(fama_values).sum()
        nan_count_period = np.isnan(period_values).sum()
        nan_count_alpha = np.isnan(alpha_values).sum()
        
        print(f"NaN値 - MAMA: {nan_count_mama}, FAMA: {nan_count_fama}, Period: {nan_count_period}, Alpha: {nan_count_alpha}")
        
        # 有効値の統計
        valid_mama = mama_values[~np.isnan(mama_values)]
        valid_fama = fama_values[~np.isnan(fama_values)]
        valid_period = period_values[~np.isnan(period_values)]
        valid_alpha = alpha_values[~np.isnan(alpha_values)]
        
        if len(valid_mama) > 0:
            print(f"MAMA値範囲: {valid_mama.min():.4f} - {valid_mama.max():.4f}")
        if len(valid_fama) > 0:
            print(f"FAMA値範囲: {valid_fama.min():.4f} - {valid_fama.max():.4f}")
        if len(valid_period) > 0:
            print(f"Period値範囲: {valid_period.min():.2f} - {valid_period.max():.2f}")
        if len(valid_alpha) > 0:
            print(f"Alpha値範囲: {valid_alpha.min():.4f} - {valid_alpha.max():.4f}")
        
        print("MAMA/FAMA計算完了")
            
    def plot(self, 
            title: str = "MAMA/FAMA Adaptive Moving Average", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_period: bool = True,
            show_alpha: bool = True,
            show_phase: bool = False,
            show_inphase_quadrature: bool = False,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとMAMA/FAMAを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_period: Period値を表示するか
            show_alpha: Alpha値を表示するか
            show_phase: Phase値を表示するか
            show_inphase_quadrature: InPhase/Quadrature成分を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.mama_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # MAMA/FAMAの値を取得
        print("MAMA/FAMAデータを取得中...")
        mama_values = self.mama_result.mama_values
        fama_values = self.mama_result.fama_values
        period_values = self.mama_result.period_values
        alpha_values = self.mama_result.alpha_values
        phase_values = self.mama_result.phase_values
        i1_values = self.mama_result.i1_values
        q1_values = self.mama_result.q1_values
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'mama': mama_values,
                'fama': fama_values,
                'period': period_values,
                'alpha': alpha_values,
                'phase': phase_values,
                'i1': i1_values,
                'q1': q1_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"MAMA/FAMAデータ確認 - MAMA NaN: {df['mama'].isna().sum()}, FAMA NaN: {df['fama'].isna().sum()}")
        
        # MAMAとFAMAのクロスオーバー検出
        df['mama_above_fama'] = (df['mama'] > df['fama']).fillna(False)
        df['mama_below_fama'] = (df['mama'] < df['fama']).fillna(False)
        
        # クロスオーバー信号の検出
        mama_above_prev = df['mama_above_fama'].shift(1).fillna(False)
        mama_below_prev = df['mama_below_fama'].shift(1).fillna(False)
        
        df['crossover_up'] = df['mama_above_fama'] & ~mama_above_prev
        df['crossover_down'] = df['mama_below_fama'] & ~mama_below_prev
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # MAMA/FAMAのプロット設定
        main_plots.append(mpf.make_addplot(df['mama'], color='blue', width=2.5, alpha=0.8, label='MAMA'))
        main_plots.append(mpf.make_addplot(df['fama'], color='red', width=2.0, alpha=0.8, label='FAMA'))
        
        # クロスオーバー信号のプロット
        mama_cross_up = df['mama'].where(df['crossover_up'])
        mama_cross_down = df['mama'].where(df['crossover_down'])
        main_plots.append(mpf.make_addplot(mama_cross_up, type='scatter', markersize=50, 
                                         color='green', marker='^', alpha=0.8, label='MAMA Cross Up'))
        main_plots.append(mpf.make_addplot(mama_cross_down, type='scatter', markersize=50, 
                                         color='red', marker='v', alpha=0.8, label='MAMA Cross Down'))
        
        # 2. サブパネルのプロット
        sub_plots = []
        
        # 出来高の有無を考慮したパネル番号のベースを決定
        base_panel = 1 if show_volume else 0
        panel_count = base_panel
        
        # Period値パネル
        if show_period:
            panel_count += 1
            period_panel = mpf.make_addplot(df['period'], panel=panel_count, color='purple', width=1.5, 
                                           ylabel='Period', secondary_y=False, label='Period')
            sub_plots.append(period_panel)
        
        # Alpha値パネル
        if show_alpha:
            panel_count += 1
            alpha_panel = mpf.make_addplot(df['alpha'], panel=panel_count, color='orange', width=1.5, 
                                          ylabel='Alpha', secondary_y=False, label='Alpha')
            sub_plots.append(alpha_panel)
        
        # Phase値パネル
        if show_phase:
            panel_count += 1
            phase_panel = mpf.make_addplot(df['phase'], panel=panel_count, color='green', width=1.5, 
                                          ylabel='Phase', secondary_y=False, label='Phase')
            sub_plots.append(phase_panel)
        
        # InPhase/Quadratureパネル
        if show_inphase_quadrature:
            panel_count += 1
            i1_panel = mpf.make_addplot(df['i1'], panel=panel_count, color='cyan', width=1.2, 
                                       ylabel='I1/Q1', secondary_y=False, label='InPhase')
            panel_count += 1
            q1_panel = mpf.make_addplot(df['q1'], panel=panel_count, color='magenta', width=1.2, 
                                       ylabel='', secondary_y=False, label='Quadrature')
            sub_plots.extend([i1_panel, q1_panel])
        
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
        
        # 出来高とパネル比率の設定
        panel_ratios = [4]  # メインチャート
        
        if show_volume:
            kwargs['volume'] = True
            panel_ratios.append(1)  # 出来高
        else:
            kwargs['volume'] = False
        
        # サブパネルの比率を追加
        if show_period:
            panel_ratios.append(1)
        if show_alpha:
            panel_ratios.append(1)
        if show_phase:
            panel_ratios.append(1)
        if show_inphase_quadrature:
            panel_ratios.extend([1, 1])
        
        if len(panel_ratios) > 1:
            kwargs['panel_ratios'] = panel_ratios
        
        # すべてのプロットを結合
        all_plots = main_plots + sub_plots
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['MAMA', 'FAMA', 'MAMA Cross Up', 'MAMA Cross Down'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各サブパネルに参照線を追加
        # axes配列は[メインチャート, 出来高（オプション）, サブパネル1, サブパネル2, ...]の順番
        panel_idx = 1 if show_volume else 0
        
        if show_period:
            panel_idx += 1
            if panel_idx < len(axes):
                # Period値の平均線を表示
                period_mean = df['period'].mean()
                axes[panel_idx].axhline(y=period_mean, color='black', linestyle='--', alpha=0.5)
                axes[panel_idx].axhline(y=20, color='gray', linestyle=':', alpha=0.5)  # 標準期間
        
        if show_alpha:
            panel_idx += 1
            if panel_idx < len(axes):
                # Alpha値の参照線（fast_limit/slow_limit）
                axes[panel_idx].axhline(y=self.mama_indicator.fast_limit, color='red', linestyle='--', alpha=0.5)
                axes[panel_idx].axhline(y=self.mama_indicator.slow_limit, color='blue', linestyle='--', alpha=0.5)
        
        if show_phase:
            panel_idx += 1
            if panel_idx < len(axes):
                # Phase値の参照線（0, ±90, ±180度）
                axes[panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[panel_idx].axhline(y=90, color='gray', linestyle=':', alpha=0.5)
                axes[panel_idx].axhline(y=-90, color='gray', linestyle=':', alpha=0.5)
                axes[panel_idx].axhline(y=180, color='gray', linestyle=':', alpha=0.3)
                axes[panel_idx].axhline(y=-180, color='gray', linestyle=':', alpha=0.3)
        
        if show_inphase_quadrature:
            panel_idx += 1
            if panel_idx < len(axes):
                # InPhase参照線
                axes[panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            panel_idx += 1
            if panel_idx < len(axes):
                # Quadrature参照線
                axes[panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== MAMA/FAMA統計 ===")
        total_points = len(df.dropna())
        mama_above_points = len(df[df['mama'] > df['fama']])
        mama_below_points = len(df[df['mama'] < df['fama']])
        crossover_up_count = df['crossover_up'].sum()
        crossover_down_count = df['crossover_down'].sum()
        
        print(f"総データ点数: {total_points}")
        print(f"MAMA > FAMA: {mama_above_points} ({mama_above_points/total_points*100:.1f}%)")
        print(f"MAMA < FAMA: {mama_below_points} ({mama_below_points/total_points*100:.1f}%)")
        print(f"クロスオーバー上向き: {crossover_up_count}")
        print(f"クロスオーバー下向き: {crossover_down_count}")
        
        if not df['period'].isna().all():
            print(f"Period値 - 平均: {df['period'].mean():.2f}, 範囲: {df['period'].min():.2f} - {df['period'].max():.2f}")
        if not df['alpha'].isna().all():
            print(f"Alpha値 - 平均: {df['alpha'].mean():.4f}, 範囲: {df['alpha'].min():.4f} - {df['alpha'].max():.4f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='MAMA/FAMAインジケーターの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--fast-limit', type=float, default=0.5, help='高速制限値')
    parser.add_argument('--slow-limit', type=float, default=0.05, help='低速制限値')
    parser.add_argument('--src-type', type=str, default='hl2', help='ソースタイプ')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-period', action='store_true', help='Period値を非表示')
    parser.add_argument('--no-alpha', action='store_true', help='Alpha値を非表示')
    parser.add_argument('--show-phase', action='store_true', help='Phase値を表示')
    parser.add_argument('--show-iq', action='store_true', help='InPhase/Quadrature成分を表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = MAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        fast_limit=args.fast_limit,
        slow_limit=args.slow_limit,
        src_type=args.src_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_period=not args.no_period,
        show_alpha=not args.no_alpha,
        show_phase=args.show_phase,
        show_inphase_quadrature=args.show_iq,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 