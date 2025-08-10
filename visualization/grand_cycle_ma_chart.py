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
from indicators.grand_cycle_ma import GrandCycleMA


class GrandCycleMAChart:
    """
    グランドサイクルMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - グランドサイクルMAMA/FAMAライン
    - Alpha値（適応速度）
    - サイクル周期
    - トレンド方向の色分け表示
    - カルマンフィルター・スムーサーの設定表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.grand_cycle_ma = None
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
                            # グランドサイクルMAパラメータ
                            detector_type: str = 'hody',
                            fast_limit: float = 0.5,
                            slow_limit: float = 0.05,
                            src_type: str = 'hlc3',
                            cycle_part: float = 0.5,
                            max_cycle: int = 50,
                            min_cycle: int = 6,
                            max_output: int = 34,
                            min_output: int = 1,
                            # カルマンフィルター・スムーサーパラメータ
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'adaptive',
                            kalman_params: Optional[Dict] = None,
                            use_smoother: bool = True,
                            smoother_type: str = 'frama',
                            smoother_params: Optional[Dict] = None,
                            # サイクル検出器固有パラメータ
                            alpha: float = 0.07,
                            bandwidth: float = 0.6,
                            center_period: float = 15.0,
                            avg_length: float = 3.0,
                            window: int = 50,
                            period_range: Tuple[int, int] = (5, 120)
                           ) -> None:
        """
        グランドサイクルMAを計算する
        
        Args:
            detector_type: サイクル検出器のタイプ
            fast_limit: 高速制限値
            slow_limit: 低速制限値
            src_type: ソースタイプ
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            use_kalman_filter: カルマンフィルターの使用
            kalman_filter_type: カルマンフィルタータイプ
            kalman_params: カルマンフィルターパラメータ
            use_smoother: スムーサーの使用
            smoother_type: スムーサータイプ
            smoother_params: スムーサーパラメータ
            alpha: アルファパラメータ（特定の検出器用）
            bandwidth: 帯域幅
            center_period: 中心周期
            avg_length: 平均長
            window: 分析ウィンドウ長
            period_range: 周期範囲のタプル
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nグランドサイクルMAを計算中...")
        
        # グランドサイクルMAを計算
        self.grand_cycle_ma = GrandCycleMA(
            detector_type=detector_type,
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_params=kalman_params or {},
            use_smoother=use_smoother,
            smoother_type=smoother_type,
            smoother_params=smoother_params or {},
            alpha=alpha,
            bandwidth=bandwidth,
            center_period=center_period,
            avg_length=avg_length,
            window=window,
            period_range=period_range
        )
        
        # グランドサイクルMAの計算
        print("計算を実行します...")
        self.result = self.grand_cycle_ma.calculate(self.data)
        
        # 結果の取得テスト
        mama_values = self.result.grand_mama_values
        fama_values = self.result.grand_fama_values
        alpha_values = self.result.alpha_values
        cycle_period = self.result.cycle_period
        
        print(f"計算完了 - MAMA: {len(mama_values)}, FAMA: {len(fama_values)}, Alpha: {len(alpha_values)}, Cycle: {len(cycle_period)}")
        
        # NaN値のチェック
        nan_count_mama = np.isnan(mama_values).sum()
        nan_count_fama = np.isnan(fama_values).sum()
        nan_count_alpha = np.isnan(alpha_values).sum()
        nan_count_cycle = np.isnan(cycle_period).sum()
        
        print(f"NaN値 - MAMA: {nan_count_mama}, FAMA: {nan_count_fama}, Alpha: {nan_count_alpha}, Cycle: {nan_count_cycle}")
        
        # 有効データの統計
        valid_mama = mama_values[~np.isnan(mama_values)]
        valid_fama = fama_values[~np.isnan(fama_values)]
        valid_alpha = alpha_values[~np.isnan(alpha_values)]
        valid_cycle = cycle_period[~np.isnan(cycle_period)]
        
        if len(valid_mama) > 0:
            print(f"MAMA統計 - 平均: {np.mean(valid_mama):.4f}, 範囲: {np.min(valid_mama):.4f} - {np.max(valid_mama):.4f}")
        if len(valid_fama) > 0:
            print(f"FAMA統計 - 平均: {np.mean(valid_fama):.4f}, 範囲: {np.min(valid_fama):.4f} - {np.max(valid_fama):.4f}")
        if len(valid_alpha) > 0:
            print(f"Alpha統計 - 平均: {np.mean(valid_alpha):.4f}, 範囲: {np.min(valid_alpha):.4f} - {np.max(valid_alpha):.4f}")
        if len(valid_cycle) > 0:
            print(f"サイクル統計 - 平均: {np.mean(valid_cycle):.2f}, 範囲: {np.min(valid_cycle):.2f} - {np.max(valid_cycle):.2f}")
        
        print("グランドサイクルMA計算完了")
            
    def plot(self, 
            title: str = "グランドサイクルMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとグランドサイクルMAを描画する
        
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
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # グランドサイクルMAの値を取得
        print("グランドサイクルMAデータを取得中...")
        mama_values = self.result.grand_mama_values
        fama_values = self.result.grand_fama_values
        alpha_values = self.result.alpha_values
        cycle_period = self.result.cycle_period
        phase_values = self.result.phase_values
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'grand_mama': mama_values,
                'grand_fama': fama_values,
                'alpha_values': alpha_values,
                'cycle_period': cycle_period,
                'phase_values': phase_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"データ確認 - MAMA NaN: {df['grand_mama'].isna().sum()}, FAMA NaN: {df['grand_fama'].isna().sum()}")
        
        # トレンド方向の計算（MAMA > FAMA で上昇トレンド）
        df['trend_direction'] = np.where(
            (df['grand_mama'] > df['grand_fama']) & (~df['grand_mama'].isna()) & (~df['grand_fama'].isna()),
            1,  # 上昇トレンド
            np.where(
                (df['grand_mama'] < df['grand_fama']) & (~df['grand_mama'].isna()) & (~df['grand_fama'].isna()),
                -1,  # 下降トレンド
                0   # ニュートラル/不明
            )
        )
        
        # トレンド方向に基づく色分け
        df['mama_uptrend'] = np.where(df['trend_direction'] == 1, df['grand_mama'], np.nan)
        df['mama_downtrend'] = np.where(df['trend_direction'] == -1, df['grand_mama'], np.nan)
        df['fama_uptrend'] = np.where(df['trend_direction'] == 1, df['grand_fama'], np.nan)
        df['fama_downtrend'] = np.where(df['trend_direction'] == -1, df['grand_fama'], np.nan)
        
        # Alpha値の正規化（表示用）
        df['alpha_normalized'] = df['alpha_values'] * 100  # パーセント表示
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # グランドサイクルMAMAのプロット設定
        main_plots.append(mpf.make_addplot(df['mama_uptrend'], color='blue', width=2.5, label='MAMA (Up)'))
        main_plots.append(mpf.make_addplot(df['mama_downtrend'], color='red', width=2.5, label='MAMA (Down)'))
        
        # グランドサイクルFAMAのプロット設定
        main_plots.append(mpf.make_addplot(df['fama_uptrend'], color='cyan', width=2, alpha=0.8, label='FAMA (Up)'))
        main_plots.append(mpf.make_addplot(df['fama_downtrend'], color='orange', width=2, alpha=0.8, label='FAMA (Down)'))
        
        # 2. オシレータープロット
        # Alpha値パネル（適応速度）
        alpha_panel = mpf.make_addplot(df['alpha_normalized'], panel=1, color='purple', width=1.5, 
                                      ylabel='Alpha (%)', secondary_y=False, label='Alpha')
        
        # サイクル周期パネル
        cycle_panel = mpf.make_addplot(df['cycle_period'], panel=2, color='green', width=1.5, 
                                     ylabel='Cycle Period', secondary_y=False, label='Cycle')
        
        # トレンド方向パネル
        trend_panel = mpf.make_addplot(df['trend_direction'], panel=3, color='black', width=2, 
                                      ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
        
        # 設定情報をタイトルに追加
        config_info = []
        if hasattr(self.grand_cycle_ma, 'detector_type'):
            config_info.append(f"検出器: {self.grand_cycle_ma.detector_type}")
        if hasattr(self.grand_cycle_ma, 'use_kalman_filter') and self.grand_cycle_ma.use_kalman_filter:
            config_info.append(f"カルマン: {self.grand_cycle_ma.kalman_filter_type}")
        if hasattr(self.grand_cycle_ma, 'use_smoother') and self.grand_cycle_ma.use_smoother:
            config_info.append(f"スムーサー: {self.grand_cycle_ma.smoother_type}")
        
        full_title = f"{title} ({', '.join(config_info)})" if config_info else title
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=full_title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        # 出来高と追加パネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (5, 1, 1.5, 1.5, 1)  # メイン:出来高:Alpha:サイクル:トレンド
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            alpha_panel = mpf.make_addplot(df['alpha_normalized'], panel=2, color='purple', width=1.5, 
                                          ylabel='Alpha (%)', secondary_y=False, label='Alpha')
            cycle_panel = mpf.make_addplot(df['cycle_period'], panel=3, color='green', width=1.5, 
                                         ylabel='Cycle Period', secondary_y=False, label='Cycle')
            trend_panel = mpf.make_addplot(df['trend_direction'], panel=4, color='black', width=2, 
                                          ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1.5, 1.5, 1)  # メイン:Alpha:サイクル:トレンド
        
        # すべてのプロットを結合
        all_plots = main_plots + [alpha_panel, cycle_panel, trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['MAMA (Up)', 'MAMA (Down)', 'FAMA (Up)', 'FAMA (Down)'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # Alpha値パネル
        alpha_ax = axes[panel_offset]
        alpha_mean = df['alpha_normalized'].mean()
        alpha_ax.axhline(y=alpha_mean, color='black', linestyle='-', alpha=0.3, label=f'平均: {alpha_mean:.1f}%')
        alpha_ax.axhline(y=self.grand_cycle_ma.fast_limit * 100, color='red', linestyle='--', alpha=0.5, label=f'Fast: {self.grand_cycle_ma.fast_limit*100:.1f}%')
        alpha_ax.axhline(y=self.grand_cycle_ma.slow_limit * 100, color='blue', linestyle='--', alpha=0.5, label=f'Slow: {self.grand_cycle_ma.slow_limit*100:.1f}%')
        
        # サイクル周期パネル
        cycle_ax = axes[panel_offset + 1]
        cycle_mean = df['cycle_period'].mean()
        cycle_ax.axhline(y=cycle_mean, color='black', linestyle='-', alpha=0.3, label=f'平均: {cycle_mean:.1f}')
        cycle_ax.axhline(y=self.grand_cycle_ma.max_cycle, color='red', linestyle='--', alpha=0.5, label=f'Max: {self.grand_cycle_ma.max_cycle}')
        cycle_ax.axhline(y=self.grand_cycle_ma.min_cycle, color='blue', linestyle='--', alpha=0.5, label=f'Min: {self.grand_cycle_ma.min_cycle}')
        
        # トレンド方向パネル
        trend_ax = axes[panel_offset + 2]
        trend_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        trend_ax.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='上昇')
        trend_ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='下降')
        trend_ax.set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== グランドサイクルMA 統計 ===")
        total_points = len(df[df['trend_direction'] != 0])
        uptrend_points = len(df[df['trend_direction'] == 1])
        downtrend_points = len(df[df['trend_direction'] == -1])
        
        print(f"総データ点数: {total_points}")
        if total_points > 0:
            print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
            print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        
        valid_alpha = df['alpha_values'].dropna()
        valid_cycle = df['cycle_period'].dropna()
        
        if len(valid_alpha) > 0:
            print(f"Alpha値 - 平均: {valid_alpha.mean():.4f}, 範囲: {valid_alpha.min():.4f} - {valid_alpha.max():.4f}")
        if len(valid_cycle) > 0:
            print(f"サイクル周期 - 平均: {valid_cycle.mean():.2f}, 範囲: {valid_cycle.min():.2f} - {valid_cycle.max():.2f}")
        
        # パフォーマンス情報
        if hasattr(self.grand_cycle_ma, 'use_kalman_filter'):
            print(f"\n=== 設定情報 ===")
            print(f"カルマンフィルター: {'有効' if self.grand_cycle_ma.use_kalman_filter else '無効'}")
            if self.grand_cycle_ma.use_kalman_filter:
                print(f"  タイプ: {self.grand_cycle_ma.kalman_filter_type}")
            print(f"スムーサー: {'有効' if self.grand_cycle_ma.use_smoother else '無効'}")
            if self.grand_cycle_ma.use_smoother:
                print(f"  タイプ: {self.grand_cycle_ma.smoother_type}")
            print(f"サイクル検出器: {self.grand_cycle_ma.detector_type}")
        
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
    parser = argparse.ArgumentParser(description='グランドサイクルMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    
    # グランドサイクルMAパラメータ
    parser.add_argument('--detector', type=str, default='hody', help='サイクル検出器タイプ')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--fast-limit', type=float, default=0.5, help='高速制限値')
    parser.add_argument('--slow-limit', type=float, default=0.05, help='低速制限値')
    
    # フィルタリング・スムージングパラメータ
    parser.add_argument('--use-kalman', action='store_true', help='カルマンフィルターを使用')
    parser.add_argument('--kalman-type', type=str, default='adaptive', help='カルマンフィルタータイプ')
    parser.add_argument('--use-smoother', action='store_true', default=True, help='スムーサーを使用')
    parser.add_argument('--smoother-type', type=str, default='frama', help='スムーサータイプ')
    
    args = parser.parse_args()
    
    # チャートを作成
    chart = GrandCycleMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        detector_type=args.detector,
        src_type=args.src_type,
        fast_limit=args.fast_limit,
        slow_limit=args.slow_limit,
        use_kalman_filter=args.use_kalman,
        kalman_filter_type=args.kalman_type,
        use_smoother=args.use_smoother,
        smoother_type=args.smoother_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()