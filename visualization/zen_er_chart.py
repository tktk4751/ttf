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
from indicators.zen_efficiency_ratio import ZenEfficiencyRatio


class ZenERChart:
    """
    ZEN効率比を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ZEN効率比値
    - トレンド強度
    - ノイズレベル
    - ヒルベルト振幅
    - ウェーブレットエネルギー
    - カルマン速度
    - サイクル位相
    - 品質スコア表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.zen_er = None
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
        
        # CSVデータソースはダミーとして渡す
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
                            base_period: int = 14,
                            src_type: str = 'hlc3',
                            adaptive_factor: float = 0.618,
                            kalman_noise_ratio: float = 0.1,
                            wavelet_threshold: float = 0.05,
                            hilbert_window: int = 21,
                            adaptation_speed: float = 0.5,
                            min_period: int = 5,
                            max_period: int = 50,
                            use_dynamic_period: bool = True,
                            detector_type: str = 'absolute_ultimate',
                            cycle_part: float = 1.0,
                            max_cycle: int = 120,
                            min_cycle: int = 5) -> None:
        """
        ZEN効率比を計算する
        
        Args:
            base_period: 基本計算期間
            src_type: 価格ソース
            adaptive_factor: 適応係数
            kalman_noise_ratio: カルマンフィルターノイズ比
            wavelet_threshold: ウェーブレットしきい値
            hilbert_window: ヒルベルト変換ウィンドウサイズ
            adaptation_speed: 適応速度
            min_period: 最小計算期間
            max_period: 最大計算期間
            use_dynamic_period: 動的期間使用フラグ
            detector_type: 検出器タイプ
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nZEN効率比を計算中...")
        
        # ZEN効率比を計算
        self.zen_er = ZenEfficiencyRatio(
            base_period=base_period,
            src_type=src_type,
            adaptive_factor=adaptive_factor,
            kalman_noise_ratio=kalman_noise_ratio,
            wavelet_threshold=wavelet_threshold,
            hilbert_window=hilbert_window,
            adaptation_speed=adaptation_speed,
            min_period=min_period,
            max_period=max_period,
            use_dynamic_period=use_dynamic_period,
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle
        )
        
        # ZEN_ERの計算
        print("計算を実行します...")
        result = self.zen_er.calculate(self.data)
        
        # 結果の検証
        print(f"計算完了 - データ点数: {len(result.zen_er)}")
        print(f"ZEN_ER - 平均: {np.nanmean(result.zen_er):.3f}, 範囲: {np.nanmin(result.zen_er):.3f} - {np.nanmax(result.zen_er):.3f}")
        print(f"トレンド強度 - 平均: {np.nanmean(result.trend_strength):.3f}")
        print(f"ノイズレベル - 平均: {np.nanmean(result.noise_level):.3f}")
        print(f"品質スコア: {result.quality_score:.3f}")
        print(f"現在のトレンド: {result.current_trend}")
        
        # NaN値のチェック
        nan_count = np.isnan(result.zen_er).sum()
        print(f"NaN値: {nan_count}個 ({nan_count/len(result.zen_er)*100:.1f}%)")
        
        print("ZEN効率比計算完了")
            
    def plot(self, 
            title: str = "ZEN効率比（超高精度効率比）", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとZEN効率比を描画する
        
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
            
        if self.zen_er is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ZEN効率比の結果を取得
        print("ZEN_ER結果を取得中...")
        result = self.zen_er._result
        if result is None:
            raise ValueError("ZEN_ER結果が取得できません。")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'zen_er': result.zen_er,
                'trend_strength': result.trend_strength,
                'noise_level': result.noise_level,
                'hilbert_amplitude': result.hilbert_amplitude,
                'wavelet_energy': result.wavelet_energy,
                'kalman_velocity': result.kalman_velocity,
                'cycle_phase': result.cycle_phase,
                'instantaneous_period': result.instantaneous_period
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド方向に基づく色分け用データ
        df['zen_er_up'] = np.where(df['trend_strength'] > 0.1, df['zen_er'], np.nan)
        df['zen_er_down'] = np.where(df['trend_strength'] < -0.1, df['zen_er'], np.nan)
        df['zen_er_neutral'] = np.where(np.abs(df['trend_strength']) <= 0.1, df['zen_er'], np.nan)
        
        # 効率比の強度レベル線
        df['er_high_threshold'] = 0.8  # 高効率閾値
        df['er_medium_threshold'] = 0.6  # 中効率閾値
        df['er_low_threshold'] = 0.4  # 低効率閾値
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # パネルの設定
        if show_volume:
            panel_ratios = (4, 1, 1.5, 1, 1, 0.8, 0.8, 0.8)  # メイン:出来高:ZEN_ER:トレンド強度:ノイズ:振幅:エネルギー:速度
            base_panel = 2  # 出来高の次から開始
        else:
            panel_ratios = (4, 1.5, 1, 1, 0.8, 0.8, 0.8)  # メイン:ZEN_ER:トレンド強度:ノイズ:振幅:エネルギー:速度
            base_panel = 1  # メインの次から開始
        
        # ZEN効率比パネル（メインパネル）
        zen_er_up_plot = mpf.make_addplot(df['zen_er_up'], panel=base_panel, color='green', width=1.5, 
                                         ylabel='ZEN Efficiency Ratio', secondary_y=False, label='ZEN_ER (Up)')
        zen_er_down_plot = mpf.make_addplot(df['zen_er_down'], panel=base_panel, color='red', width=1.5, 
                                           secondary_y=False, label='ZEN_ER (Down)')
        zen_er_neutral_plot = mpf.make_addplot(df['zen_er_neutral'], panel=base_panel, color='gray', width=1.0, 
                                              secondary_y=False, label='ZEN_ER (Neutral)')
        
        # 効率比閾値線
        high_threshold_plot = mpf.make_addplot(df['er_high_threshold'], panel=base_panel, color='darkgreen', 
                                              linestyle='--', width=0.8, alpha=0.7, secondary_y=False)
        medium_threshold_plot = mpf.make_addplot(df['er_medium_threshold'], panel=base_panel, color='orange', 
                                                linestyle='--', width=0.8, alpha=0.7, secondary_y=False)
        low_threshold_plot = mpf.make_addplot(df['er_low_threshold'], panel=base_panel, color='red', 
                                             linestyle='--', width=0.8, alpha=0.7, secondary_y=False)
        
        # トレンド強度パネル
        trend_panel = mpf.make_addplot(df['trend_strength'], panel=base_panel+1, color='purple', width=1.5, 
                                      ylabel='Trend Strength', secondary_y=False, label='Trend Strength')
        
        # ノイズレベルパネル
        noise_panel = mpf.make_addplot(df['noise_level'], panel=base_panel+2, color='orange', width=1.2, 
                                      ylabel='Noise Level', secondary_y=False, label='Noise Level')
        
        # ヒルベルト振幅パネル
        hilbert_panel = mpf.make_addplot(df['hilbert_amplitude'], panel=base_panel+3, color='cyan', width=1.0, 
                                        ylabel='Hilbert Amplitude', secondary_y=False, label='Hilbert Amp')
        
        # ウェーブレットエネルギーパネル
        wavelet_panel = mpf.make_addplot(df['wavelet_energy'], panel=base_panel+4, color='magenta', width=1.0, 
                                        ylabel='Wavelet Energy', secondary_y=False, label='Wavelet Energy')
        
        # カルマン速度パネル
        velocity_panel = mpf.make_addplot(df['kalman_velocity'], panel=base_panel+5, color='brown', width=1.0, 
                                         ylabel='Kalman Velocity', secondary_y=False, label='Kalman Velocity')
        
        # すべてのプロットを結合
        all_plots = [
            zen_er_up_plot, zen_er_down_plot, zen_er_neutral_plot,
            high_threshold_plot, medium_threshold_plot, low_threshold_plot,
            trend_panel, noise_panel, hilbert_panel, wavelet_panel, velocity_panel
        ]
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=f"{title} (品質スコア: {result.quality_score:.3f})",
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            volume=show_volume,
            panel_ratios=panel_ratios,
            addplot=all_plots,
            returnfig=True
        )
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_idx = base_panel
        
        # ZEN効率比パネルの参照線
        axes[panel_idx].axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.5, label='High Efficiency')
        axes[panel_idx].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium Efficiency')
        axes[panel_idx].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Low Efficiency')
        axes[panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_idx].set_ylim(-0.05, 1.05)
        
        # トレンド強度パネルの参照線
        panel_idx += 1
        axes[panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.5)
        axes[panel_idx].axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
        axes[panel_idx].axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
        axes[panel_idx].set_ylim(-1.1, 1.1)
        
        # ノイズレベルパネルの参照線
        panel_idx += 1
        axes[panel_idx].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        axes[panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_idx].set_ylim(-0.05, 1.05)
        
        # ヒルベルト振幅パネル
        panel_idx += 1
        amp_mean = df['hilbert_amplitude'].mean()
        axes[panel_idx].axhline(y=amp_mean, color='cyan', linestyle='--', alpha=0.5)
        
        # ウェーブレットエネルギーパネル
        panel_idx += 1
        energy_mean = df['wavelet_energy'].mean()
        axes[panel_idx].axhline(y=energy_mean, color='magenta', linestyle='--', alpha=0.5)
        
        # カルマン速度パネル
        panel_idx += 1
        axes[panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.5)
        
        # 統計情報の表示
        self._print_statistics(df, result)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def _print_statistics(self, df: pd.DataFrame, result) -> None:
        """統計情報の出力"""
        print(f"\n=== ZEN効率比統計 ===")
        print(f"品質スコア: {result.quality_score:.3f}")
        print(f"現在のトレンド: {result.current_trend}")
        
        # 効率比レベル分析
        high_eff = (df['zen_er'] >= 0.8).sum()
        medium_eff = ((df['zen_er'] >= 0.6) & (df['zen_er'] < 0.8)).sum()
        low_eff = ((df['zen_er'] >= 0.4) & (df['zen_er'] < 0.6)).sum()
        very_low_eff = (df['zen_er'] < 0.4).sum()
        total_valid = (~df['zen_er'].isna()).sum()
        
        if total_valid > 0:
            print(f"\n効率比レベル分析:")
            print(f"高効率 (≥0.8): {high_eff}点 ({high_eff/total_valid*100:.1f}%)")
            print(f"中効率 (0.6-0.8): {medium_eff}点 ({medium_eff/total_valid*100:.1f}%)")
            print(f"低効率 (0.4-0.6): {low_eff}点 ({low_eff/total_valid*100:.1f}%)")
            print(f"非効率 (<0.4): {very_low_eff}点 ({very_low_eff/total_valid*100:.1f}%)")
        
        # トレンド分析
        uptrend = (df['trend_strength'] > 0.3).sum()
        downtrend = (df['trend_strength'] < -0.3).sum()
        range_trend = (np.abs(df['trend_strength']) <= 0.3).sum()
        
        print(f"\nトレンド分析:")
        print(f"上昇トレンド: {uptrend}点 ({uptrend/total_valid*100:.1f}%)")
        print(f"下降トレンド: {downtrend}点 ({downtrend/total_valid*100:.1f}%)")
        print(f"レンジ相場: {range_trend}点 ({range_trend/total_valid*100:.1f}%)")
        
        # ノイズ分析
        low_noise = (df['noise_level'] < 0.3).sum()
        medium_noise = ((df['noise_level'] >= 0.3) & (df['noise_level'] < 0.7)).sum()
        high_noise = (df['noise_level'] >= 0.7).sum()
        
        print(f"\nノイズレベル分析:")
        print(f"低ノイズ (<0.3): {low_noise}点 ({low_noise/total_valid*100:.1f}%)")
        print(f"中ノイズ (0.3-0.7): {medium_noise}点 ({medium_noise/total_valid*100:.1f}%)")
        print(f"高ノイズ (≥0.7): {high_noise}点 ({high_noise/total_valid*100:.1f}%)")
        
        # 数値統計
        print(f"\n数値統計:")
        print(f"ZEN_ER - 平均: {df['zen_er'].mean():.3f}, 標準偏差: {df['zen_er'].std():.3f}")
        print(f"トレンド強度 - 平均: {df['trend_strength'].mean():.3f}, 標準偏差: {df['trend_strength'].std():.3f}")
        print(f"ノイズレベル - 平均: {df['noise_level'].mean():.3f}, 標準偏差: {df['noise_level'].std():.3f}")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='ZEN効率比の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--base-period', type=int, default=14, help='基本計算期間')
    parser.add_argument('--adaptive-factor', type=float, default=0.618, help='適応係数')
    parser.add_argument('--kalman-noise', type=float, default=0.1, help='カルマンノイズ比')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ZenERChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        base_period=args.base_period,
        src_type=args.src_type,
        adaptive_factor=args.adaptive_factor,
        kalman_noise_ratio=args.kalman_noise
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 