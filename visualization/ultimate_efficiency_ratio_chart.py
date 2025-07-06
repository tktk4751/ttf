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
from indicators.ultimate_efficiency_ratio import UltimateEfficiencyRatio


class UltimateEfficiencyRatioChart:
    """
    Ultimate Efficiency Ratioを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Ultimate Efficiency Ratio（統合効率率）
    - ハイパー効率率
    - トレンドシグナル
    - 量子解析結果（コヒーレンス、もつれ効果）
    - ヒルベルト変換結果（振幅、位相）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_er = None
        self.er_result = None
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
                            period: int = 14,
                            src_type: str = 'hlc3',
                            hilbert_window: int = 12,
                            her_window: int = 16,
                            slope_index: int = 3,
                            range_threshold: float = 0.003
                           ) -> None:
        """
        Ultimate Efficiency Ratioを計算する
        
        Args:
            period: 基本期間（従来ERとの互換性用）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            hilbert_window: ヒルベルト変換ウィンドウ
            her_window: ハイパー効率率ウィンドウ
            slope_index: トレンド判定期間
            range_threshold: レンジ判定しきい値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nUltimate Efficiency Ratioを計算中...")
        
        # Ultimate Efficiency Ratioを計算
        self.ultimate_er = UltimateEfficiencyRatio(
            period=period,
            src_type=src_type,
            hilbert_window=hilbert_window,
            her_window=her_window,
            slope_index=slope_index,
            range_threshold=range_threshold
        )
        
        # Ultimate ERの計算
        print("計算を実行します...")
        self.er_result = self.ultimate_er.calculate(self.data)
        
        # 計算結果の確認
        print(f"計算完了 - Ultimate ER: {len(self.er_result.values)}")
        print(f"ハイパー効率率: {len(self.er_result.hyper_efficiency)}")
        print(f"トレンドシグナル: {len(self.er_result.trend_signals)}")
        
        # NaN値のチェック
        nan_count_ultimate = np.isnan(self.er_result.values).sum()
        nan_count_hyper = np.isnan(self.er_result.hyper_efficiency).sum()
        trend_count = (self.er_result.trend_signals != 0).sum()
        
        print(f"NaN値 - Ultimate ER: {nan_count_ultimate}, ハイパーER: {nan_count_hyper}")
        print(f"トレンド値 - 有効: {trend_count}, 上昇: {(self.er_result.trend_signals == 1).sum()}, 下降: {(self.er_result.trend_signals == -1).sum()}")
        print(f"現在のトレンド: {self.er_result.current_trend}")
        
        print("Ultimate Efficiency Ratio計算完了")
            
    def plot(self, 
            title: str = "Ultimate Efficiency Ratio V3.0", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_quantum: bool = True,
            show_hilbert: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUltimate Efficiency Ratioを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_quantum: 量子解析結果を表示するか
            show_hilbert: ヒルベルト変換結果を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.er_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Ultimate ERの結果を取得
        print("Ultimate ER データを取得中...")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'ultimate_er': self.er_result.values,
                'hyper_efficiency': self.er_result.hyper_efficiency,
                'trend_signals': self.er_result.trend_signals,
                'trend_strength': self.er_result.trend_strength,
                'quantum_coherence': self.er_result.quantum_coherence,
                'quantum_entanglement': self.er_result.quantum_entanglement,
                'hilbert_amplitude': self.er_result.hilbert_amplitude,
                'hilbert_phase': self.er_result.hilbert_phase
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"Ultimate ER データ確認 - NaN: {df['ultimate_er'].isna().sum()}")
        
        # トレンド方向に基づく効率率の色分け
        df['er_uptrend'] = np.where(df['trend_signals'] == 1, df['ultimate_er'], np.nan)
        df['er_downtrend'] = np.where(df['trend_signals'] == -1, df['ultimate_er'], np.nan)
        df['er_range'] = np.where(df['trend_signals'] == 0, df['ultimate_er'], np.nan)
        
        # ハイパー効率率の色分け
        df['hyper_uptrend'] = np.where(df['trend_signals'] == 1, df['hyper_efficiency'], np.nan)
        df['hyper_downtrend'] = np.where(df['trend_signals'] == -1, df['hyper_efficiency'], np.nan)
        df['hyper_range'] = np.where(df['trend_signals'] == 0, df['hyper_efficiency'], np.nan)
        
        # 量子解析の正規化（0-1範囲へ）
        df['quantum_coherence_norm'] = df['quantum_coherence']
        df['quantum_entanglement_norm'] = df['quantum_entanglement']
        
        # ヒルベルト振幅の正規化
        if not df['hilbert_amplitude'].isna().all():
            amp_max = df['hilbert_amplitude'].max()
            if amp_max > 0:
                df['hilbert_amplitude_norm'] = df['hilbert_amplitude'] / amp_max
            else:
                df['hilbert_amplitude_norm'] = df['hilbert_amplitude']
        else:
            df['hilbert_amplitude_norm'] = df['hilbert_amplitude']
        
        # ヒルベルト位相の正規化（-π to π → 0 to 1）
        df['hilbert_phase_norm'] = (df['hilbert_phase'] + np.pi) / (2 * np.pi)
        
        # NaN値を含む行の確認
        nan_rows = df[df['ultimate_er'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
        
        # mplfinanceでプロット用の設定
        # メインチャート上のプロット（価格とトレンドシグナル用のマーカー）
        main_plots = []
        
        # トレンドシグナルの表示（価格チャート上）
        # 全データに対してシグナルマーカーを作成（NaNで非表示）
        up_markers = np.where(df['trend_signals'] == 1, df['low'] * 0.995, np.nan)
        down_markers = np.where(df['trend_signals'] == -1, df['high'] * 1.005, np.nan)
        
        # 上昇シグナルマーカー
        if not np.isnan(up_markers).all():
            main_plots.append(mpf.make_addplot(up_markers, type='scatter', 
                                             markersize=30, marker='^', color='green', alpha=0.7))
        
        # 下降シグナルマーカー
        if not np.isnan(down_markers).all():
            main_plots.append(mpf.make_addplot(down_markers, type='scatter', 
                                             markersize=30, marker='v', color='red', alpha=0.7))
        
        # パネル設定
        panel_num = 1
        
        # 効率率パネル
        er_panel = []
        er_panel.append(mpf.make_addplot(df['er_uptrend'], panel=panel_num, color='green', width=2, 
                                       ylabel='Efficiency Ratio', label='Ultimate ER (Up)'))
        er_panel.append(mpf.make_addplot(df['er_downtrend'], panel=panel_num, color='red', width=2, 
                                       label='Ultimate ER (Down)'))
        er_panel.append(mpf.make_addplot(df['er_range'], panel=panel_num, color='gray', width=1, 
                                       label='Ultimate ER (Range)'))
        er_panel.append(mpf.make_addplot(df['hyper_uptrend'], panel=panel_num, color='lightgreen', width=1.5, 
                                       linestyle='--', alpha=0.7, label='Hyper ER (Up)'))
        er_panel.append(mpf.make_addplot(df['hyper_downtrend'], panel=panel_num, color='lightcoral', width=1.5, 
                                       linestyle='--', alpha=0.7, label='Hyper ER (Down)'))
        panel_num += 1
        
        # トレンド強度パネル
        trend_panel = [mpf.make_addplot(df['trend_strength'], panel=panel_num, color='blue', width=1.5, 
                                      ylabel='Trend Strength', label='Trend Strength')]
        panel_num += 1
        
        # 量子解析パネル（オプション）
        quantum_panel = []
        if show_quantum:
            quantum_panel.append(mpf.make_addplot(df['quantum_coherence_norm'], panel=panel_num, color='purple', width=1.2, 
                                                ylabel='Quantum Analysis', label='Coherence'))
            quantum_panel.append(mpf.make_addplot(df['quantum_entanglement_norm'], panel=panel_num, color='magenta', width=1.2, 
                                                linestyle='--', alpha=0.8, label='Entanglement'))
            panel_num += 1
        
        # ヒルベルト変換パネル（オプション）
        hilbert_panel = []
        if show_hilbert:
            hilbert_panel.append(mpf.make_addplot(df['hilbert_amplitude_norm'], panel=panel_num, color='orange', width=1.2, 
                                                ylabel='Hilbert Transform', label='Amplitude'))
            hilbert_panel.append(mpf.make_addplot(df['hilbert_phase_norm'], panel=panel_num, color='brown', width=1.2, 
                                                linestyle=':', alpha=0.8, label='Phase'))
            panel_num += 1
        
        # パネル比率の計算
        total_panels = 1  # メインチャート
        if show_volume:
            total_panels += 1
        total_panels += 2  # 効率率 + トレンド強度
        if show_quantum:
            total_panels += 1
        if show_hilbert:
            total_panels += 1
        
        # パネル比率の設定
        if show_volume:
            if show_quantum and show_hilbert:
                panel_ratios = (4, 1, 1.5, 1, 1, 1)  # メイン:出来高:効率率:トレンド:量子:ヒルベルト
            elif show_quantum:
                panel_ratios = (4, 1, 1.5, 1, 1)     # メイン:出来高:効率率:トレンド:量子
            elif show_hilbert:
                panel_ratios = (4, 1, 1.5, 1, 1)     # メイン:出来高:効率率:トレンド:ヒルベルト
            else:
                panel_ratios = (4, 1, 1.5, 1)        # メイン:出来高:効率率:トレンド
        else:
            if show_quantum and show_hilbert:
                panel_ratios = (4, 1.5, 1, 1, 1)     # メイン:効率率:トレンド:量子:ヒルベルト
            elif show_quantum:
                panel_ratios = (4, 1.5, 1, 1)        # メイン:効率率:トレンド:量子
            elif show_hilbert:
                panel_ratios = (4, 1.5, 1, 1)        # メイン:効率率:トレンド:ヒルベルト
            else:
                panel_ratios = (4, 1.5, 1)           # メイン:効率率:トレンド
        
        # 出来高表示時のパネル番号調整
        if show_volume:
            # 出来高を表示する場合は、他のパネル番号を+1する
            er_panel = []
            er_panel.append(mpf.make_addplot(df['er_uptrend'], panel=2, color='green', width=2, 
                                           ylabel='Efficiency Ratio', label='Ultimate ER (Up)'))
            er_panel.append(mpf.make_addplot(df['er_downtrend'], panel=2, color='red', width=2, 
                                           label='Ultimate ER (Down)'))
            er_panel.append(mpf.make_addplot(df['er_range'], panel=2, color='gray', width=1, 
                                           label='Ultimate ER (Range)'))
            er_panel.append(mpf.make_addplot(df['hyper_uptrend'], panel=2, color='lightgreen', width=1.5, 
                                           linestyle='--', alpha=0.7, label='Hyper ER (Up)'))
            er_panel.append(mpf.make_addplot(df['hyper_downtrend'], panel=2, color='lightcoral', width=1.5, 
                                           linestyle='--', alpha=0.7, label='Hyper ER (Down)'))
            
            trend_panel = [mpf.make_addplot(df['trend_strength'], panel=3, color='blue', width=1.5, 
                                          ylabel='Trend Strength', label='Trend Strength')]
            
            panel_offset = 4
            if show_quantum:
                quantum_panel = []
                quantum_panel.append(mpf.make_addplot(df['quantum_coherence_norm'], panel=panel_offset, color='purple', width=1.2, 
                                                    ylabel='Quantum Analysis', label='Coherence'))
                quantum_panel.append(mpf.make_addplot(df['quantum_entanglement_norm'], panel=panel_offset, color='magenta', width=1.2, 
                                                    linestyle='--', alpha=0.8, label='Entanglement'))
                panel_offset += 1
            if show_hilbert:
                hilbert_panel = []
                hilbert_panel.append(mpf.make_addplot(df['hilbert_amplitude_norm'], panel=panel_offset, color='orange', width=1.2, 
                                                    ylabel='Hilbert Transform', label='Amplitude'))
                hilbert_panel.append(mpf.make_addplot(df['hilbert_phase_norm'], panel=panel_offset, color='brown', width=1.2, 
                                                    linestyle=':', alpha=0.8, label='Phase'))
        
        # すべてのプロットを結合
        all_plots = main_plots + er_panel + trend_panel
        if show_quantum and quantum_panel:
            all_plots.extend(quantum_panel)
        if show_hilbert and hilbert_panel:
            all_plots.extend(hilbert_panel)
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            panel_ratios=panel_ratios,
            volume=show_volume,
            addplot=all_plots
        )
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 参照線の追加
        try:
            panel_idx = 1 if show_volume else 0
            
            # 効率率パネルの参照線
            panel_idx += 1
            if panel_idx < len(axes):
                axes[panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Middle')
                axes[panel_idx].axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='High')
                axes[panel_idx].axhline(y=0.3, color='red', linestyle=':', alpha=0.5, label='Low')
                axes[panel_idx].set_ylim(0, 1)
            
            # トレンド強度パネルの参照線
            panel_idx += 1
            if panel_idx < len(axes):
                axes[panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                axes[panel_idx].axhline(y=0.7, color='green', linestyle=':', alpha=0.5)
                axes[panel_idx].axhline(y=0.3, color='red', linestyle=':', alpha=0.5)
                axes[panel_idx].set_ylim(0, 1)
            
            # 量子解析パネルの参照線
            if show_quantum:
                panel_idx += 1
                if panel_idx < len(axes):
                    axes[panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                    axes[panel_idx].axhline(y=0.7, color='purple', linestyle=':', alpha=0.5)
                    axes[panel_idx].set_ylim(0, 1)
            
            # ヒルベルト変換パネルの参照線
            if show_hilbert:
                panel_idx += 1
                if panel_idx < len(axes):
                    axes[panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
                    axes[panel_idx].set_ylim(0, 1)
        except Exception as e:
            print(f"参照線の追加でエラーが発生しました: {e}")
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n=== Ultimate Efficiency Ratio 統計 ===")
        valid_er = df['ultimate_er'].dropna()
        valid_hyper = df['hyper_efficiency'].dropna()
        total_signals = len(df[df['trend_signals'] != 0])
        uptrend_signals = len(df[df['trend_signals'] == 1])
        downtrend_signals = len(df[df['trend_signals'] == -1])
        
        print(f"総データ点数: {len(df)}")
        print(f"Ultimate ER - 平均: {valid_er.mean():.3f}, 範囲: {valid_er.min():.3f} - {valid_er.max():.3f}")
        print(f"Hyper ER - 平均: {valid_hyper.mean():.3f}, 範囲: {valid_hyper.min():.3f} - {valid_hyper.max():.3f}")
        print(f"値の違い確認 - Ultimate ER と Hyper ER は同じ？: {np.array_equal(valid_er.values, valid_hyper.values)}")
        print(f"トレンドシグナル - 総数: {total_signals}")
        print(f"上昇シグナル: {uptrend_signals} ({uptrend_signals/total_signals*100 if total_signals > 0 else 0:.1f}%)")
        print(f"下降シグナル: {downtrend_signals} ({downtrend_signals/total_signals*100 if total_signals > 0 else 0:.1f}%)")
        print(f"現在のトレンド: {self.er_result.current_trend}")
        
        if show_quantum:
            valid_coherence = df['quantum_coherence'].dropna()
            valid_entanglement = df['quantum_entanglement'].dropna()
            print(f"量子コヒーレンス - 平均: {valid_coherence.mean():.3f}")
            print(f"量子もつれ - 平均: {valid_entanglement.mean():.3f}")
        
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
    parser = argparse.ArgumentParser(description='Ultimate Efficiency Ratioの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=14, help='基本期間')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--hilbert-window', type=int, default=12, help='ヒルベルト変換ウィンドウ')
    parser.add_argument('--her-window', type=int, default=16, help='ハイパー効率率ウィンドウ')
    parser.add_argument('--slope-index', type=int, default=3, help='トレンド判定期間')
    parser.add_argument('--range-threshold', type=float, default=0.003, help='レンジ判定しきい値')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-quantum', action='store_true', help='量子解析を非表示')
    parser.add_argument('--no-hilbert', action='store_true', help='ヒルベルト変換を非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateEfficiencyRatioChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        src_type=args.src_type,
        hilbert_window=args.hilbert_window,
        her_window=args.her_window,
        slope_index=args.slope_index,
        range_threshold=args.range_threshold
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_quantum=not args.no_quantum,
        show_hilbert=not args.no_hilbert,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 