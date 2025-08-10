#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Dict, Any, Optional, Tuple

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# 統合トレンド・サイクル検出器
from indicators.trend_filter.unified_trend_cycle_detector import UnifiedTrendCycleDetector


class UnifiedTrendCycleChart:
    """
    統合トレンド・サイクル検出器を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 統合トレンド強度とサイクル信頼度
    - 統合状態（買い: 緑、中立: グレー、売り: 赤）
    - 統合フェーザー角度
    - コンセンサス強度
    - 統合シグナル
    - 個別手法の貢献度比較
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.unified_detector = None
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
                            period: int = 20,
                            trend_length: int = 20,
                            trend_threshold: float = 0.5,
                            adaptability_factor: float = 0.7,
                            src_type: str = 'close',
                            enable_consensus_filter: bool = True,
                            min_consensus_threshold: float = 0.6
                           ) -> None:
        """
        統合トレンド・サイクル検出器を計算する
        
        Args:
            period: 基本サイクル分析周期
            trend_length: トレンド分析長
            trend_threshold: トレンド判定閾値
            adaptability_factor: 適応性係数
            src_type: ソースタイプ
            enable_consensus_filter: コンセンサスフィルター有効化
            min_consensus_threshold: 最小コンセンサス閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n統合トレンド・サイクル検出器を計算中...")
        
        # 統合検出器を初期化
        self.unified_detector = UnifiedTrendCycleDetector(
            period=period,
            trend_length=trend_length,
            trend_threshold=trend_threshold,
            adaptability_factor=adaptability_factor,
            src_type=src_type,
            enable_consensus_filter=enable_consensus_filter,
            min_consensus_threshold=min_consensus_threshold
        )
        
        # 統合検出器の計算
        print("計算を実行します...")
        result = self.unified_detector.calculate(self.data)
        
        # 結果の取得テスト
        trend_strength = result.unified_trend_strength
        cycle_confidence = result.unified_cycle_confidence
        unified_state = result.unified_state
        unified_signal = result.unified_signal
        consensus_strength = result.consensus_strength
        
        print(f"統合計算完了 - トレンド強度: {len(trend_strength)}, サイクル信頼度: {len(cycle_confidence)}")
        print(f"統合状態: {len(unified_state)}, 統合シグナル: {len(unified_signal)}")
        
        # NaN値のチェック
        nan_count_trend = np.isnan(trend_strength).sum()
        nan_count_cycle = np.isnan(cycle_confidence).sum()
        state_count = (unified_state != 0).sum()
        signal_count = (unified_signal != 0).sum()
        
        print(f"NaN値 - トレンド強度: {nan_count_trend}, サイクル信頼度: {nan_count_cycle}")
        print(f"有効値 - 状態: {state_count}, シグナル: {signal_count}")
        print(f"コンセンサス強度平均: {np.nanmean(consensus_strength):.4f}")
        
        print("統合トレンド・サイクル検出器計算完了")
            
    def plot(self, 
            title: str = "統合トレンド・サイクル検出器", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと統合トレンド・サイクル検出器を描画する
        
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
            
        if self.unified_detector is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 統合検出器の値を取得
        print("統合検出器データを取得中...")
        result = self.unified_detector.calculate(self.data)
        
        # 各指標の取得
        trend_strength = result.unified_trend_strength
        cycle_confidence = result.unified_cycle_confidence
        unified_state = result.unified_state
        unified_signal = result.unified_signal
        phase_angle = result.unified_phase_angle
        consensus_strength = result.consensus_strength
        real_component = result.real_component
        imag_component = result.imag_component
        magnitude = result.magnitude
        
        # 個別手法の結果
        cycle_results = result.cycle_results
        trend_results = result.trend_results
        phasor_results = result.phasor_results
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_strength': trend_strength,
                'cycle_confidence': cycle_confidence,
                'unified_state': unified_state,
                'unified_signal': unified_signal,
                'phase_angle': phase_angle,
                'consensus_strength': consensus_strength,
                'real_component': real_component,
                'imag_component': imag_component,
                'magnitude': magnitude,
                'cycle_real': cycle_results['real'],
                'trend_correlation': trend_results['correlation'],
                'phasor_real': phasor_results['real']
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # 統合状態の色分け準備
        # 買い(+1): 緑、中立(0): グレー、売り(-1): 赤
        df['state_buy'] = np.where(df['unified_state'] == 1, 1, np.nan)
        df['state_neutral'] = np.where(df['unified_state'] == 0, 0, np.nan)
        df['state_sell'] = np.where(df['unified_state'] == -1, -1, np.nan)
        
        # シグナルの色分け
        df['signal_buy'] = np.where(df['unified_signal'] == 1, 1, np.nan)
        df['signal_sell'] = np.where(df['unified_signal'] == -1, -1, np.nan)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（トレンド強度とサイクル信頼度）
        main_plots = []
        
        # トレンド強度を価格に重ねて表示（透明度付き）
        price_range = df['high'].max() - df['low'].min()
        trend_overlay = df['low'].min() + (df['trend_strength'] * price_range * 0.2)
        cycle_overlay = df['low'].min() + (df['cycle_confidence'] * price_range * 0.15)
        
        main_plots.append(mpf.make_addplot(trend_overlay, color='blue', width=1, alpha=0.3, label='Trend Strength'))
        main_plots.append(mpf.make_addplot(cycle_overlay, color='purple', width=1, alpha=0.3, label='Cycle Confidence'))
        
        # 2. 統合状態パネル（買い: 緑、中立: グレー、売り: 赤）
        state_panel_num = 1
        if show_volume:
            state_panel_num = 2
            
        state_buy_plot = mpf.make_addplot(df['state_buy'], panel=state_panel_num, color='green', width=3, 
                                        ylabel='統合状態', secondary_y=False, label='買い', type='line')
        state_neutral_plot = mpf.make_addplot(df['state_neutral'], panel=state_panel_num, color='gray', width=3, 
                                            secondary_y=False, label='中立', type='line')
        state_sell_plot = mpf.make_addplot(df['state_sell'], panel=state_panel_num, color='red', width=3, 
                                         secondary_y=False, label='売り', type='line')
        
        # 3. 統合指標パネル
        indicators_panel_num = state_panel_num + 1
        trend_strength_plot = mpf.make_addplot(df['trend_strength'], panel=indicators_panel_num, color='blue', width=1.5, 
                                             ylabel='統合指標', secondary_y=False, label='トレンド強度')
        cycle_confidence_plot = mpf.make_addplot(df['cycle_confidence'], panel=indicators_panel_num, color='purple', width=1.5, 
                                               secondary_y=True, label='サイクル信頼度')
        
        # 4. コンセンサス強度パネル
        consensus_panel_num = indicators_panel_num + 1
        consensus_plot = mpf.make_addplot(df['consensus_strength'], panel=consensus_panel_num, color='orange', width=1.5, 
                                        ylabel='コンセンサス', secondary_y=False, label='コンセンサス強度')
        
        # 5. フェーザー成分パネル
        phasor_panel_num = consensus_panel_num + 1
        real_plot = mpf.make_addplot(df['real_component'], panel=phasor_panel_num, color='blue', width=1.2, 
                                   ylabel='フェーザー成分', secondary_y=False, label='Real成分')
        imag_plot = mpf.make_addplot(df['imag_component'], panel=phasor_panel_num, color='red', width=1.2, 
                                   secondary_y=False, label='Imag成分')
        magnitude_plot = mpf.make_addplot(df['magnitude'], panel=phasor_panel_num, color='black', width=1.5, 
                                        secondary_y=True, label='強度')
        
        # 6. 個別手法比較パネル
        comparison_panel_num = phasor_panel_num + 1
        cycle_method_plot = mpf.make_addplot(np.abs(df['cycle_real']), panel=comparison_panel_num, color='green', width=1.2, 
                                           ylabel='個別手法', secondary_y=False, label='サイクル手法')
        trend_method_plot = mpf.make_addplot(np.abs(df['trend_correlation']), panel=comparison_panel_num, color='blue', width=1.2, 
                                           secondary_y=False, label='トレンド手法')
        phasor_method_plot = mpf.make_addplot(np.abs(df['phasor_real']), panel=comparison_panel_num, color='purple', width=1.2, 
                                            secondary_y=False, label='フェーザー手法')
        
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
        
        # 出来高と追加パネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1.5, 1, 1, 1, 1)  # メイン:出来高:状態:指標:コンセンサス:フェーザー:比較
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1.5, 1, 1, 1, 1)  # メイン:状態:指標:コンセンサス:フェーザー:比較
        
        # すべてのプロットを結合
        all_plots = main_plots + [
            state_buy_plot, state_neutral_plot, state_sell_plot,
            trend_strength_plot, cycle_confidence_plot,
            consensus_plot,
            real_plot, imag_plot, magnitude_plot,
            cycle_method_plot, trend_method_plot, phasor_method_plot
        ]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Trend Strength', 'Cycle Confidence'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # 統合状態パネル
        state_ax = axes[1 + panel_offset]
        state_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        state_ax.axhline(y=1, color='green', linestyle='--', alpha=0.3)
        state_ax.axhline(y=-1, color='red', linestyle='--', alpha=0.3)
        state_ax.set_ylim(-1.5, 1.5)
        
        # 統合指標パネル
        indicators_ax = axes[2 + panel_offset]
        indicators_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        indicators_ax.axhline(y=0.7, color='blue', linestyle='--', alpha=0.3)
        indicators_ax.set_ylim(0, 1)
        
        # コンセンサス強度パネル
        consensus_ax = axes[3 + panel_offset]
        consensus_ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        consensus_ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3)
        consensus_ax.set_ylim(0, 1)
        
        # フェーザー成分パネル
        phasor_ax = axes[4 + panel_offset]
        phasor_ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        phasor_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        phasor_ax.axhline(y=-0.5, color='black', linestyle='--', alpha=0.3)
        
        # 個別手法比較パネル
        comparison_ax = axes[5 + panel_offset]
        comparison_ax.axhline(y=0.3, color='black', linestyle='--', alpha=0.3)
        comparison_ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        comparison_ax.set_ylim(0, 1)
        
        # 統計情報の表示
        print(f"\n=== 統合検出器統計 ===")
        total_points = len(df)
        buy_points = len(df[df['unified_state'] == 1])
        sell_points = len(df[df['unified_state'] == -1])
        neutral_points = len(df[df['unified_state'] == 0])
        
        buy_signals = len(df[df['unified_signal'] == 1])
        sell_signals = len(df[df['unified_signal'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"買い状態: {buy_points} ({buy_points/total_points*100:.1f}%)")
        print(f"売り状態: {sell_points} ({sell_points/total_points*100:.1f}%)")
        print(f"中立状態: {neutral_points} ({neutral_points/total_points*100:.1f}%)")
        print(f"買いシグナル: {buy_signals}回")
        print(f"売りシグナル: {sell_signals}回")
        print(f"平均トレンド強度: {df['trend_strength'].mean():.4f}")
        print(f"平均サイクル信頼度: {df['cycle_confidence'].mean():.4f}")
        print(f"平均コンセンサス強度: {df['consensus_strength'].mean():.4f}")
        
        # 個別手法の貢献度
        print(f"\n=== 個別手法の平均強度 ===")
        print(f"サイクル手法: {np.abs(df['cycle_real']).mean():.4f}")
        print(f"トレンド手法: {np.abs(df['trend_correlation']).mean():.4f}")
        print(f"フェーザー手法: {np.abs(df['phasor_real']).mean():.4f}")
        
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
    parser = argparse.ArgumentParser(description='統合トレンド・サイクル検出器の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=20, help='基本サイクル分析周期')
    parser.add_argument('--trend-length', type=int, default=20, help='トレンド分析長')
    parser.add_argument('--trend-threshold', type=float, default=0.6, help='トレンド判定閾値')
    parser.add_argument('--adaptability', type=float, default=0.7, help='適応性係数')
    parser.add_argument('--src-type', type=str, default='close', help='ソースタイプ')
    parser.add_argument('--no-consensus', action='store_true', help='コンセンサスフィルターを無効化')
    parser.add_argument('--consensus-threshold', type=float, default=0.6, help='最小コンセンサス閾値')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UnifiedTrendCycleChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        trend_length=args.trend_length,
        trend_threshold=args.trend_threshold,
        adaptability_factor=args.adaptability,
        src_type=args.src_type,
        enable_consensus_filter=not args.no_consensus,
        min_consensus_threshold=args.consensus_threshold
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()