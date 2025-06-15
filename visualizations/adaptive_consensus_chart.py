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
from indicators.adaptive_consensus_cycle import AdaptiveConsensusCycle


class AdaptiveConsensusCycleChart:
    """
    適応的コンセンサスサイクル検出器を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 適応的コンセンサスサイクル値の表示
    - 各検出手法の結果（Phase Accumulator、Dual Differential、Homodyne、Bandpass Zero）
    - 信頼性スコアとコンセンサス重み
    - ノイズレベル分析
    - 統計情報とパラメータ表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.adaptive_consensus_cycle = None
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

    def generate_synthetic_data(
        self, 
        length: int = 500, 
        base_price: float = 100.0,
        noise_level: float = 0.3,
        trend_strength: float = 0.02
    ) -> pd.DataFrame:
        """
        合成テストデータの生成
        
        Args:
            length: データ長
            base_price: ベース価格
            noise_level: ノイズレベル
            trend_strength: トレンド強度
            
        Returns:
            合成価格データのDataFrame
        """
        np.random.seed(42)
        time = np.arange(length)
        
        # 複雑な周期変化（複数の周期が混在）
        freq1 = np.linspace(1/30., 1/20., length)  # 30→20周期
        freq2 = np.linspace(1/15., 1/8., length)   # 15→8周期
        freq3 = np.ones(length) * (1/45.)          # 固定45周期
        
        phase1 = np.cumsum(freq1 * 2 * np.pi)
        phase2 = np.cumsum(freq2 * 2 * np.pi)
        phase3 = np.cumsum(freq3 * 2 * np.pi)
        
        # 複数のサイクル成分（強度も時間変化）
        signal_part1 = 2.0 * np.sin(phase1) * (1 + 0.3 * np.sin(time / 50))
        signal_part2 = 0.8 * np.sin(phase2) * (1 + 0.2 * np.cos(time / 30))
        signal_part3 = 0.4 * np.sin(phase3)  # 長周期成分
        
        # 非線形トレンドとノイズ
        trend = trend_strength * (time + 0.1 * time**1.2)
        noise = np.random.randn(length) * noise_level * (1 + 0.5 * np.sin(time / 40))
        
        # 価格データ生成
        close = signal_part1 + signal_part2 + signal_part3 + trend + base_price + noise
        
        # OHLC生成（クローズ価格ベース）
        volatility = 0.5 + 0.3 * np.abs(np.sin(time / 25))
        high = close + np.abs(np.random.randn(length) * volatility)
        low = close - np.abs(np.random.randn(length) * volatility)
        open_price = close + np.random.randn(length) * 0.3
        
        # 日時インデックス生成
        date_range = pd.date_range(start='2023-01-01', periods=length, freq='H')
        
        # ボリューム生成（価格変動と相関）
        price_change = np.abs(np.diff(close, prepend=close[0]))
        volume = 3000 + 2000 * price_change / np.mean(price_change) + np.abs(np.random.randn(length) * 1000)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }, index=date_range)
        
        self.data = df
        return df

    def calculate_indicators(self,
                            adaptation_speed: float = 0.1,
                            noise_sensitivity: float = 0.5,
                            consensus_threshold: float = 0.6,
                            cycle_part: float = 1.0,
                            max_output: int = 120,
                            min_output: int = 13,
                            src_type: str = 'hlc3',
                            bandwidth: float = 0.6,
                            center_period: float = 15.0
                           ) -> None:
        """
        適応的コンセンサスサイクル検出器を計算する
        
        Args:
            adaptation_speed: 適応速度 (0.01-0.5)
            noise_sensitivity: ノイズ感度 (0.1-1.0)
            consensus_threshold: コンセンサス閾値 (0.3-0.9)
            cycle_part: サイクル部分倍率
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ
            bandwidth: バンドパスフィルタ帯域幅
            center_period: 中心周期
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()またはgenerate_synthetic_data()を先に実行してください。")
            
        print("\n適応的コンセンサスサイクル検出器を計算中...")
        
        # 適応的コンセンサスサイクル検出器を計算
        self.adaptive_consensus_cycle = AdaptiveConsensusCycle(
            adaptation_speed=adaptation_speed,
            noise_sensitivity=noise_sensitivity,
            consensus_threshold=consensus_threshold,
            cycle_part=cycle_part,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            bandwidth=bandwidth,
            center_period=center_period
        )
        
        # 計算実行
        print("計算を実行します...")
        cycle_values = self.adaptive_consensus_cycle.calculate(self.data)
        
        print(f"適応的コンセンサス計算完了 - サイクル値: {len(cycle_values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(cycle_values).sum()
        valid_count = len(cycle_values) - nan_count
        
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        
        if valid_count > 0:
            print(f"サイクル統計 - 平均: {np.nanmean(cycle_values):.2f}, 範囲: [{np.nanmin(cycle_values):.1f}, {np.nanmax(cycle_values):.1f}]")
        
        # 詳細結果の取得
        detailed_result = self.adaptive_consensus_cycle.get_detailed_result()
        method_results = self.adaptive_consensus_cycle.get_method_results()
        
        if detailed_result is not None:
            print(f"詳細結果取得成功:")
            print(f"  - Phase Accumulator: 平均 {np.nanmean(detailed_result.phase_accumulator):.2f}")
            print(f"  - Dual Differential: 平均 {np.nanmean(detailed_result.dual_differential):.2f}")
            print(f"  - Homodyne: 平均 {np.nanmean(detailed_result.homodyne):.2f}")
            print(f"  - Bandpass Zero: 平均 {np.nanmean(detailed_result.bandpass_zero):.2f}")
            print(f"  - 平均信頼性スコア: {np.nanmean(detailed_result.reliability_scores):.3f}")
            print(f"  - 平均ノイズレベル: {np.nanmean(detailed_result.noise_level):.3f}")
        
        print("適応的コンセンサスサイクル検出器計算完了")
            
    def plot(self, 
            title: str = "適応的コンセンサスサイクル検出器", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_methods_detail: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと適応的コンセンサスサイクル検出器を描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_methods_detail: 各手法の詳細を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
            
        if self.adaptive_consensus_cycle is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 適応的コンセンサスサイクル検出器の値を取得
        print("適応的コンセンサスデータを取得中...")
        cycle_values = self.adaptive_consensus_cycle.calculate(self.data)
        
        # 詳細結果の取得
        detailed_result = self.adaptive_consensus_cycle.get_detailed_result()
        method_results = self.adaptive_consensus_cycle.get_method_results()
        
        # 全データの時系列データフレームを作成
        full_df_data = {
            'consensus_cycle': cycle_values,
        }
        
        if detailed_result is not None:
            full_df_data.update({
                'phase_accumulator': detailed_result.phase_accumulator,
                'dual_differential': detailed_result.dual_differential,
                'homodyne': detailed_result.homodyne,
                'bandpass_zero': detailed_result.bandpass_zero,
                'consensus_weight': detailed_result.consensus_weight,
                'reliability_scores': detailed_result.reliability_scores,
                'noise_level': detailed_result.noise_level,
            })
        
        full_df = pd.DataFrame(index=self.data.index, data=full_df_data)
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"コンセンサスサイクルデータ確認 - NaN: {df['consensus_cycle'].isna().sum()}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # コンセンサスサイクル値をメインチャートの右軸に表示
        consensus_plot = mpf.make_addplot(df['consensus_cycle'], panel=0, color='purple', width=2.5, 
                                         secondary_y=True, ylabel='Consensus Cycle')
        main_plots.append(consensus_plot)
        
        # パネル設定を明確に管理
        current_panel = 1
        panel_configs = []
        
        # 2. 各検出手法の結果（詳細表示時）
        if show_methods_detail and detailed_result is not None:
            # Phase Accumulator vs Dual Differential
            pa_panel = mpf.make_addplot(df['phase_accumulator'], panel=current_panel, color='blue', width=1.5, 
                                       ylabel='PA & DD Period', secondary_y=False)
            dd_panel = mpf.make_addplot(df['dual_differential'], panel=current_panel, color='red', width=1.5, 
                                       secondary_y=False)
            main_plots.extend([pa_panel, dd_panel])
            panel_configs.append(('PA & DD', current_panel))
            current_panel += 1
            
            # Homodyne vs Bandpass Zero
            hm_panel = mpf.make_addplot(df['homodyne'], panel=current_panel, color='green', width=1.5, 
                                       ylabel='HM & BZ Period', secondary_y=False)
            bz_panel = mpf.make_addplot(df['bandpass_zero'], panel=current_panel, color='orange', width=1.5, 
                                       secondary_y=False)
            main_plots.extend([hm_panel, bz_panel])
            panel_configs.append(('HM & BZ', current_panel))
            current_panel += 1
        
        # 3. 信頼性スコアとコンセンサス重み（詳細結果がある場合のみ）
        if detailed_result is not None:
            reliability_panel = mpf.make_addplot(df['reliability_scores'], panel=current_panel, color='darkblue', width=1.8, 
                                                ylabel='Reliability', secondary_y=False)
            consensus_weight_panel = mpf.make_addplot(df['consensus_weight'], panel=current_panel, color='darkred', width=1.8, 
                                                     secondary_y=True)
            main_plots.extend([reliability_panel, consensus_weight_panel])
            panel_configs.append(('Reliability', current_panel))
            current_panel += 1
            
            # 4. ノイズレベル
            noise_panel = mpf.make_addplot(df['noise_level'], panel=current_panel, color='brown', width=1.5, 
                                          ylabel='Noise Level', secondary_y=False)
            main_plots.append(noise_panel)
            panel_configs.append(('Noise', current_panel))
            current_panel += 1
        
        # 出来高設定と全パネルの番号調整
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
        
        if show_volume:
            kwargs['volume'] = True
            # 出来高パネルが挿入されるため、全ての追加パネルの番号を+1
            for plot in main_plots:
                if hasattr(plot, 'panel') and plot.panel > 0:
                    plot.panel += 1
            
            # パネル設定も更新
            for i, (name, panel_num) in enumerate(panel_configs):
                panel_configs[i] = (name, panel_num + 1)
        else:
            kwargs['volume'] = False
        
        # パネル比率の設定
        # mplfinanceでは出来高パネルは自動で処理されるため、手動で追加したパネルのみをカウント
        total_manual_panels = len(panel_configs)  # 手動で追加したパネル数
        
        # パネル比率を動的に生成
        panel_ratios = [4]  # メインパネル
        
        # 手動で追加したパネルの比率
        for _ in panel_configs:
            panel_ratios.append(1)
        
        # 出来高がある場合、mplfinanceが自動でパネル比率を調整
        # 出来高パネルは自動処理されるため、panel_ratiosには含めない
        
        kwargs['panel_ratios'] = tuple(panel_ratios)
        kwargs['addplot'] = main_plots
        
        print(f"パネル設定: 手動パネル数={total_manual_panels}, 比率={panel_ratios}, 設定={panel_configs}")
        print(f"出来高表示: {show_volume}")
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 軸の調整と凡例追加
        base_panel = 1 if show_volume else 0
        
        if show_methods_detail and detailed_result is not None:
            # Phase Accumulator & Dual Differentialパネル
            pa_dd_ax = axes[base_panel + 1]
            pa_dd_ax.axhline(y=self.adaptive_consensus_cycle.min_cycle, color='black', linestyle='--', alpha=0.5)
            pa_dd_ax.axhline(y=self.adaptive_consensus_cycle.max_cycle, color='black', linestyle='--', alpha=0.5)
            pa_dd_ax.legend(['Phase Accumulator', 'Dual Differential', f'Min: {self.adaptive_consensus_cycle.min_cycle}', f'Max: {self.adaptive_consensus_cycle.max_cycle}'], 
                           loc='upper left')
            
            # Homodyne & Bandpass Zeroパネル
            hm_bz_ax = axes[base_panel + 2]
            hm_bz_ax.axhline(y=self.adaptive_consensus_cycle.min_cycle, color='black', linestyle='--', alpha=0.5)
            hm_bz_ax.axhline(y=self.adaptive_consensus_cycle.max_cycle, color='black', linestyle='--', alpha=0.5)
            hm_bz_ax.legend(['Homodyne', 'Bandpass Zero', f'Min: {self.adaptive_consensus_cycle.min_cycle}', f'Max: {self.adaptive_consensus_cycle.max_cycle}'], 
                           loc='upper left')
            
            # 信頼性スコア & コンセンサス重みパネル
            reliability_ax = axes[base_panel + 3]
            reliability_ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            reliability_ax.axhline(y=self.adaptive_consensus_cycle.consensus_threshold, color='red', linestyle='--', alpha=0.7)
            
            # ノイズレベルパネル
            noise_ax = axes[base_panel + 4]
            noise_mean = df['noise_level'].mean()
            noise_ax.axhline(y=noise_mean, color='gray', linestyle='-', alpha=0.5)
        else:
            # コンセンサスサイクルのみのパネル
            consensus_ax = axes[base_panel + 1] if len(axes) > base_panel + 1 else axes[-1]
            consensus_ax.axhline(y=self.adaptive_consensus_cycle.min_cycle, color='black', linestyle='--', alpha=0.5)
            consensus_ax.axhline(y=self.adaptive_consensus_cycle.max_cycle, color='black', linestyle='--', alpha=0.5)
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n=== 適応的コンセンサスサイクル検出器 統計 ===")
        valid_cycles = df['consensus_cycle'].dropna()
        
        if len(valid_cycles) > 0:
            print(f"総データ点数: {len(df)}")
            print(f"有効サイクル点数: {len(valid_cycles)}")
            print(f"平均サイクル: {valid_cycles.mean():.2f}")
            print(f"標準偏差: {valid_cycles.std():.2f}")
            print(f"範囲: {valid_cycles.min():.2f} - {valid_cycles.max():.2f}")
            
            # パラメータ情報
            print(f"\n=== パラメータ ===")
            print(f"適応速度: {self.adaptive_consensus_cycle.adaptation_speed}")
            print(f"ノイズ感度: {self.adaptive_consensus_cycle.noise_sensitivity}")
            print(f"コンセンサス閾値: {self.adaptive_consensus_cycle.consensus_threshold}")
            print(f"最小サイクル: {self.adaptive_consensus_cycle.min_cycle}")
            print(f"最大サイクル: {self.adaptive_consensus_cycle.max_cycle}")
            print(f"ソースタイプ: {self.adaptive_consensus_cycle.src_type}")
            
            # 各手法の統計
            if detailed_result is not None:
                print(f"\n=== 各手法の統計 ===")
                methods = ['phase_accumulator', 'dual_differential', 'homodyne', 'bandpass_zero']
                for method in methods:
                    if method in df.columns:
                        method_data = df[method].dropna()
                        if len(method_data) > 0:
                            print(f"{method}: 平均 {method_data.mean():.2f}, 標準偏差 {method_data.std():.2f}")
                
                print(f"\n=== 品質指標 ===")
                reliability_data = df['reliability_scores'].dropna()
                consensus_data = df['consensus_weight'].dropna()
                noise_data = df['noise_level'].dropna()
                
                if len(reliability_data) > 0:
                    print(f"平均信頼性スコア: {reliability_data.mean():.3f}")
                if len(consensus_data) > 0:
                    print(f"平均コンセンサス重み: {consensus_data.mean():.3f}")
                if len(noise_data) > 0:
                    print(f"平均ノイズレベル: {noise_data.mean():.3f}")
        
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
    parser = argparse.ArgumentParser(description='適応的コンセンサスサイクル検出器の描画')
    parser.add_argument('--config', '-c', type=str, help='設定ファイルのパス')
    parser.add_argument('--synthetic', '-syn', action='store_true', help='合成データを使用')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--adaptation-speed', type=float, default=0.1, help='適応速度')
    parser.add_argument('--noise-sensitivity', type=float, default=0.5, help='ノイズ感度')
    parser.add_argument('--consensus-threshold', type=float, default=0.6, help='コンセンサス閾値')
    parser.add_argument('--no-methods-detail', action='store_true', help='各手法の詳細を非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = AdaptiveConsensusCycleChart()
    
    if args.synthetic:
        print("合成データを生成中...")
        chart.generate_synthetic_data(length=500, noise_level=0.4)
    elif args.config:
        chart.load_data_from_config(args.config)
    else:
        print("--configまたは--syntheticオプションを指定してください")
        return
    
    chart.calculate_indicators(
        src_type=args.src_type,
        adaptation_speed=args.adaptation_speed,
        noise_sensitivity=args.noise_sensitivity,
        consensus_threshold=args.consensus_threshold
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_methods_detail=not args.no_methods_detail,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 