#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
エラーズ ヒルベルト判別機チャート可視化機能

ジョンエラーズ氏のヒルベルト変換理論に基づく市場状態判別機の包括的な可視化：
- ローソク足チャートと市場状態の色分け背景
- トレンド・サイクル強度の表示
- ヒルベルト変換成分（I/Q）の可視化
- 位相レート・周波数・DC/AC分析
- 実際の相場データでのリアルタイムテスト
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非対話的バックエンドを使用
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# エラーズ ヒルベルト判別機
from indicators.ehlers_hilbert_discriminator import EhlersHilbertDiscriminator


class EhlersHilbertDiscriminatorChart:
    """
    エラーズ ヒルベルト判別機を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 市場状態の背景色分け（トレンド/サイクル）
    - トレンド・サイクル強度
    - ヒルベルト変換成分（I/Q）
    - 位相レート・周波数分析
    - DC/AC成分分析
    - 判別信頼度
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hilbert_discriminator = None
        self.fig = None
        self.axes = None
        self.result = None
    
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
        print("\n🔄 データを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📅 期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📊 データ数: {len(self.data)}")
        
        return self.data

    def calculate_indicators(self,
                            src_type: str = 'close',                    # 価格ソース
                            filter_length: int = 7,                     # ヒルベルトフィルター長
                            smoothing_factor: float = 0.2,              # 平滑化係数
                            analysis_window: int = 14,                  # 分析ウィンドウ
                            phase_rate_threshold: float = 0.05,         # 位相レート閾値（調整済み）
                            dc_ac_ratio_threshold: float = 1.2          # DC/AC比率閾値（調整済み）
                           ) -> None:
        """
        エラーズ ヒルベルト判別機を計算する
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            filter_length: ヒルベルトフィルター長（推奨: 7）
            smoothing_factor: 位相レート平滑化係数（0-1）
            analysis_window: 市場状態分析ウィンドウ
            phase_rate_threshold: 位相レート安定性閾値
            dc_ac_ratio_threshold: DC/AC比率判別閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n⚡ エラーズ ヒルベルト判別機を計算中...")
        
        # エラーズ ヒルベルト判別機を初期化
        self.hilbert_discriminator = EhlersHilbertDiscriminator(
            src_type=src_type,
            filter_length=filter_length,
            smoothing_factor=smoothing_factor,
            analysis_window=analysis_window,
            phase_rate_threshold=phase_rate_threshold,
            dc_ac_ratio_threshold=dc_ac_ratio_threshold
        )
        
        # 計算実行
        print("🔬 市場状態分析を実行中...")
        self.result = self.hilbert_discriminator.calculate(self.data)
        
        # 結果の確認
        print(f"✅ 計算完了 - データポイント数: {len(self.result.trend_mode)}")
        
        # 統計情報
        trend_mode_pct = np.mean(self.result.trend_mode) * 100
        cycle_mode_pct = 100 - trend_mode_pct
        avg_trend_strength = np.nanmean(self.result.trend_strength)
        avg_cycle_strength = np.nanmean(self.result.cycle_strength)
        avg_confidence = np.nanmean(self.result.confidence)
        
        print(f"📊 市場状態統計:")
        print(f"   - トレンドモード: {trend_mode_pct:.1f}%")
        print(f"   - サイクルモード: {cycle_mode_pct:.1f}%")
        print(f"   - 平均トレンド強度: {avg_trend_strength:.3f}")
        print(f"   - 平均サイクル強度: {avg_cycle_strength:.3f}")
        print(f"   - 平均信頼度: {avg_confidence:.3f}")
        
        # NaN値のチェック
        nan_counts = {
            'trend_mode': np.isnan(self.result.trend_mode.astype(float)).sum(),
            'in_phase': np.isnan(self.result.in_phase).sum(),
            'quadrature': np.isnan(self.result.quadrature).sum(),
            'phase_rate': np.isnan(self.result.phase_rate).sum(),
            'confidence': np.isnan(self.result.confidence).sum()
        }
        
        print(f"🔍 NaN値チェック: {nan_counts}")
        print("🎯 エラーズ ヒルベルト判別機計算完了")
            
    def plot(self, 
            title: str = "エラーズ ヒルベルト判別機 - 市場状態分析", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとエラーズ ヒルベルト判別機を描画する
        
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
            
        if self.hilbert_discriminator is None or self.result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        print(f"📈 チャート描画準備中... ({len(df)}データポイント)")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_mode': self.result.trend_mode.astype(float),
                'market_state': self.result.market_state.astype(float),
                'in_phase': self.result.in_phase,
                'quadrature': self.result.quadrature,
                'instantaneous_phase': self.result.instantaneous_phase,
                'phase_rate': self.result.phase_rate,
                'dc_component': self.result.dc_component,
                'ac_component': self.result.ac_component,
                'trend_strength': self.result.trend_strength,
                'cycle_strength': self.result.cycle_strength,
                'amplitude': self.result.amplitude,
                'frequency': self.result.frequency,
                'confidence': self.result.confidence
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"📊 チャートデータ準備完了 - 行数: {len(df)}")
        print(f"🔍 データ確認 - トレンドモードNaN: {df['trend_mode'].isna().sum()}, 信頼度NaN: {df['confidence'].isna().sum()}")
        
        # 市場状態別の背景色用データ準備
        df['trend_mode_trend'] = np.where(df['trend_mode'] == 1, 1, np.nan)
        df['trend_mode_cycle'] = np.where(df['trend_mode'] == 0, 0, np.nan)
        
        # 市場状態レベル別のデータ準備
        df['market_range'] = np.where(df['market_state'] == 0, 0, np.nan)
        df['market_weak_trend'] = np.where(df['market_state'] == 1, 1, np.nan)
        df['market_strong_trend'] = np.where(df['market_state'] == 2, 2, np.nan)
        
        # DC/AC比率の計算
        df['dc_ac_ratio'] = np.where(
            df['ac_component'] > 1e-10, 
            np.abs(df['dc_component']) / df['ac_component'], 
            1.0
        )
        
        # 位相を度数に変換
        df['phase_degrees'] = df['instantaneous_phase'] * 180 / np.pi
        
        # 正規化周波数をパーセント表示用に変換
        df['frequency_percent'] = df['frequency'] * 100
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # パネル番号の基準設定
        panel_offset = 2 if show_volume else 1
        
        # すべてのプロットを配列に追加
        all_plots = [
            # パネル1: 強度
            mpf.make_addplot(df['trend_strength'], panel=panel_offset, color='red', width=1.5, 
                           ylabel='Strength', label='Trend Strength'),
            mpf.make_addplot(df['cycle_strength'], panel=panel_offset, color='blue', width=1.5, 
                           label='Cycle Strength'),
            
            # パネル2: ヒルベルト成分
            mpf.make_addplot(df['in_phase'], panel=panel_offset+1, color='green', width=1.2, 
                           ylabel='Hilbert I/Q', label='In-Phase'),
            mpf.make_addplot(df['quadrature'], panel=panel_offset+1, color='orange', width=1.2, 
                           label='Quadrature'),
            
            # パネル3: 位相分析
            mpf.make_addplot(df['phase_rate'], panel=panel_offset+2, color='purple', width=1.2,
                           ylabel='Phase Rate', label='Phase Rate'),
            
            # パネル4: DC/AC + トレンドモード
            mpf.make_addplot(df['dc_component'], panel=panel_offset+3, color='red', width=1.2,
                           ylabel='DC/AC', label='DC Component'),
            mpf.make_addplot(df['ac_component'], panel=panel_offset+3, color='blue', width=1.2,
                           label='AC Component'),
            
            # パネル5: 信頼度
            mpf.make_addplot(df['confidence'], panel=panel_offset+4, color='darkgreen', width=1.5,
                           ylabel='Confidence', label='Confidence')
        ]
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            tight_layout=True
        )
        
        # 出来高とパネル比率の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1.2, 1.2, 1.2, 1.2, 1.2)  # メイン:出来高:強度:I/Q:位相:DC/AC:信頼度
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1.2, 1.2, 1.2, 1.2, 1.2)  # メイン:強度:I/Q:位相:DC/AC:信頼度
        
        kwargs['addplot'] = all_plots
        
        # プロット実行
        print("🎨 チャートを描画中...")
        fig, axes = mpf.plot(df, **kwargs)
        
        # 背景色の追加（市場状態別）
        print("🎨 市場状態の背景色を追加中...")
        main_ax = axes[0]
        
        # トレンドモードとサイクルモードの背景色
        for i in range(len(df)):
            try:
                if not pd.isna(df.iloc[i]['trend_mode']):
                    x_pos = i
                    if df.iloc[i]['trend_mode'] == 1:  # トレンドモード
                        main_ax.axvspan(x_pos, x_pos+1, alpha=0.1, color='red', zorder=0)
                    else:  # サイクルモード  
                        main_ax.axvspan(x_pos, x_pos+1, alpha=0.1, color='blue', zorder=0)
            except:
                continue
        
        # 各パネルに参照線を追加
        panel_start = 2 if show_volume else 1
        
        # 強度パネル（パネル1）
        axes[panel_start].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_start].set_ylim(0, 1)
        
        # ヒルベルト成分パネル（パネル2）
        axes[panel_start+1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 位相レートパネル（パネル3）
        axes[panel_start+2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_start+2].axhline(y=self.hilbert_discriminator.phase_rate_threshold, 
                                    color='purple', linestyle='--', alpha=0.7)
        axes[panel_start+2].axhline(y=-self.hilbert_discriminator.phase_rate_threshold, 
                                    color='purple', linestyle='--', alpha=0.7)
        
        # DC/ACパネル（パネル4）
        axes[panel_start+3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 信頼度パネル（パネル5）
        axes[panel_start+4].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_start+4].set_ylim(0, 1)
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n📊 === 市場分析結果 ===")
        total_points = len(df[~df['trend_mode'].isna()])
        trend_points = len(df[df['trend_mode'] == 1])
        cycle_points = len(df[df['trend_mode'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"トレンドモード: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"サイクルモード: {cycle_points} ({cycle_points/total_points*100:.1f}%)")
        print(f"平均信頼度: {df['confidence'].mean():.3f}")
        print(f"平均トレンド強度: {df['trend_strength'].mean():.3f}")
        print(f"平均サイクル強度: {df['cycle_strength'].mean():.3f}")
        
        # 最新の市場状態
        current_state = self.hilbert_discriminator.get_current_market_state_description()
        print(f"🎯 現在の市場状態: {current_state}")
        
        # 保存または表示
        if savefig:
            output_dir = os.path.dirname(savefig)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"💾 チャートを保存しました: {savefig}")
        
        # メモリクリア
        plt.close(fig)
        print("✅ チャート描画完了")

    def export_analysis_report(self, output_path: str) -> None:
        """分析レポートをファイル出力"""
        if self.result is None:
            print("❌ 分析結果がありません")
            return
        
        print(f"📝 分析レポートを出力中: {output_path}")
        
        # メタデータ取得
        metadata = self.hilbert_discriminator.get_discriminator_metadata()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== エラーズ ヒルベルト判別機 分析レポート ===\n\n")
            
            # 基本情報
            f.write("【基本情報】\n")
            for key, value in metadata.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 統計情報
            trend_mode_pct = np.mean(self.result.trend_mode) * 100
            f.write("【市場状態統計】\n")
            f.write(f"  トレンドモード: {trend_mode_pct:.1f}%\n")
            f.write(f"  サイクルモード: {100-trend_mode_pct:.1f}%\n")
            f.write(f"  平均トレンド強度: {np.nanmean(self.result.trend_strength):.3f}\n")
            f.write(f"  平均サイクル強度: {np.nanmean(self.result.cycle_strength):.3f}\n")
            f.write(f"  平均信頼度: {np.nanmean(self.result.confidence):.3f}\n")
            f.write("\n")
            
            # 現在の状態
            f.write("【現在の市場状態】\n")
            current_state = self.hilbert_discriminator.get_current_market_state_description()
            f.write(f"  {current_state}\n")
            
        print(f"✅ レポート出力完了: {output_path}")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='エラーズ ヒルベルト判別機の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--report', '-r', type=str, help='分析レポート出力パス')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソースタイプ')
    parser.add_argument('--filter-length', type=int, default=7, help='ヒルベルトフィルター長')
    parser.add_argument('--smoothing-factor', type=float, default=0.2, help='平滑化係数')
    parser.add_argument('--analysis-window', type=int, default=14, help='分析ウィンドウ')
    parser.add_argument('--phase-threshold', type=float, default=0.05, help='位相レート閾値')
    parser.add_argument('--dc-ac-threshold', type=float, default=1.2, help='DC/AC比率閾値')
    args = parser.parse_args()
    
    try:
        print("🚀 エラーズ ヒルベルト判別機 - 実市場データ分析")
        print("=" * 60)
        
        # チャートを作成
        chart = EhlersHilbertDiscriminatorChart()
        chart.load_data_from_config(args.config)
        chart.calculate_indicators(
            src_type=args.src_type,
            filter_length=args.filter_length,
            smoothing_factor=args.smoothing_factor,
            analysis_window=args.analysis_window,
            phase_rate_threshold=args.phase_threshold,
            dc_ac_ratio_threshold=args.dc_ac_threshold
        )
        
        # チャート描画
        output_path = args.output
        if not output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"visualization/output/ehlers_hilbert_discriminator_real_{timestamp}.png"
        
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            savefig=output_path
        )
        
        # レポート出力
        if args.report:
            chart.export_analysis_report(args.report)
        
        print("\n🎉 分析完了！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 