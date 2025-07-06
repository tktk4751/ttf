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
from indicators.ultimate_supreme_cycle_detector import UltimateSupremeCycleDetector


class UltimateSupremeCycleChart:
    """
    🚀 Ultimate Supreme Cycle Detector - 人類史上最強サイクル検出器表示チャートクラス
    
    表示内容:
    - ローソク足と出来高
    - 支配的サイクル期間（メインライン）
    - サイクル強度とコンフィデンス
    - 量子コヒーレンス
    - 市場レジーム・ボラティリティレジーム
    - カオス指標・位相空間トポロジー指標
    - 適応速度・追従精度
    - パフォーマンスメトリクス表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_detector = None
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
        
        print("\n🚀 Ultimate Supreme Cycle Detector - データを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📊 期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📈 データ数: {len(self.data)}")
        
        return self.data

    def calculate_indicators(self) -> None:
        """
        🚀 Ultimate Supreme Cycle Detector を計算する
        パラメータは最適化済みの値をハードコーディング
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🌀 Ultimate Supreme Cycle Detector を計算中...")
        
        # 人類史上最強のサイクル検出器パラメータ（中期サイクル最適化版）
        self.cycle_detector = UltimateSupremeCycleDetector(
            # 基本設定（中期サイクル20-100期間に最適化）
            period_range=(20, 100),                # 20-100期間に制限
            adaptivity_factor=0.75,                # 適応性を下げて中期に集中
            tracking_sensitivity=0.85,             # 追従感度を調整
            
            # 量子パラメータ（中期サイクル向け調整）
            quantum_coherence_threshold=0.70,      # 閾値を下げて中期サイクルを重視
            entanglement_strength=0.75,            # もつれ強度を調整
            
            # 情報理論パラメータ（中期最適化）
            entropy_window=20,                     # ウィンドウを短縮
            information_gain_threshold=0.60,       # 閾値を下げる
            
            # 統合融合パラメータ（中期サイクル向け）
            chaos_embedding_dimension=3,           # 次元を下げる
            topology_analysis_window=25,           # ウィンドウを短縮
            attractor_reconstruction_delay=2,      # 遅延を短縮
            
            # 適応制御パラメータ（中期最適化）
            fast_adapt_alpha=0.4,                  # 高速適応を強化
            slow_adapt_alpha=0.08,                 # 低速適応を調整
            regime_switch_threshold=0.6,           # 閾値を下げる
            
            # 追従性制御（中期サイクル向け）
            tracking_lag_tolerance=1,              # 遅延許容値を短縮
            noise_immunity_factor=0.80,            # ノイズ耐性を調整
            signal_purity_threshold=0.85,          # 純度閾値を調整
            
            # 価格ソース
            src_type='hlc3'
        )
        
        print("🔬 量子情報統合計算を実行します...")
        result = self.cycle_detector.calculate(self.data)
        
        print(f"✅ Ultimate Supreme Cycle Detector 計算完了")
        print(f"🎯 現在の支配的サイクル: {result.current_cycle:.1f}期間")
        print(f"💪 現在のサイクル強度: {result.current_strength:.3f}")
        print(f"🎉 現在の信頼度: {result.current_confidence:.3f}")
        
        # パフォーマンスメトリクス
        metrics = self.cycle_detector.get_performance_metrics()
        print(f"\n📊 パフォーマンスメトリクス:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("🎉 Ultimate Supreme Cycle Detector 計算完了")
            
    def plot(self, 
            title: str = "🚀 Ultimate Supreme Cycle Detector", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            figsize: Tuple[int, int] = (16, 12),
            savefig: Optional[str] = None) -> None:
        """チャート描画"""
        if self.data is None or self.cycle_detector is None:
            raise ValueError("データまたはインジケーターが計算されていません。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        result = self.cycle_detector.get_result()
        if result is None:
            print("❌ 計算結果が見つかりません")
            return
        
        # データフレーム作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'dominant_cycle': result.dominant_cycle,
                'cycle_strength': result.cycle_strength,
                'quantum_coherence': result.quantum_coherence,
            }
        )
        
        df = df.join(full_df)
        
        # プロット設定
        plots = []
        
        # サイクル期間パネル
        if (~df['dominant_cycle'].isna()).sum() > 0:
            plots.append(mpf.make_addplot(df['dominant_cycle'], panel=2, color='purple', width=2, ylabel='Cycle Period'))
        
        # 量子コヒーレンスパネル
        if (~df['quantum_coherence'].isna()).sum() > 0:
            plots.append(mpf.make_addplot(df['quantum_coherence'], panel=3, color='cyan', width=1.5, ylabel='Quantum Coherence'))
        
        if len(plots) == 0:
            print("⚠️ 表示可能なデータがありません")
            return
        
        try:
            fig, axes = mpf.plot(
                df,
                type='candle',
                figsize=figsize,
                title=f"{title}\n現在サイクル: {result.current_cycle:.1f}期間",
                volume=True,
                addplot=plots,
                panel_ratios=(4, 1, 2, 1.5),
                returnfig=True
            )
            
            self.fig = fig
            self.axes = axes
            
            if savefig:
                plt.savefig(savefig, dpi=150, bbox_inches='tight')
                print(f"💾 チャートを保存しました: {savefig}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"⚠️ プロットエラー: {e}")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='🚀 Ultimate Supreme Cycle Detector の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    
    args = parser.parse_args()
    
    chart = UltimateSupremeCycleChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 