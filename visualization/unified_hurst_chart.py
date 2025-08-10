#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合ハースト指数チャート表示システム

複数のハースト指数計算手法を実際の相場データで比較・可視化します。

実装手法:
1. R/S法 (Rescaled Range Statistics)
2. DFA法 (Detrended Fluctuation Analysis) 
3. ウェーブレット法 (Daubechies Wavelet Method)
4. 統合ハースト指数 (Consensus Hurst)
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import seaborn as sns

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# ハースト指数インジケーター
from indicators.unified_hurst_exponent import UnifiedHurstExponent


class UnifiedHurstChart:
    """
    統合ハースト指数を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 全ハースト指数手法の比較表示
    - 統合ハースト指数とコンセンサス
    - 持続性レジーム分析
    - 信頼度スコア表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hurst_indicators = {}
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
                           hurst_configs: List[Dict[str, Any]] = None
                           ) -> None:
        """
        統合ハースト指数インジケーターを計算する
        
        Args:
            hurst_configs: ハースト指数の設定リスト
                例: [
                    {'window_size': 50, 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_50'},
                    {'window_size': 100, 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_100'}
                ]
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
        
        # デフォルトの設定
        if hurst_configs is None:
            hurst_configs = [
                {'window_size': 50, 'src_type': 'close', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_50_Close'},
                {'window_size': 80, 'src_type': 'hlc3', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_80_HLC3'},
                {'window_size': 120, 'src_type': 'hl2', 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': 'Hurst_120_HL2'}
            ]
        
        print("\n統合ハースト指数インジケーターを計算中...")
        
        for config in hurst_configs:
            print(f"  計算中: {config['name']}")
            
            # ハースト指数インジケーターを作成
            hurst = UnifiedHurstExponent(
                window_size=config['window_size'],
                src_type=config.get('src_type', 'close'),
                enable_rs=config.get('enable_rs', True),
                enable_dfa=config.get('enable_dfa', True),
                enable_wavelet=config.get('enable_wavelet', True)
            )
            
            # 計算実行
            result = hurst.calculate(self.data)
            
            # 結果を保存
            self.hurst_indicators[config['name']] = {
                'indicator': hurst,
                'result': result,
                'config': config
            }
            
            # 統計情報を表示
            valid_consensus = result.hurst_consensus[~np.isnan(result.hurst_consensus)]
            valid_confidence = result.confidence_score[~np.isnan(result.confidence_score)]
            
            if len(valid_consensus) > 0:
                print(f"    有効値数: {len(valid_consensus)}")
                print(f"    コンセンサス範囲: {np.min(valid_consensus):.4f} - {np.max(valid_consensus):.4f}")
                print(f"    平均信頼度: {np.mean(valid_confidence):.4f}")
                
                # 最新値解釈
                latest_hurst = valid_consensus[-1] if len(valid_consensus) > 0 else None
                if latest_hurst is not None:
                    if latest_hurst < 0.45:
                        interpretation = "強い反持続性（平均回帰傾向）"
                    elif latest_hurst < 0.48:
                        interpretation = "弱い反持続性"
                    elif latest_hurst < 0.52:
                        interpretation = "ランダムウォーク"
                    elif latest_hurst < 0.55:
                        interpretation = "弱い持続性"
                    else:
                        interpretation = "強い持続性（トレンド継続傾向）"
                    print(f"    最新解釈: {interpretation} (H={latest_hurst:.4f})")
        
        print("統合ハースト指数計算完了")
            
    def plot(self, 
            title: str = "統合ハースト指数分析", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 18),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            show_individual_methods: bool = True,
            show_plot: bool = True) -> None:
        """
        ローソク足チャートと統合ハースト指数を描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            show_individual_methods: 個別手法を表示するか
            show_plot: matplotlibで表示するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if not self.hurst_indicators:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # ハースト指数データを全データフレームに追加
        print("ハースト指数データを準備中...")
        for name, indicator_data in self.hurst_indicators.items():
            result = indicator_data['result']
            config = indicator_data['config']
            
            # 全データの時系列データフレームを作成
            full_df = pd.DataFrame(
                index=self.data.index,
                data={
                    f'{name}_consensus': result.hurst_consensus,
                    f'{name}_rs': result.hurst_rs,
                    f'{name}_dfa': result.hurst_dfa,
                    f'{name}_wavelet': result.hurst_wavelet,
                    f'{name}_confidence': result.confidence_score,
                    f'{name}_persistence': result.persistence_regime
                }
            )
            
            # 絞り込み後のデータに対してインジケーターデータを結合
            df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # コンセンサスハースト指数のプロット設定
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        line_styles = ['-', '--', '-.', ':', '-', '--']
        
        # 各設定のコンセンサスハースト指数をメインに表示しないで、別パネルに表示
        additional_plots = []
        panel_count = 0
        
        # パネル1: コンセンサスハースト指数
        panel_count += 1
        for i, (name, indicator_data) in enumerate(self.hurst_indicators.items()):
            config = indicator_data['config']
            color = colors[i % len(colors)]
            style_line = line_styles[i % len(line_styles)]
            
            # データの有効性チェック
            consensus_data = df[f'{name}_consensus'].dropna()
            if len(consensus_data) > 0:
                additional_plots.append(
                    mpf.make_addplot(
                        df[f'{name}_consensus'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color=color, 
                        width=2,
                        linestyle=style_line,
                        ylabel='Hurst Exponent',
                        label=f"{name} Consensus (w={config['window_size']})"
                    )
                )
        
        # パネル2: 個別手法比較（最初の設定のみ）
        if show_individual_methods and self.hurst_indicators:
            panel_count += 1
            first_name = list(self.hurst_indicators.keys())[0]
            
            # 各手法のデータ有効性チェック
            rs_data = df[f'{first_name}_rs'].dropna()
            dfa_data = df[f'{first_name}_dfa'].dropna()
            wavelet_data = df[f'{first_name}_wavelet'].dropna()
            
            individual_method_plots = []
            if len(rs_data) > 0:
                individual_method_plots.append(
                    mpf.make_addplot(
                        df[f'{first_name}_rs'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color='darkblue', 
                        width=1.5,
                        linestyle='-',
                        ylabel='Individual Methods',
                        label='R/S Method'
                    )
                )
            if len(dfa_data) > 0:
                individual_method_plots.append(
                    mpf.make_addplot(
                        df[f'{first_name}_dfa'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color='darkgreen', 
                        width=1.5,
                        linestyle='--',
                        label='DFA Method'
                    )
                )
            if len(wavelet_data) > 0:
                individual_method_plots.append(
                    mpf.make_addplot(
                        df[f'{first_name}_wavelet'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color='darkred', 
                        width=1.5,
                        linestyle='-.',
                        label='Wavelet Method'
                    )
                )
            
            additional_plots.extend(individual_method_plots)
        
        # パネル3: 信頼度スコア
        panel_count += 1
        for i, (name, indicator_data) in enumerate(self.hurst_indicators.items()):
            color = colors[i % len(colors)]
            confidence_data = df[f'{name}_confidence'].dropna()
            if len(confidence_data) > 0:
                additional_plots.append(
                    mpf.make_addplot(
                        df[f'{name}_confidence'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color=color, 
                        width=1.5,
                        alpha=0.7,
                        ylabel='Confidence Score',
                        label=f"{name} Confidence"
                    )
                )
        
        # パネル4: 持続性レジーム
        panel_count += 1
        for i, (name, indicator_data) in enumerate(self.hurst_indicators.items()):
            color = colors[i % len(colors)]
            persistence_data = df[f'{name}_persistence'].dropna()
            if len(persistence_data) > 0:
                additional_plots.append(
                    mpf.make_addplot(
                        df[f'{name}_persistence'], 
                        panel=panel_count if not show_volume else panel_count + 1,
                        color=color, 
                        width=2,
                        ylabel='Persistence Regime',
                        label=f"{name} Persistence"
                    )
                )
        
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
        
        # パネル比率の計算（プロットがあるパネルのみ）
        panel_ratios = [4]  # メインチャート
        
        if show_volume:
            panel_ratios.append(1)  # 出来高
        
        # 有効なデータがあるパネルのみ追加
        has_consensus = any(len(df[f'{name}_consensus'].dropna()) > 0 for name in self.hurst_indicators.keys())
        has_individual = False
        has_confidence = any(len(df[f'{name}_confidence'].dropna()) > 0 for name in self.hurst_indicators.keys())
        has_persistence = any(len(df[f'{name}_persistence'].dropna()) > 0 for name in self.hurst_indicators.keys())
        
        if show_individual_methods and self.hurst_indicators:
            first_name = list(self.hurst_indicators.keys())[0]
            has_individual = (len(df[f'{first_name}_rs'].dropna()) > 0 or 
                            len(df[f'{first_name}_dfa'].dropna()) > 0 or
                            len(df[f'{first_name}_wavelet'].dropna()) > 0)
        
        if has_consensus:
            panel_ratios.append(2)  # コンセンサス
        if has_individual:
            panel_ratios.append(2)  # 個別手法
        if has_confidence:
            panel_ratios.append(1)  # 信頼度
        if has_persistence:
            panel_ratios.append(1.5)  # 持続性レジーム
        
        # 出来高と追加パネルの設定
        kwargs['volume'] = show_volume
        kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # すべてのプロットを結合
        if additional_plots:
            kwargs['addplot'] = additional_plots
        else:
            # プロットが無い場合はダミープロットを作成
            dummy_data = pd.Series([np.nan] * len(df), index=df.index)
            kwargs['addplot'] = [mpf.make_addplot(dummy_data, panel=1, alpha=0)]
            panel_ratios = [4, 1] if show_volume else [4]
            kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 参照線の追加
        panel_idx = 1 if show_volume else 0
        
        # コンセンサスハースト指数の参照線
        panel_idx += 1
        axes[panel_idx].axhline(y=0.5, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Random Walk (H=0.5)')
        axes[panel_idx].axhline(y=0.45, color='red', linestyle='--', alpha=0.6, label='Strong Anti-persistence')
        axes[panel_idx].axhline(y=0.55, color='green', linestyle='--', alpha=0.6, label='Strong Persistence')
        axes[panel_idx].set_ylim(0.2, 0.8)
        axes[panel_idx].legend(loc='upper right', fontsize=8)
        
        # 個別手法の参照線
        if show_individual_methods:
            panel_idx += 1
            axes[panel_idx].axhline(y=0.5, color='black', linestyle='-', alpha=0.8, linewidth=2)
            axes[panel_idx].axhline(y=0.45, color='red', linestyle='--', alpha=0.6)
            axes[panel_idx].axhline(y=0.55, color='green', linestyle='--', alpha=0.6)
            axes[panel_idx].set_ylim(0.2, 0.8)
            axes[panel_idx].legend(loc='upper right', fontsize=8)
        
        # 信頼度スコアの参照線
        panel_idx += 1
        axes[panel_idx].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Confidence')
        axes[panel_idx].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
        axes[panel_idx].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Low Confidence')
        axes[panel_idx].set_ylim(0, 1)
        
        # 持続性レジームの参照線
        panel_idx += 1
        axes[panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Neutral')
        axes[panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Strong Persistence')
        axes[panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Strong Anti-persistence')
        axes[panel_idx].axhline(y=0.5, color='lightgreen', linestyle=':', alpha=0.5, label='Weak Persistence')
        axes[panel_idx].axhline(y=-0.5, color='lightcoral', linestyle=':', alpha=0.5, label='Weak Anti-persistence')
        axes[panel_idx].set_ylim(-1.2, 1.2)
        
        # 統計情報の表示
        print(f"\n=== ハースト指数統計情報 ===")
        for name, indicator_data in self.hurst_indicators.items():
            result = indicator_data['result']
            config = indicator_data['config']
            
            # 表示期間に対応する統計を計算
            consensus_slice = df[f'{name}_consensus'].dropna()
            confidence_slice = df[f'{name}_confidence'].dropna()
            persistence_slice = df[f'{name}_persistence'].dropna()
            
            if len(consensus_slice) > 0:
                print(f"\n{name}:")
                print(f"  設定: window={config['window_size']}, src_type={config.get('src_type', 'close')}")
                print(f"  コンセンサスH: {consensus_slice.mean():.4f} ± {consensus_slice.std():.4f}")
                print(f"  信頼度: {confidence_slice.mean():.4f} (範囲: {confidence_slice.min():.4f} - {confidence_slice.max():.4f})")
                
                # 持続性分析
                strong_persistence_pct = (persistence_slice > 0.5).sum() / len(persistence_slice) * 100
                strong_anti_persistence_pct = (persistence_slice < -0.5).sum() / len(persistence_slice) * 100
                random_walk_pct = (np.abs(persistence_slice) <= 0.5).sum() / len(persistence_slice) * 100
                
                print(f"  強い持続性: {strong_persistence_pct:.1f}%")
                print(f"  強い反持続性: {strong_anti_persistence_pct:.1f}%") 
                print(f"  ランダムウォーク類似: {random_walk_pct:.1f}%")
        
        # 手法間比較分析
        self._analyze_method_comparison(df)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        
        # matplotlib表示
        if show_plot:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
    
    def _analyze_method_comparison(self, df: pd.DataFrame) -> None:
        """手法間の比較分析を実行"""
        print(f"\n=== 手法間比較分析 ===")
        
        if not self.hurst_indicators:
            return
        
        # 最初の設定で手法間比較
        first_name = list(self.hurst_indicators.keys())[0]
        
        rs_values = df[f'{first_name}_rs'].dropna()
        dfa_values = df[f'{first_name}_dfa'].dropna()
        wavelet_values = df[f'{first_name}_wavelet'].dropna()
        consensus_values = df[f'{first_name}_consensus'].dropna()
        
        if len(rs_values) > 0 and len(dfa_values) > 0 and len(wavelet_values) > 0:
            print(f"\n{first_name}の手法別統計:")
            print(f"  R/S法: 平均={rs_values.mean():.4f}, 標準偏差={rs_values.std():.4f}")
            print(f"  DFA法: 平均={dfa_values.mean():.4f}, 標準偏差={dfa_values.std():.4f}")
            print(f"  ウェーブレット法: 平均={wavelet_values.mean():.4f}, 標準偏差={wavelet_values.std():.4f}")
            print(f"  コンセンサス: 平均={consensus_values.mean():.4f}, 標準偏差={consensus_values.std():.4f}")
            
            # 相関分析
            try:
                # 共通期間でのデータ抽出
                common_idx = df.index[
                    df[f'{first_name}_rs'].notna() & 
                    df[f'{first_name}_dfa'].notna() & 
                    df[f'{first_name}_wavelet'].notna()
                ]
                
                if len(common_idx) > 10:
                    rs_common = df.loc[common_idx, f'{first_name}_rs']
                    dfa_common = df.loc[common_idx, f'{first_name}_dfa']
                    wavelet_common = df.loc[common_idx, f'{first_name}_wavelet']
                    
                    corr_rs_dfa = np.corrcoef(rs_common, dfa_common)[0, 1]
                    corr_rs_wavelet = np.corrcoef(rs_common, wavelet_common)[0, 1]
                    corr_dfa_wavelet = np.corrcoef(dfa_common, wavelet_common)[0, 1]
                    
                    print(f"\n  手法間相関:")
                    print(f"    R/S vs DFA: {corr_rs_dfa:.4f}")
                    print(f"    R/S vs ウェーブレット: {corr_rs_wavelet:.4f}")
                    print(f"    DFA vs ウェーブレット: {corr_dfa_wavelet:.4f}")
                    
                    # 最も優れた手法の判定
                    print(f"\n  === 手法評価 ===")
                    
                    # 安定性（標準偏差の逆数）
                    stability_rs = 1 / (rs_values.std() + 0.001)
                    stability_dfa = 1 / (dfa_values.std() + 0.001)
                    stability_wavelet = 1 / (wavelet_values.std() + 0.001)
                    
                    print(f"    安定性スコア:")
                    print(f"      R/S法: {stability_rs:.4f}")
                    print(f"      DFA法: {stability_dfa:.4f}")
                    print(f"      ウェーブレット法: {stability_wavelet:.4f}")
                    
                    # 0.5からの偏差（ランダムウォークからの距離）
                    deviation_rs = np.mean(np.abs(rs_values - 0.5))
                    deviation_dfa = np.mean(np.abs(dfa_values - 0.5))
                    deviation_wavelet = np.mean(np.abs(wavelet_values - 0.5))
                    
                    print(f"    識別能力（0.5からの偏差）:")
                    print(f"      R/S法: {deviation_rs:.4f}")
                    print(f"      DFA法: {deviation_dfa:.4f}")
                    print(f"      ウェーブレット法: {deviation_wavelet:.4f}")
                    
                    # 総合評価
                    score_rs = stability_rs * 0.4 + deviation_rs * 0.6
                    score_dfa = stability_dfa * 0.4 + deviation_dfa * 0.6
                    score_wavelet = stability_wavelet * 0.4 + deviation_wavelet * 0.6
                    
                    scores = [
                        ("R/S法", score_rs),
                        ("DFA法", score_dfa),
                        ("ウェーブレット法", score_wavelet)
                    ]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    
                    print(f"\n    総合評価ランキング:")
                    for i, (method, score) in enumerate(scores, 1):
                        print(f"      {i}位: {method} (スコア: {score:.4f})")
                    
                    # 推奨事項
                    best_method = scores[0][0]
                    print(f"\n    推奨: {best_method}が最も優れた性能を示しています")
                    
                    if best_method == "DFA法":
                        print("      理由: DFA法はトレンド除去により安定した結果を提供し、金融時系列に適しています")
                    elif best_method == "R/S法":
                        print("      理由: R/S法は古典的で解釈しやすく、長期記憶特性の検出に優れています")
                    elif best_method == "ウェーブレット法":
                        print("      理由: ウェーブレット法は周波数領域での分析により、複雑な構造を捉えます")
                        
            except Exception as e:
                print(f"  相関分析エラー: {e}")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='統合ハースト指数の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--window1', type=int, default=50, help='1つ目のウィンドウサイズ')
    parser.add_argument('--window2', type=int, default=80, help='2つ目のウィンドウサイズ')
    parser.add_argument('--window3', type=int, default=120, help='3つ目のウィンドウサイズ')
    parser.add_argument('--src1', type=str, default='close', help='1つ目のソースタイプ')
    parser.add_argument('--src2', type=str, default='hlc3', help='2つ目のソースタイプ')
    parser.add_argument('--src3', type=str, default='hl2', help='3つ目のソースタイプ')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-individual', action='store_true', help='個別手法を非表示')
    parser.add_argument('--no-show', action='store_true', help='matplotlibでの表示を無効化（保存のみ）')
    args = parser.parse_args()
    
    # ハースト指数設定の作成
    hurst_configs = [
        {'window_size': args.window1, 'src_type': args.src1, 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': f'Hurst_{args.window1}_{args.src1.upper()}'},
        {'window_size': args.window2, 'src_type': args.src2, 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': f'Hurst_{args.window2}_{args.src2.upper()}'},
        {'window_size': args.window3, 'src_type': args.src3, 'enable_rs': True, 'enable_dfa': True, 'enable_wavelet': True, 'name': f'Hurst_{args.window3}_{args.src3.upper()}'}
    ]
    
    # チャートを作成
    chart = UnifiedHurstChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(hurst_configs=hurst_configs)
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_individual_methods=not args.no_individual,
        show_plot=not args.no_show,
        savefig=args.output
    )


if __name__ == "__main__":
    main()