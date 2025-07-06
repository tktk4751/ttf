#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Ultimate Breakout Channel Live Demo
リアルタイム市場データでのUltimate Breakout Channel実装デモ

z_adaptive_trend.pyの実装スタイルを参考に作成された、
実用的なリアルタイム市場データ分析・可視化ツール

機能:
- Binance APIからのリアルタイムデータ取得
- Ultimate Breakout Channelの計算・表示
- ブレイクアウトシグナルの視覚化
- トレンド強度とマーケットレジームの分析
- 日本語フォント対応の美しいチャート
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import mplfinance as mpf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# インジケーターのインポート
from indicators.ultimate_breakout_channel import UltimateBreakoutChannel, UBC
from api.binance_data_fetcher import BinanceDataFetcher

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class UltimateBreakoutChannelLiveDemo:
    """
    Ultimate Breakout Channel リアルタイムデモクラス
    """
    
    def __init__(self, symbol: str = 'BTC/USDT', timeframe: str = '4h', market_type: str = 'spot'):
        """
        初期化
        
        Args:
            symbol: 取引ペア（デフォルト: 'BTC/USDT'）
            timeframe: 時間足（デフォルト: '4h'）
            market_type: 市場タイプ（'spot' または 'future'）
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.market_type = market_type
        
        # データ取得クラス
        self.data_fetcher = BinanceDataFetcher(market_type=market_type)
        
        # Ultimate Breakout Channel インジケーター
        self.ubc = UltimateBreakoutChannel(
            atr_period=14,
            base_multiplier=2.0,
            min_multiplier=1.0,
            max_multiplier=8.0,
            hilbert_window=8,
            her_window=14,
            wavelet_window=16,
            src_type='hlc3',
            min_signal_quality=0.3
        )
        
        # データとして結果保存用
        self.data = None
        self.result = None
        
        print(f"🚀 Ultimate Breakout Channel Live Demo 初期化完了")
        print(f"📊 銘柄: {symbol}, 時間足: {timeframe}, 市場: {market_type}")
    
    def fetch_data(self, limit: int = 500) -> pd.DataFrame:
        """
        最新の市場データを取得
        
        Args:
            limit: 取得するデータ数（デフォルト: 500）
            
        Returns:
            価格データのDataFrame
        """
        print(f"\n📡 {self.symbol} の最新データを取得中...")
        
        try:
            # 最新データを取得
            self.data = self.data_fetcher.get_latest_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            if self.data.empty:
                raise ValueError("データの取得に失敗しました")
            
            print(f"✅ データ取得完了: {len(self.data)}件")
            print(f"📅 期間: {self.data.index[0]} → {self.data.index[-1]}")
            print(f"💰 最新価格: ${self.data['close'].iloc[-1]:,.2f}")
            
            return self.data
            
        except Exception as e:
            print(f"⚠️ データ取得エラー: {e}")
            # フォールバック: サンプルデータを生成
            print("🔄 サンプルデータを生成します...")
            return self._generate_sample_data(limit)
    
    def _generate_sample_data(self, n_samples: int) -> pd.DataFrame:
        """
        サンプルデータ生成（データ取得失敗時のフォールバック）
        
        Args:
            n_samples: サンプル数
            
        Returns:
            サンプルデータ
        """
        np.random.seed(42)
        
        # 時系列生成
        end_time = datetime.now()
        if self.timeframe == '1h':
            start_time = end_time - timedelta(hours=n_samples)
            freq = '1H'
        elif self.timeframe == '4h':
            start_time = end_time - timedelta(hours=n_samples * 4)
            freq = '4H'
        elif self.timeframe == '1d':
            start_time = end_time - timedelta(days=n_samples)
            freq = '1D'
        else:
            start_time = end_time - timedelta(hours=n_samples)
            freq = '1H'
        
        dates = pd.date_range(start=start_time, end=end_time, periods=n_samples)
        
        # 価格データ生成（BTC風のボラティリティ）
        price_base = 45000
        returns = np.random.normal(0, 0.025, n_samples)  # 2.5%のボラティリティ
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.1  # サイクル的トレンド
        
        cumulative_returns = np.cumsum(returns + trend * 0.1)
        close_prices = price_base * np.exp(cumulative_returns)
        
        # OHLV生成
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
        
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        volume = np.random.exponential(1000, n_samples)
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        print(f"📊 サンプルデータ生成完了: {len(df)}件")
        self.data = df
        return df
    
    def calculate_indicators(self) -> None:
        """
        Ultimate Breakout Channelインジケーターを計算
        """
        if self.data is None or self.data.empty:
            raise ValueError("データが読み込まれていません")
        
        print(f"\n🧮 Ultimate Breakout Channel 計算中...")
        
        try:
            # インジケーター計算
            self.result = self.ubc.calculate(self.data)
            
            # 結果取得
            channels = self.ubc.get_channels()
            signals = self.ubc.get_breakout_signals()
            signal_quality = self.ubc.get_signal_quality()
            trend_analysis = self.ubc.get_trend_analysis()
            
            # 結果統計
            if channels is not None:
                upper, lower, center = channels
                valid_count = (~np.isnan(upper)).sum()
                print(f"✅ チャネル計算完了: {valid_count}/{len(upper)} 有効データ")
            
            if signals is not None:
                total_signals = int(np.sum(np.abs(signals)))
                buy_signals = int(np.sum(signals == 1))
                sell_signals = int(np.sum(signals == -1))
                print(f"🎯 シグナル: 買い={buy_signals}, 売り={sell_signals}, 合計={total_signals}")
            
            if signal_quality is not None:
                avg_quality = np.nanmean(signal_quality[signal_quality > 0])
                print(f"📊 平均シグナル品質: {avg_quality:.3f}")
            
            # 知能レポート取得
            intelligence_report = self.ubc.get_intelligence_report()
            print(f"🧠 現在の市場状況: {intelligence_report.get('current_trend', 'unknown')}")
            print(f"🔮 信頼度: {intelligence_report.get('current_confidence', 0):.3f}")
            print(f"🌊 レジーム: {intelligence_report.get('current_regime', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ 計算エラー: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_comprehensive_chart(self, 
                                  title: str = None,
                                  figsize: tuple = (16, 12),
                                  last_n: int = None,
                                  save_path: str = None) -> None:
        """
        包括的なチャート作成
        
        Args:
            title: チャートタイトル
            figsize: チャートサイズ
            last_n: 表示する最新n個のデータ
            save_path: 保存パス
        """
        if self.data is None or self.result is None:
            raise ValueError("データまたは計算結果がありません")
        
        # データの準備
        data = self.data.copy()
        if last_n is not None:
            data = data.tail(last_n)
        
        # インジケーター結果の取得
        channels = self.ubc.get_channels()
        signals = self.ubc.get_breakout_signals()
        signal_quality = self.ubc.get_signal_quality()
        trend_analysis = self.ubc.get_trend_analysis()
        
        if channels is None:
            print("⚠️ チャネルデータが取得できません")
            return
        
        upper, lower, center = channels
        
        # データ範囲の調整
        if last_n is not None:
            upper = upper[-last_n:]
            lower = lower[-last_n:]
            center = center[-last_n:]
            if signals is not None:
                signals = signals[-last_n:]
            if signal_quality is not None:
                signal_quality = signal_quality[-last_n:]
            if trend_analysis is not None:
                for key in trend_analysis:
                    if isinstance(trend_analysis[key], np.ndarray):
                        trend_analysis[key] = trend_analysis[key][-last_n:]
        
        # チャート作成
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 1])
        fig.suptitle(
            title or f"🚀 Ultimate Breakout Channel - {self.symbol} ({self.timeframe})",
            fontsize=16, fontweight='bold'
        )
        
        # === 1. メインチャート（価格 + チャネル + シグナル） ===
        ax1 = axes[0]
        
        # 価格データプロット
        ax1.plot(data.index, data['close'], 'k-', linewidth=1.5, label='価格', alpha=0.8)
        
        # チャネルプロット
        valid_mask = ~np.isnan(upper) & ~np.isnan(lower) & ~np.isnan(center)
        if np.any(valid_mask):
            valid_dates = data.index[valid_mask]
            ax1.plot(valid_dates, upper[valid_mask], 'r-', linewidth=2, label='上部チャネル', alpha=0.7)
            ax1.plot(valid_dates, lower[valid_mask], 'g-', linewidth=2, label='下部チャネル', alpha=0.7)
            ax1.plot(valid_dates, center[valid_mask], 'b--', linewidth=1.5, label='センターライン', alpha=0.7)
            
            # チャネル間の塗りつぶし
            ax1.fill_between(valid_dates, upper[valid_mask], lower[valid_mask], 
                           alpha=0.1, color='gray', label='チャネル')
        
        # ブレイクアウトシグナル
        if signals is not None:
            buy_mask = signals == 1
            sell_mask = signals == -1
            
            if np.any(buy_mask):
                ax1.scatter(data.index[buy_mask], data['close'][buy_mask], 
                          color='lime', marker='^', s=100, label='買いシグナル', zorder=5)
            
            if np.any(sell_mask):
                ax1.scatter(data.index[sell_mask], data['close'][sell_mask], 
                          color='red', marker='v', s=100, label='売りシグナル', zorder=5)
        
        ax1.set_ylabel('価格 (USDT)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # === 2. トレンド強度とシグナル品質 ===
        ax2 = axes[1]
        
        if trend_analysis is not None and 'trend_strength' in trend_analysis:
            trend_strength = trend_analysis['trend_strength']
            valid_trend = ~np.isnan(trend_strength)
            if np.any(valid_trend):
                ax2.plot(data.index[valid_trend], trend_strength[valid_trend], 
                        'purple', linewidth=2, label='トレンド強度')
        
        if signal_quality is not None:
            valid_quality = ~np.isnan(signal_quality) & (signal_quality > 0)
            if np.any(valid_quality):
                ax2.scatter(data.index[valid_quality], signal_quality[valid_quality], 
                          color='orange', s=30, alpha=0.7, label='シグナル品質')
        
        ax2.set_ylabel('強度/品質', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # === 3. 量子コヒーレンスとハイパー効率率 ===
        ax3 = axes[2]
        
        if trend_analysis is not None:
            if 'quantum_coherence' in trend_analysis:
                quantum_coherence = trend_analysis['quantum_coherence']
                valid_coherence = ~np.isnan(quantum_coherence)
                if np.any(valid_coherence):
                    ax3.plot(data.index[valid_coherence], quantum_coherence[valid_coherence], 
                            'cyan', linewidth=1.5, label='量子コヒーレンス')
            
            if 'hyper_efficiency' in trend_analysis:
                hyper_efficiency = trend_analysis['hyper_efficiency']
                valid_efficiency = ~np.isnan(hyper_efficiency)
                if np.any(valid_efficiency):
                    ax3.plot(data.index[valid_efficiency], hyper_efficiency[valid_efficiency], 
                            'magenta', linewidth=1.5, label='ハイパー効率率')
        
        ax3.set_ylabel('指標値', fontsize=12)
        ax3.set_xlabel('時間', fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(data)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 チャートを保存しました: {save_path}")
        
        plt.show()
        
        # 統計情報表示
        self._print_statistics()
    
    def _print_statistics(self) -> None:
        """
        統計情報を表示
        """
        print(f"\n📊 === Ultimate Breakout Channel 統計情報 ===")
        
        # 基本情報
        intelligence_report = self.ubc.get_intelligence_report()
        print(f"🎯 現在のトレンド: {intelligence_report.get('current_trend', 'N/A')}")
        print(f"🔮 現在の信頼度: {intelligence_report.get('current_confidence', 0):.3f}")
        print(f"🌊 現在のレジーム: {intelligence_report.get('current_regime', 'N/A')}")
        print(f"📈 総シグナル数: {intelligence_report.get('total_signals', 0)}")
        print(f"📊 平均シグナル品質: {intelligence_report.get('avg_signal_quality', 0):.3f}")
        print(f"💪 トレンド強度: {intelligence_report.get('trend_strength', 0):.3f}")
        print(f"🌀 量子コヒーレンス: {intelligence_report.get('quantum_coherence', 0):.3f}")
        print(f"⚡ システム効率: {intelligence_report.get('system_efficiency', 0):.3f}")
        
        # 市場分析
        market_analysis = self.ubc.get_market_analysis()
        if market_analysis:
            print(f"\n🏪 === 市場分析 ===")
            print(f"📈 トレンド率: {market_analysis.get('trending_ratio', 0):.1%}")
            print(f"🌊 サイクル強度: {market_analysis.get('cycle_strength', 0):.3f}")
        
        # 最新価格情報
        if self.data is not None and not self.data.empty:
            latest_price = self.data['close'].iloc[-1]
            price_change = self.data['close'].iloc[-1] - self.data['close'].iloc[-2] if len(self.data) > 1 else 0
            price_change_pct = (price_change / self.data['close'].iloc[-2] * 100) if len(self.data) > 1 else 0
            
            print(f"\n💰 === 価格情報 ===")
            print(f"💵 最新価格: ${latest_price:,.2f}")
            print(f"📊 変化額: ${price_change:+,.2f}")
            print(f"📈 変化率: {price_change_pct:+.2f}%")


def run_live_demo():
    """
    ライブデモを実行
    """
    print("🚀 Ultimate Breakout Channel Live Demo 開始")
    print("=" * 60)
    
    # デモインスタンス作成
    demo = UltimateBreakoutChannelLiveDemo(
        symbol='BTC/USDT',
        timeframe='4h',
        market_type='spot'
    )
    
    try:
        # 1. データ取得
        demo.fetch_data(limit=500)
        
        # 2. インジケーター計算
        demo.calculate_indicators()
        
        # 3. チャート作成
        save_path = f"examples/output/ultimate_breakout_channel_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        demo.create_comprehensive_chart(
            title=f"🚀 Ultimate Breakout Channel Live Analysis - {demo.symbol}",
            last_n=200,  # 最新200件を表示
            save_path=save_path
        )
        
        print("\n✅ Ultimate Breakout Channel Live Demo 完了!")
        
    except Exception as e:
        print(f"⚠️ デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def run_custom_demo(symbol: str = 'ETH/USDT', 
                   timeframe: str = '1h', 
                   limit: int = 300,
                   min_multiplier: float = 1.0,
                   max_multiplier: float = 6.0):
    """
    カスタムパラメータでのデモ実行
    
    Args:
        symbol: 取引ペア
        timeframe: 時間足
        limit: データ数
        min_multiplier: 最小乗数
        max_multiplier: 最大乗数
    """
    print(f"🚀 Ultimate Breakout Channel カスタムデモ - {symbol}")
    print("=" * 60)
    
    # カスタムデモインスタンス作成
    demo = UltimateBreakoutChannelLiveDemo(
        symbol=symbol,
        timeframe=timeframe,
        market_type='spot'
    )
    
    # カスタムパラメータでUBC初期化
    demo.ubc = UltimateBreakoutChannel(
        atr_period=14,
        base_multiplier=2.0,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        hilbert_window=8,
        her_window=14,
        wavelet_window=16,
        src_type='hlc3',
        min_signal_quality=0.3
    )
    
    try:
        # デモ実行
        demo.fetch_data(limit=limit)
        demo.calculate_indicators()
        
        save_path = f"examples/output/ultimate_breakout_channel_custom_{symbol.replace('/', '')}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        demo.create_comprehensive_chart(
            title=f"🚀 Ultimate Breakout Channel Custom - {symbol} ({timeframe})",
            last_n=min(200, limit),
            save_path=save_path
        )
        
        print(f"\n✅ カスタムデモ完了: {symbol}")
        
    except Exception as e:
        print(f"⚠️ カスタムデモ実行中にエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Ultimate Breakout Channel Live Demo")
    print("=" * 60)
    print("1. BTC/USDT 4時間足でのライブデモ")
    print("2. ETH/USDT 1時間足でのカスタムデモ")
    print("3. 複数銘柄での比較分析")
    print("=" * 60)
    
    # output ディレクトリが存在しない場合は作成
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # メインデモ実行
        run_live_demo()
        
        print("\n" + "="*60)
        print("🔄 追加デモを実行中...")
        
        # カスタムデモ実行
        run_custom_demo(
            symbol='ETH/USDT',
            timeframe='1h',
            limit=300,
            min_multiplier=0.8,
            max_multiplier=5.0
        )
        
        print("\n🎉 すべてのデモが完了しました!")
        print(f"📁 結果は {output_dir} ディレクトリに保存されています。")
        
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによってデモが中断されました")
    except Exception as e:
        print(f"\n⚠️ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc() 