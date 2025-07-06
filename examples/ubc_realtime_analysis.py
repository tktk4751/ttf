#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Ultimate Breakout Channel リアルタイム分析ツール

z_adaptive_trend.pyの実装スタイルを参考に作成された、
Ultimate Breakout Channelのリアルタイム市場データ分析・可視化ツール

主な機能:
- Binance APIからのリアルタイムデータ取得
- Ultimate Breakout Channelの計算・可視化
- ブレイクアウトシグナルの表示
- トレンド強度とマーケットレジームの分析
- 統計情報とパフォーマンスメトリクス
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパスの追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 必要なモジュールのインポート
try:
    from indicators.ultimate_breakout_channel import UltimateBreakoutChannel
    from api.binance_data_fetcher import BinanceDataFetcher
    print("✅ 必要なモジュールのインポート成功")
except ImportError as e:
    print(f"⚠️ インポートエラー: {e}")
    print("🔄 代替実装を使用します")

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']


def generate_sample_data(n_samples: int = 500, symbol: str = 'BTC/USDT') -> pd.DataFrame:
    """
    サンプルデータ生成（リアルデータ取得失敗時のフォールバック）
    
    Args:
        n_samples: サンプル数
        symbol: 銘柄名（表示用）
        
    Returns:
        サンプル価格データ
    """
    print(f"📊 {symbol}のサンプルデータを生成中...")
    
    np.random.seed(42)
    
    # 時系列インデックス生成
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=n_samples * 4)  # 4時間足想定
    dates = pd.date_range(start=start_time, end=end_time, periods=n_samples)
    
    # 価格データ生成（現実的なBTC価格動向をシミュレート）
    base_price = 45000
    
    # トレンド成分
    trend = np.linspace(0, 0.15, n_samples)
    
    # ボラティリティ成分
    volatility = np.random.normal(0, 0.025, n_samples)
    
    # サイクル成分（市場の周期性）
    cycle = np.sin(np.linspace(0, 6*np.pi, n_samples)) * 0.05
    
    # ノイズ成分
    noise = np.random.normal(0, 0.01, n_samples)
    
    # 価格系列合成
    log_returns = trend * 0.01 + volatility + cycle + noise
    cumulative_returns = np.cumsum(log_returns)
    close_prices = base_price * np.exp(cumulative_returns)
    
    # OHLV データ生成
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.008, n_samples)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.008, n_samples)))
    
    # 開始価格（前日終値）
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # 出来高（対数正規分布）
    volume = np.random.lognormal(7, 0.5, n_samples)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    print(f"✅ サンプルデータ生成完了: {len(df)}件")
    print(f"📅 期間: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 価格範囲: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    
    return df


def fetch_realtime_data(symbol: str = 'BTC/USDT', timeframe: str = '4h', limit: int = 500) -> pd.DataFrame:
    """
    リアルタイムデータ取得
    
    Args:
        symbol: 取引ペア
        timeframe: 時間足
        limit: データ数
        
    Returns:
        価格データ
    """
    print(f"📡 {symbol} ({timeframe}) のリアルタイムデータを取得中...")
    
    try:
        # Binanceデータフェッチャー初期化
        fetcher = BinanceDataFetcher(market_type='spot')
        
        # データ取得
        data = fetcher.get_latest_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        if data.empty:
            raise ValueError("取得データが空です")
        
        print(f"✅ リアルタイムデータ取得成功: {len(data)}件")
        print(f"📅 期間: {data.index[0]} → {data.index[-1]}")
        print(f"💰 最新価格: ${data['close'].iloc[-1]:,.2f}")
        
        return data
        
    except Exception as e:
        print(f"⚠️ リアルタイムデータ取得失敗: {e}")
        print("🔄 サンプルデータを使用します")
        return generate_sample_data(limit, symbol)


def calculate_ubc_indicators(data: pd.DataFrame, config: dict = None) -> dict:
    """
    Ultimate Breakout Channel インジケーターを計算
    
    Args:
        data: 価格データ
        config: インジケーター設定
        
    Returns:
        計算結果辞書
    """
    if config is None:
        config = {
            'atr_period': 14,
            'base_multiplier': 2.0,
            'min_multiplier': 1.0,
            'max_multiplier': 8.0,
            'hilbert_window': 8,
            'her_window': 14,
            'wavelet_window': 16,
            'src_type': 'hlc3',
            'min_signal_quality': 0.3
        }
    
    print(f"🧮 Ultimate Breakout Channel 計算中...")
    print(f"📊 設定: ATR={config['atr_period']}, 乗数={config['min_multiplier']}-{config['max_multiplier']}")
    
    try:
        # UBC インジケーター初期化
        ubc = UltimateBreakoutChannel(**config)
        
        # 計算実行
        result = ubc.calculate(data)
        
        # 結果取得
        channels = ubc.get_channels()
        signals = ubc.get_breakout_signals()
        signal_quality = ubc.get_signal_quality()
        trend_analysis = ubc.get_trend_analysis()
        intelligence_report = ubc.get_intelligence_report()
        market_analysis = ubc.get_market_analysis()
        
        # 結果の検証
        if channels is not None:
            upper, lower, center = channels
            valid_count = (~np.isnan(upper)).sum()
            print(f"✅ チャネル計算完了: {valid_count}/{len(upper)} 有効データ")
        else:
            print("⚠️ チャネルデータの取得に失敗")
            
        if signals is not None:
            total_signals = int(np.sum(np.abs(signals)))
            buy_signals = int(np.sum(signals == 1))
            sell_signals = int(np.sum(signals == -1))
            print(f"🎯 シグナル: 買い={buy_signals}, 売り={sell_signals}, 合計={total_signals}")
        else:
            print("⚠️ シグナルデータの取得に失敗")
            
        return {
            'ubc': ubc,
            'result': result,
            'channels': channels,
            'signals': signals,
            'signal_quality': signal_quality,
            'trend_analysis': trend_analysis,
            'intelligence_report': intelligence_report,
            'market_analysis': market_analysis
        }
        
    except Exception as e:
        print(f"❌ UBC 計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return {}


def create_comprehensive_chart(data: pd.DataFrame, 
                             indicators: dict, 
                             symbol: str = 'BTC/USDT',
                             timeframe: str = '4h',
                             last_n: int = 200,
                             save_path: str = None) -> None:
    """
    包括的なチャート作成・表示
    
    Args:
        data: 価格データ
        indicators: インジケーター計算結果
        symbol: 銘柄名
        timeframe: 時間足
        last_n: 表示するデータ数
        save_path: 保存パス
    """
    if not indicators or 'channels' not in indicators:
        print("⚠️ インジケーターデータが不正です")
        return
        
    print(f"🎨 チャート作成中...")
    
    # データの準備
    plot_data = data.tail(last_n).copy()
    
    # インジケーターデータの準備
    channels = indicators['channels']
    signals = indicators['signals']
    signal_quality = indicators['signal_quality']
    trend_analysis = indicators['trend_analysis']
    
    if channels is None:
        print("⚠️ チャネルデータがありません")
        return
        
    upper, lower, center = channels
    
    # データ範囲調整
    upper = upper[-last_n:] if len(upper) >= last_n else upper
    lower = lower[-last_n:] if len(lower) >= last_n else lower
    center = center[-last_n:] if len(center) >= last_n else center
    
    if signals is not None:
        signals = signals[-last_n:] if len(signals) >= last_n else signals
    if signal_quality is not None:
        signal_quality = signal_quality[-last_n:] if len(signal_quality) >= last_n else signal_quality
    
    # チャート作成
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
    fig.suptitle(f'🚀 Ultimate Breakout Channel Analysis - {symbol} ({timeframe})', 
                fontsize=16, fontweight='bold')
    
    # === 1. メインチャート（価格 + チャネル + シグナル） ===
    ax1 = axes[0]
    
    # 価格ライン
    ax1.plot(plot_data.index, plot_data['close'], 'k-', linewidth=1.5, label='価格', alpha=0.8)
    
    # チャネルライン
    valid_mask = ~np.isnan(upper) & ~np.isnan(lower) & ~np.isnan(center)
    if np.any(valid_mask):
        valid_dates = plot_data.index[valid_mask]
        
        # 上部・下部チャネル
        ax1.plot(valid_dates, upper[valid_mask], 'r-', linewidth=2, label='上部チャネル', alpha=0.7)
        ax1.plot(valid_dates, lower[valid_mask], 'g-', linewidth=2, label='下部チャネル', alpha=0.7)
        ax1.plot(valid_dates, center[valid_mask], 'b--', linewidth=1.5, label='センターライン', alpha=0.7)
        
        # チャネル間塗りつぶし
        ax1.fill_between(valid_dates, upper[valid_mask], lower[valid_mask], 
                        alpha=0.1, color='gray', label='チャネル範囲')
    
    # ブレイクアウトシグナル
    if signals is not None:
        buy_mask = (signals == 1)
        sell_mask = (signals == -1)
        
        if np.any(buy_mask):
            ax1.scatter(plot_data.index[buy_mask], plot_data['close'][buy_mask], 
                       color='lime', marker='^', s=120, label='買いシグナル', zorder=5, edgecolor='black')
        
        if np.any(sell_mask):
            ax1.scatter(plot_data.index[sell_mask], plot_data['close'][sell_mask], 
                       color='red', marker='v', s=120, label='売りシグナル', zorder=5, edgecolor='black')
    
    ax1.set_ylabel('価格 (USDT)', fontsize=12)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # === 2. トレンド強度・シグナル品質 ===
    ax2 = axes[1]
    
    if trend_analysis and 'trend_strength' in trend_analysis:
        trend_strength = trend_analysis['trend_strength']
        if trend_strength is not None:
            trend_strength = trend_strength[-last_n:] if len(trend_strength) >= last_n else trend_strength
            valid_trend = ~np.isnan(trend_strength)
            if np.any(valid_trend):
                ax2.plot(plot_data.index[valid_trend], trend_strength[valid_trend], 
                        'purple', linewidth=2, label='トレンド強度', alpha=0.8)
    
    if signal_quality is not None:
        valid_quality = ~np.isnan(signal_quality) & (signal_quality > 0)
        if np.any(valid_quality):
            ax2.scatter(plot_data.index[valid_quality], signal_quality[valid_quality], 
                       color='orange', s=30, alpha=0.7, label='シグナル品質')
    
    ax2.set_ylabel('強度/品質', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # === 3. 量子コヒーレンス・ハイパー効率率 ===
    ax3 = axes[2]
    
    if trend_analysis:
        # 量子コヒーレンス
        if 'quantum_coherence' in trend_analysis and trend_analysis['quantum_coherence'] is not None:
            quantum_coherence = trend_analysis['quantum_coherence']
            quantum_coherence = quantum_coherence[-last_n:] if len(quantum_coherence) >= last_n else quantum_coherence
            valid_coherence = ~np.isnan(quantum_coherence)
            if np.any(valid_coherence):
                ax3.plot(plot_data.index[valid_coherence], quantum_coherence[valid_coherence], 
                        'cyan', linewidth=1.5, label='量子コヒーレンス', alpha=0.8)
        
        # ハイパー効率率
        if 'hyper_efficiency' in trend_analysis and trend_analysis['hyper_efficiency'] is not None:
            hyper_efficiency = trend_analysis['hyper_efficiency']
            hyper_efficiency = hyper_efficiency[-last_n:] if len(hyper_efficiency) >= last_n else hyper_efficiency
            valid_efficiency = ~np.isnan(hyper_efficiency)
            if np.any(valid_efficiency):
                ax3.plot(plot_data.index[valid_efficiency], hyper_efficiency[valid_efficiency], 
                        'magenta', linewidth=1.5, label='ハイパー効率率', alpha=0.8)
    
    ax3.set_ylabel('指標値', fontsize=12)
    ax3.set_xlabel('時間', fontsize=12)
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # 日付軸フォーマット
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(plot_data)//8)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 チャートを保存: {save_path}")
    
    plt.show()
    
    # 統計情報の表示
    print_comprehensive_statistics(indicators, plot_data)


def print_comprehensive_statistics(indicators: dict, data: pd.DataFrame) -> None:
    """
    包括的な統計情報を表示
    
    Args:
        indicators: インジケーター結果
        data: 価格データ
    """
    print(f"\n" + "="*60)
    print(f"📊 Ultimate Breakout Channel 統計サマリー")
    print(f"="*60)
    
    # 知能レポート
    if 'intelligence_report' in indicators and indicators['intelligence_report']:
        report = indicators['intelligence_report']
        print(f"🧠 知能分析:")
        print(f"  🎯 現在トレンド: {report.get('current_trend', 'N/A')}")
        print(f"  🔮 信頼度: {report.get('current_confidence', 0):.3f}")
        print(f"  🌊 マーケットレジーム: {report.get('current_regime', 'N/A')}")
        print(f"  📈 総シグナル数: {report.get('total_signals', 0)}")
        print(f"  📊 平均シグナル品質: {report.get('avg_signal_quality', 0):.3f}")
        print(f"  💪 トレンド強度: {report.get('trend_strength', 0):.3f}")
        print(f"  🌀 量子コヒーレンス: {report.get('quantum_coherence', 0):.3f}")
        print(f"  ⚡ システム効率: {report.get('system_efficiency', 0):.3f}")
    
    # マーケット分析
    if 'market_analysis' in indicators and indicators['market_analysis']:
        market = indicators['market_analysis']
        print(f"\n🏪 マーケット分析:")
        print(f"  📈 総トレンド率: {market.get('trending_ratio', 0):.1%}")
        print(f"    🚀 超強トレンド: {market.get('very_strong_trend_ratio', 0):.1%}")
        print(f"    🔥 強トレンド: {market.get('strong_trend_ratio', 0):.1%}")
        print(f"    📊 中トレンド: {market.get('moderate_trend_ratio', 0):.1%}")
        print(f"    📉 弱トレンド: {market.get('weak_trend_ratio', 0):.1%}")
        print(f"  🔄 総サイクル率: {market.get('cycling_ratio', 0):.1%}")
        print(f"    🌀 強サイクル: {market.get('strong_cycle_ratio', 0):.1%}")
        print(f"    🌊 弱サイクル: {market.get('weak_cycle_ratio', 0):.1%}")
        print(f"  📊 レンジ率: {market.get('range_ratio', 0):.1%}")
        print(f"  💫 サイクル強度: {market.get('cycle_strength', 0):.3f}")
        print(f"  📊 分析ポイント数: {market.get('total_regime_points', 0)}")
    
    # 価格統計
    if not data.empty:
        latest_price = data['close'].iloc[-1]
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2] if len(data) > 1 else 0
        price_change_pct = (price_change / data['close'].iloc[-2] * 100) if len(data) > 1 else 0
        volatility = data['close'].pct_change().std() * 100
        
        print(f"\n💰 価格統計:")
        print(f"  💵 最新価格: ${latest_price:,.2f}")
        print(f"  📊 変化額: ${price_change:+,.2f}")
        print(f"  📈 変化率: {price_change_pct:+.2f}%")
        print(f"  📉 ボラティリティ: {volatility:.2f}%")
        print(f"  🔄 価格範囲: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    
    print(f"="*60)


def main():
    """
    メイン実行関数
    """
    print("🚀 Ultimate Breakout Channel リアルタイム分析ツール")
    print("="*60)
    
    # 設定
    symbol = 'BTC/USDT'
    timeframe = '4h'
    limit = 500
    
    # outputディレクトリ作成
    output_dir = "examples/output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. データ取得
        print(f"\n📡 データ取得フェーズ")
        data = fetch_realtime_data(symbol=symbol, timeframe=timeframe, limit=limit)
        
        if data.empty:
            print("❌ データ取得に失敗しました")
            return
        
        # 2. インジケーター計算
        print(f"\n🧮 インジケーター計算フェーズ")
        indicators = calculate_ubc_indicators(data)
        
        if not indicators:
            print("❌ インジケーター計算に失敗しました")
            return
        
        # 3. チャート作成・表示
        print(f"\n🎨 可視化フェーズ")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"{output_dir}/ubc_analysis_{symbol.replace('/', '')}_{timeframe}_{timestamp}.png"
        
        create_comprehensive_chart(
            data=data,
            indicators=indicators,
            symbol=symbol,
            timeframe=timeframe,
            last_n=200,
            save_path=save_path
        )
        
        print(f"\n✅ 分析完了!")
        print(f"📁 結果ファイル: {save_path}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()


# カスタム分析関数
def run_custom_analysis(symbol: str = 'ETH/USDT', 
                       timeframe: str = '1h',
                       min_multiplier: float = 0.8,
                       max_multiplier: float = 6.0,
                       limit: int = 300):
    """
    カスタムパラメータでの分析実行
    
    Args:
        symbol: 分析対象銘柄
        timeframe: 時間足
        min_multiplier: 最小チャネル乗数
        max_multiplier: 最大チャネル乗数
        limit: データ数
    """
    print(f"🚀 カスタム分析: {symbol} ({timeframe})")
    print("="*60)
    
    # カスタム設定
    config = {
        'atr_period': 14,
        'base_multiplier': 2.0,
        'min_multiplier': min_multiplier,
        'max_multiplier': max_multiplier,
        'hilbert_window': 8,
        'her_window': 14,
        'wavelet_window': 16,
        'src_type': 'hlc3',
        'min_signal_quality': 0.3
    }
    
    try:
        # データ取得
        data = fetch_realtime_data(symbol=symbol, timeframe=timeframe, limit=limit)
        
        # インジケーター計算
        indicators = calculate_ubc_indicators(data, config)
        
        # チャート作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"examples/output/ubc_custom_{symbol.replace('/', '')}_{timeframe}_{timestamp}.png"
        
        create_comprehensive_chart(
            data=data,
            indicators=indicators,
            symbol=symbol,
            timeframe=timeframe,
            last_n=min(200, limit),
            save_path=save_path
        )
        
        print(f"✅ カスタム分析完了: {symbol}")
        
    except Exception as e:
        print(f"❌ カスタム分析エラー: {e}")


if __name__ == "__main__":
    # メイン分析実行
    main()
    
    print(f"\n" + "="*60)
    print(f"🔄 追加分析...")
    
    # カスタム分析実行
    run_custom_analysis(
        symbol='ETH/USDT',
        timeframe='1h',
        min_multiplier=0.8,
        max_multiplier=5.0,
        limit=300
    )
    
    print(f"\n🎉 すべての分析が完了しました!")
