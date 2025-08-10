#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
import sys
import os

# パスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.laguerre_rsi import LaguerreRSI
from data.data_loader import DataLoader,CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource


def load_config(config_path: str = 'config.yaml') -> dict:
    """設定ファイルを読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}")
        return {}


def load_market_data(config: dict) -> dict:
    """設定ファイルから市場データを読み込む"""
    try:
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
        print("Loading and processing data...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        return processed_data
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return {}


def create_laguerre_rsi_chart(
    data: pd.DataFrame,
    symbol: str = "Crypto",
    gamma_basic: float = 0.9,
    gamma_sensitive: float = 0.5,
    period: float = 20.0,
    use_roofing_filter: bool = True,
    src_type: str = 'oc2',
    save_path: str = None,
    show_plot: bool = True
):
    """
    ラゲールRSIのチャートを作成する
    
    Args:
        data: 価格データ（OHLCV）
        symbol: シンボル名
        gamma_basic: 基本版のガンマ値
        gamma_sensitive: 高感度版のガンマ値
        period: UltimateSmoother期間
        use_roofing_filter: ルーフィングフィルターを使用するか
        src_type: ソースタイプ
        save_path: 保存パス（Noneの場合は保存しない）
        show_plot: プロットを表示するか
    """
    
    # 直近500本のデータに制限
    if len(data) > 500:
        data = data.tail(500)
        print(f"直近500本のデータを使用")
    
    print(f"\n=== {symbol} ラゲールRSI分析 ===")
    print(f"データ期間: {data.index[0]} - {data.index[-1]}")
    print(f"データポイント数: {len(data)}")
    
    # 基本版ラゲールRSI
    print(f"\n基本版ラゲールRSI計算中... (gamma={gamma_basic})")
    lrsi_basic = LaguerreRSI(
        gamma=gamma_basic,
        src_type=src_type,
        period=period,
        use_roofing_filter=True
    )
    result_basic = lrsi_basic.calculate(data)
    
    # 高感度版ラゲールRSI
    print(f"高感度版ラゲールRSI計算中... (gamma={gamma_sensitive})")
    lrsi_sensitive = LaguerreRSI(
        gamma=gamma_sensitive,
        src_type=src_type,
        period=period,
        use_roofing_filter=True
    )
    result_sensitive = lrsi_sensitive.calculate(data)
    
    # ルーフィングフィルター版ラゲールRSI（オプション）
    result_filtered = None
    if use_roofing_filter:
        print(f"ルーフィングフィルター版ラゲールRSI計算中... (gamma={gamma_basic})")
        lrsi_filtered = LaguerreRSI(
            gamma=gamma_basic,
            src_type=src_type,
            period=period,
            use_roofing_filter=True,
            roofing_hp_cutoff=48.0,
            roofing_ss_band_edge=10.0
        )
        result_filtered = lrsi_filtered.calculate(data)
    
    # データの統計情報
    print(f"\n=== 統計情報 ===")
    valid_basic = np.sum(~np.isnan(result_basic.values))
    valid_sensitive = np.sum(~np.isnan(result_sensitive.values))
    
    if valid_basic > 0:
        mean_basic = np.nanmean(result_basic.values)
        std_basic = np.nanstd(result_basic.values)
        overbought_basic = np.sum(result_basic.values > 0.8) / valid_basic * 100
        oversold_basic = np.sum(result_basic.values < 0.2) / valid_basic * 100
        print(f"基本版 (gamma={gamma_basic}): 平均={mean_basic:.4f}±{std_basic:.4f}, "
              f"買われすぎ={overbought_basic:.1f}%, 売られすぎ={oversold_basic:.1f}%")
    
    if valid_sensitive > 0:
        mean_sensitive = np.nanmean(result_sensitive.values)
        std_sensitive = np.nanstd(result_sensitive.values)
        overbought_sensitive = np.sum(result_sensitive.values > 0.8) / valid_sensitive * 100
        oversold_sensitive = np.sum(result_sensitive.values < 0.2) / valid_sensitive * 100
        print(f"高感度版 (gamma={gamma_sensitive}): 平均={mean_sensitive:.4f}±{std_sensitive:.4f}, "
              f"買われすぎ={overbought_sensitive:.1f}%, 売られすぎ={oversold_sensitive:.1f}%")
    
    if result_filtered is not None:
        valid_filtered = np.sum(~np.isnan(result_filtered.values))
        if valid_filtered > 0:
            mean_filtered = np.nanmean(result_filtered.values)
            std_filtered = np.nanstd(result_filtered.values)
            print(f"ルーフィング版 (gamma={gamma_basic}): 平均={mean_filtered:.4f}±{std_filtered:.4f}")
    
    # チャートの作成
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'{symbol} - ラゲールRSI分析', fontsize=16, fontweight='bold', color='white')
    
    # 価格チャート（上部）
    ax_price = axes[0]
    ax_price.plot(data.index, data['close'], label='Close Price', color='cyan', linewidth=1.5)
    
    # 移動平均を追加（参考用）
    if len(data) >= 20:
        sma20 = data['close'].rolling(window=20).mean()
        ax_price.plot(data.index, sma20, label='SMA(20)', color='orange', alpha=0.7, linewidth=1)
    
    ax_price.set_ylabel('価格', fontsize=12, color='white')
    ax_price.set_title(f'{symbol} 価格チャート', fontsize=14, color='white')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    ax_price.tick_params(colors='white')
    
    # ラゲールRSIチャート（下部）
    ax_rsi = axes[1]
    
    # RSI値をプロット
    ax_rsi.plot(data.index, result_basic.values, 
               label=f'ラゲールRSI (γ={gamma_basic})', 
               color='lime', linewidth=2, alpha=0.8)
    
    ax_rsi.plot(data.index, result_sensitive.values, 
               label=f'高感度版 (γ={gamma_sensitive})', 
               color='yellow', linewidth=1.5, alpha=0.9)
    
    if result_filtered is not None:
        ax_rsi.plot(data.index, result_filtered.values, 
                   label=f'ルーフィング版 (γ={gamma_basic})', 
                   color='magenta', linewidth=1.5, alpha=0.8, linestyle='--')
    
    # 買われすぎ・売られすぎラインの追加
    ax_rsi.axhline(y=0.8, color='red', linestyle='-', alpha=0.7, linewidth=1.5, label='買われすぎ (0.8)')
    ax_rsi.axhline(y=0.2, color='blue', linestyle='-', alpha=0.7, linewidth=1.5, label='売られすぎ (0.2)')
    ax_rsi.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='中間 (0.5)')
    
    # 買われすぎ・売られすぎエリアのハイライト
    ax_rsi.fill_between(data.index, 0.8, 1.0, alpha=0.1, color='red', label='買われすぎエリア')
    ax_rsi.fill_between(data.index, 0.0, 0.2, alpha=0.1, color='blue', label='売られすぎエリア')
    
    ax_rsi.set_ylabel('ラゲールRSI値', fontsize=12, color='white')
    ax_rsi.set_xlabel('日時', fontsize=12, color='white')
    ax_rsi.set_title('ラゲールRSI比較', fontsize=14, color='white')
    ax_rsi.set_ylim(0, 1)
    ax_rsi.legend(loc='upper left')
    ax_rsi.grid(True, alpha=0.3)
    ax_rsi.tick_params(colors='white')
    
    # X軸の日付フォーマット
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 統計テキストの追加
    stats_text = f"""統計情報:
基本版: 平均={mean_basic:.3f}, シグナル={overbought_basic+oversold_basic:.1f}%
高感度版: 平均={mean_sensitive:.3f}, シグナル={overbought_sensitive+oversold_sensitive:.1f}%"""
    
    ax_rsi.text(0.02, 0.98, stats_text, transform=ax_rsi.transAxes, 
               verticalalignment='top', fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"チャートを保存しました: {save_path}")
    
    # 表示 - 常に表示するように変更
    plt.show()
    
    return fig, axes


def main():
    """メイン関数"""
    print("=== ラゲールRSIチャート生成 ===")
    
    # 設定ファイル読み込み
    config = load_config()
    if not config:
        print("設定ファイルが読み込めませんでした")
        return
    
    # データ読み込み
    market_data = load_market_data(config)
    if not market_data:
        print("市場データが読み込めませんでした")
        return
    
    print(f"読み込んだシンボル: {list(market_data.keys())}")
    
    # 各シンボルのチャートを生成
    for symbol, data in market_data.items():
        if len(data) < 50:
            print(f"スキップ: {symbol} (データが不十分: {len(data)}ポイント)")
            continue
        
        try:
            # チャート生成設定
            save_path = f"laguerre_rsi_{symbol.replace('/', '_')}_chart.png"
            
            # チャート作成
            create_laguerre_rsi_chart(
                data=data,
                symbol=symbol,
                gamma_basic=0.5,
                gamma_sensitive=0.3,
                period=20.0,
                use_roofing_filter=True,
                src_type='close',
                save_path=save_path,
                show_plot=True  # 常に表示するように変更
            )
            
            print(f"✓ {symbol} のチャートを作成しました")
            
        except Exception as e:
            print(f"✗ {symbol} のチャート作成エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== チャート生成完了 ===")


if __name__ == "__main__":

        main()