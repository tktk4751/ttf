#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any
import mplfinance as mpf
import pandas as pd
from pathlib import Path

from logger import get_logger
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from indicators.stochastic import Stochastic
from indicators.stochastic_rsi import StochasticRSI


class Chart:
    """
    チャート表示クラス
    ローソク足チャートとインジケーターを表示する
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        コンストラクタ
        
        Args:
            data: チャートデータ
        """
        self.logger = get_logger()
        self.data = data
        self.indicators: List[Dict[str, Any]] = []
        
        # データの検証
        self._validate_data()
    
    def _validate_data(self) -> None:
        """データの形式を検証する"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns 
                         if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(
                f"必要なカラムが不足しています: {', '.join(missing_columns)}"
            )
    
    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> None:
        """
        Stochasticインジケーターを追加する
        
        Args:
            k_period: %K期間
            d_period: %D期間
        """
        stoch = Stochastic(k_period, d_period)
        result = stoch.calculate(self.data)
        
        self.indicators.append({
            'name': stoch.name,
            'panel': 2,
            'data': pd.DataFrame({
                '%K': result.k,
                '%D': result.d
            }, index=self.data.index),
            'colors': ['blue', 'red']
        })
    
    def add_stochastic_rsi(
        self,
        period: int = 14,
        k_period: int = 3,
        d_period: int = 3
    ) -> None:
        """
        Stochastic RSIインジケーターを追加する
        
        Args:
            period: RSI期間
            k_period: %K期間
            d_period: %D期間
        """
        stoch_rsi = StochasticRSI(period, k_period, d_period)
        result = stoch_rsi.calculate(self.data)
        
        self.indicators.append({
            'name': stoch_rsi.name,
            'panel': 3,
            'data': pd.DataFrame({
                '%K': result.k,
                '%D': result.d
            }, index=self.data.index),
            'colors': ['purple', 'orange']
        })
    
    def show(
        self,
        title: Optional[str] = None,
        volume: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        チャートを表示する
        
        Args:
            title: チャートのタイトル
            volume: 出来高を表示するかどうか
            save_path: 保存先のパス
        """
        # スタイルの設定
        style = mpf.make_mpf_style(
            base_mpf_style='charles',
            gridstyle='',
            y_on_right=False
        )
        
        # プロットの設定
        kwargs = {
            'type': 'candle',
            'style': style,
            'volume': volume,
            'panel_ratios': (6, 2, 2, 2),  # メイン:出来高:Stoch:StochRSI
            'title': title,
            'warn_too_much_data': 10000,
            'figsize': (15, 10)
        }
        
        # インジケーターの追加
        if self.indicators:
            addplots = []
            for ind in self.indicators:
                for col, color in zip(ind['data'].columns, ind['colors']):
                    addplots.append(
                        mpf.make_addplot(
                            ind['data'][col],
                            panel=ind['panel'],
                            color=color,
                            secondary_y=False,
                            ylabel=ind['name']
                        )
                    )
            kwargs['addplot'] = addplots
        
        # チャートの表示/保存
        if save_path:
            kwargs['savefig'] = save_path
            self.logger.info(f"チャートを保存しました: {save_path}")
        
        mpf.plot(self.data, **kwargs)
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> 'Chart':
        """
        設定からチャートを作成する
        
        Args:
            config: 設定辞書
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
        
        Returns:
            Chartインスタンス
        """
        # データの読み込み
        data_dir = config.get('data', {}).get('data_dir', 'data')
        symbol = config.get('data', {}).get('symbol', 'BTCUSDT')
        timeframe = config.get('data', {}).get('timeframe', '1h')
        
        loader = DataLoader(data_dir)
        processor = DataProcessor()
        
        data = loader.load_data(symbol, timeframe)
        data = processor.process(data)
        
        # チャートの作成
        chart = cls(data)
        
        # インジケーターの追加
        chart.add_stochastic()
        chart.add_stochastic_rsi()
        
        return chart 