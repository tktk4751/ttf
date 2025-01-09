#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any
import mplfinance as mpf
import pandas as pd
from pathlib import Path

from logger import get_logger
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from indicators.adx import ADX
from indicators.roc import ROC
from indicators.mfi import MFI
from main import Config


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
    
    def add_adx(self, period: int = 14) -> None:
        """
        ADXインジケーターを追加する
        
        Args:
            period: 期間
        """
        adx = ADX(period)
        result = adx.calculate(self.data)
        
        self.indicators.append({
            'name': adx.name,
            'panel': 2,
            'data': pd.DataFrame({
                'ADX': result.adx,
                # '+DI': result.plus_di,
                # '-DI': result.minus_di
            }, index=self.data.index),
            'colors': ['blue']
        })
    
    def add_roc(self, period: int = 12) -> None:
        """
        ROCインジケーターを追加する
        
        Args:
            period: 期間
        """
        roc = ROC(period)
        result = roc.calculate(self.data)
        
        self.indicators.append({
            'name': roc.name,
            'panel': 3,
            'data': pd.DataFrame({
                'ROC': result
            }, index=self.data.index),
            'colors': ['purple']
        })
    
    def add_mfi(self, period: int = 14) -> None:
        """
        MFIインジケーターを追加する
        
        Args:
            period: 期間
        """
        mfi = MFI(period)
        result = mfi.calculate(self.data)
        
        self.indicators.append({
            'name': mfi.name,
            'panel': 4,
            'data': pd.DataFrame({
                'MFI': result
            }, index=self.data.index),
            'colors': ['orange']
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
            'panel_ratios': (6, 2, 2,),  # メイン:出来高:ADX:ROC:MFI
            'title': title,
            'warn_too_much_data': 10000,
            'figsize': (15, 12)
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
            start_date: 開始日 (YYYY-MM-DD) - 設定ファイルの値を上書きする場合に指定
            end_date: 終了日 (YYYY-MM-DD) - 設定ファイルの値を上書きする場合に指定
        
        Returns:
            Chartインスタンス
        """
        # データの読み込み
        data_dir = config.get('data', {}).get('data_dir', 'data')
        symbol = config.get('data', {}).get('symbol', 'BTCUSDT')
        timeframe = config.get('data', {}).get('timeframe', '1h')
        
        # 期間の取得（引数で指定がない場合は設定ファイルの値を使用）
        start = start_date or config.get('data', {}).get('start')
        end = end_date or config.get('data', {}).get('end')
        
        # 日付文字列をdatetimeオブジェクトに変換
        from datetime import datetime
        start_dt = datetime.strptime(start, '%Y-%m-%d') if start else None
        end_dt = datetime.strptime(end, '%Y-%m-%d') if end else None
        
        loader = DataLoader(data_dir)
        processor = DataProcessor()
        
        data = loader.load_data(symbol, timeframe, start_dt, end_dt)
        data = processor.process(data)
        
        # チャートの作成
        chart = cls(data)
        
        # インジケーターの追加
        chart.add_adx()
        # chart.add_roc()
        # chart.add_mfi()
        
        return chart 