#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.z_bb.breakout_entry import ZBBBreakoutEntrySignal


class ZBBSimpleSignalGenerator(BaseSignalGenerator):
    """
    Zボリンジャーバンドのシグナル生成クラス（トレンドフィルターなし）
    
    特徴:
    - ZBBBreakoutEntrySignalのみを使用したシンプルな戦略
    - Numba最適化による高速計算
    
    シグナル条件:
    - ロングエントリー: ZBBBreakoutがロングシグナル(1)
    - ショートエントリー: ZBBBreakoutがショートシグナル(-1)
    - ロング決済: ZBBBreakoutがショートシグナル(-1)
    - ショート決済: ZBBBreakoutがロングシグナル(1)
    """
    
    def __init__(
        self,
        # ZBBBreakoutEntrySignalのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 2.5,
        min_multiplier: float = 1.0,
        max_cycle_part: float = 0.5,
        max_max_cycle: int = 144,
        max_min_cycle: int = 10,
        max_max_output: int = 89,
        max_min_output: int = 13,
        min_cycle_part: float = 0.25,
        min_max_cycle: int = 55,
        min_min_cycle: int = 5,
        min_max_output: int = 21,
        min_min_output: int = 5,
        src_type: str = 'hlc3',
        lookback: int = 1,
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: ZBBのサイクル検出器の種類
            lp_period: ZBBのローパスフィルター期間
            hp_period: ZBBのハイパスフィルター期間
            cycle_part: ZBBのサイクル部分倍率
            max_multiplier: ZBBの最大標準偏差乗数
            min_multiplier: ZBBの最小標準偏差乗数
            max_cycle_part: ZBBの最大標準偏差サイクル部分
            max_max_cycle: ZBBの最大標準偏差最大サイクル
            max_min_cycle: ZBBの最大標準偏差最小サイクル
            max_max_output: ZBBの最大標準偏差最大出力
            max_min_output: ZBBの最大標準偏差最小出力
            min_cycle_part: ZBBの最小標準偏差サイクル部分
            min_max_cycle: ZBBの最小標準偏差最大サイクル
            min_min_cycle: ZBBの最小標準偏差最小サイクル
            min_max_output: ZBBの最小標準偏差最大出力
            min_min_output: ZBBの最小標準偏差最小出力
            src_type: ZBBの価格ソースタイプ
            lookback: ZBBのルックバック期間
        """
        super().__init__("ZBBSimpleSignalGenerator")
        
        # ZBBBreakoutEntrySignalの初期化
        self.zbb_breakout = ZBBBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            max_cycle_part=max_cycle_part,
            max_max_cycle=max_max_cycle,
            max_min_cycle=max_min_cycle,
            max_max_output=max_max_output,
            max_min_output=max_min_output,
            min_cycle_part=min_cycle_part,
            min_max_cycle=min_max_cycle,
            min_min_cycle=min_min_cycle,
            min_max_output=min_max_output,
            min_min_output=min_min_output,
            src_type=src_type,
            lookback=lookback
        )
        
        # ルックバック期間の保存
        self.lookback = lookback
        
        # キャッシュ用変数
        self._entry_signals = None
        self._data_len = 0
        self._data_hash = None
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> Optional[str]:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合は必要なカラムのみハッシュする
                cols = ['open', 'high', 'low', 'close']
                return str(hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns)))))
            else:
                # NumPy配列の場合は全体をハッシュする
                return str(hash(tuple(map(tuple, data))))
        except Exception:
            return None
        
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
        
        Returns:
            np.ndarray: エントリーシグナル (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # データのサイズ変更を検出
            data_len = len(data)
            data_hash = self._get_data_hash(data)
            
            # キャッシュが有効な場合はキャッシュを返す
            if data_hash == self._data_hash and data_len == self._data_len and self._entry_signals is not None:
                return self._entry_signals
            
            # キャッシュの更新
            self._data_len = data_len
            self._data_hash = data_hash
            
            # ZBBのブレイクアウトシグナルを取得（トレンドフィルターなし）
            breakout_signals = self.zbb_breakout.generate(data)
            
            # 結果をキャッシュして返す
            self._entry_signals = breakout_signals
            return self._entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            import traceback
            self.logger.error(f"ZBBSimpleSignalGenerator計算中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return np.zeros(len(data), dtype=np.int8)
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを取得する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべき場合はTrue
        """
        try:
            # エントリーシグナルを事前に計算（キャッシュも行う）
            if self._entry_signals is None or len(self._entry_signals) != len(data):
                self.get_entry_signals(data)
                
            # ZBBのブレイクアウトシグナルを取得
            breakout_signals = self.zbb_breakout.generate(data)
            
            # 指定されたインデックスのブレイクアウトシグナルを取得
            current_signal = breakout_signals[index]
            
            # ポジションに基づいてエグジット条件を判定
            if position == 1:  # ロングポジション
                # ショートシグナルが出た場合にエグジット
                return current_signal == -1
            elif position == -1:  # ショートポジション
                # ロングシグナルが出た場合にエグジット
                return current_signal == 1
            
            return False
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、エグジットしないと判断
            import traceback
            self.logger.error(f"ZBBSimpleSignalGenerator エグジット計算中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        Zボリンジャーバンドのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.zbb_breakout.generate(data)
            
        return self.zbb_breakout.get_bands()
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            self.zbb_breakout.generate(data)
            
        return self.zbb_breakout.get_cycle_er()
    
    def reset(self) -> None:
        """
        シグナル生成器の状態をリセットする
        """
        self.zbb_breakout.reset() if hasattr(self.zbb_breakout, 'reset') else None
        self._entry_signals = None
        self._data_len = 0
        self._data_hash = None 