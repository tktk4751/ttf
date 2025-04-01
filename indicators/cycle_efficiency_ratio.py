#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import njit

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC


class CycleEfficiencyRatio(Indicator):
    """
    サイクル効率比(CER)インジケーター
    
    ドミナントサイクルを使用して動的なウィンドウサイズでの効率比を計算します。
    """
    
    def __init__(
        self,
        detector_type: str = 'hody',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 89,
        min_output: int = 5,
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                - 'dft': 離散フーリエ変換
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 144）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 89）
            min_output: 最小出力値（デフォルト: 5）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
        """
        super().__init__(f"CycleEfficiencyRatio({detector_type}, {cycle_part}, {max_cycle}, {min_cycle})")
        
        # パラメータ保存
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        
        # ドミナントサイクル検出器
        self.dc_detector = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            lp_period=lp_period,
            hp_period=hp_period
        )
        
        # 結果キャッシュ
        self._values = None
        self._data_hash = None
    
    def _get_data_hash(self, data) -> str:
        """
        データハッシュを取得（キャッシュ用）
        """
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合はNumpyハッシュを使用
            data_hash = hash(tuple(data.values.flatten()))
        else:
            # Numpy配列の場合は直接ハッシュを計算
            data_hash = hash(tuple(data.flatten()) if hasattr(data, 'flatten') else str(data))
        
        # パラメータハッシュを追加
        params_hash = hash((
            self.detector_type, self.lp_period, self.hp_period,
            self.cycle_part, self.max_cycle, self.min_cycle,
            self.max_output, self.min_output, self.src_type
        ))
        
        return f"{data_hash}_{params_hash}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        サイクル効率比を計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            # ハッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._values
            
            # ドミナントサイクルの計算
            dc_values = self.dc_detector.calculate(data)
            
            # 価格データを取得
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    prices = data['close'].values
                else:
                    # 適切なソースを使用
                    prices = self.dc_detector.calculate_source_values(data, self.src_type)
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    prices = data[:, 3]  # close
                else:
                    prices = data
            
            # サイクル効率比を計算
            from .efficiency_ratio import calculate_efficiency_ratio_for_period
            
            # ドミナントサイクル値の平均値を整数として計算
            # NaNや0値を除外し、最低値として8を設定
            valid_dc = dc_values[~np.isnan(dc_values) & (dc_values > 0)]
            period = int(np.mean(valid_dc)) if len(valid_dc) > 0 else 14
            period = max(8, min(period, 34))  # 8〜34の範囲に制限
            
            # 整数ピリオドで効率比を計算
            er_values = calculate_efficiency_ratio_for_period(prices, period)
            
            # 結果を保存
            self._values = er_values
            self._data_hash = data_hash
            
            return er_values
            
        except Exception as e:
            import traceback
            error_msg = f"サイクル効率比計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            return np.array([])
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"CycleEfficiencyRatio(detector_type={self.detector_type}, cycle_part={self.cycle_part}, " \
               f"max_cycle={self.max_cycle}, min_cycle={self.min_cycle})" 