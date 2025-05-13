#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import njit

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC
from .efficiency_ratio import calculate_efficiency_ratio_for_period # efficiency_ratio をインポート


class CycleVolatilityER(Indicator): # クラス名を変更
    """
    サイクルボラティリティ効率比 (CVER) インジケーター
    
    ドミナントサイクルを使用して動的なウィンドウサイズでの
    サイクル平均真の範囲 (CATR) の効率比を計算します。
    価格ではなくボラティリティの変動効率を測定します。
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
        src_type: str = 'hlc3' # ドミナントサイクル計算用のソースタイプ
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
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
            src_type: ドミナントサイクル計算用のソースタイプ（デフォルト: 'hlc3'）
                     効率比の計算にはCATR値が使用されます。
        """
        # クラス名に合わせて super().__init__ の引数を更新
        super().__init__(f"CycleVolatilityER({detector_type}, {cycle_part}, {max_cycle}, {min_cycle})")
        
        # パラメータ保存
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type # ドミナントサイクル計算用に保持
        
        # ドミナントサイクル検出器 (価格データからサイクルを検出するために使用)
        self.dc_detector = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type, # 価格データからサイクルを検出
            lp_period=lp_period,
            hp_period=hp_period
        )
        
        # 結果キャッシュ
        self._values = None
        self._data_hash = None
    
    def _get_data_hash(self, price_data, catr_values) -> str: # 引数に catr_values を追加
        """
        データハッシュを取得（キャッシュ用）
        価格データとCATRデータの両方を考慮します。
        """
        # 価格データのハッシュ
        if isinstance(price_data, pd.DataFrame):
            price_hash = hash(tuple(price_data.values.flatten()))
        else:
            price_hash = hash(tuple(price_data.flatten()) if hasattr(price_data, 'flatten') else str(price_data))
            
        # CATRデータのハッシュ
        catr_hash = hash(tuple(catr_values.flatten()) if hasattr(catr_values, 'flatten') else str(catr_values))
        
        # パラメータハッシュを追加
        params_hash = hash((
            self.detector_type, self.lp_period, self.hp_period,
            self.cycle_part, self.max_cycle, self.min_cycle,
            self.max_output, self.min_output, self.src_type
        ))
        
        # 2つのデータハッシュとパラメータハッシュを結合
        return f"{price_hash}_{catr_hash}_{params_hash}"
    
    def calculate(self, price_data: Union[pd.DataFrame, np.ndarray], catr_values: np.ndarray) -> np.ndarray: # 引数を変更
        """
        サイクルボラティリティ効率比を計算
        
        Args:
            price_data: OHLC価格データ（DataFrameまたはNumpy配列）。ドミナントサイクル計算用。
            catr_values: CATRの絶対値（金額ベース）の配列。効率比計算用。
        
        Returns:
            np.ndarray: サイクルボラティリティ効率比の値
        """
        try:
            # 入力データの長さをチェック
            if isinstance(price_data, pd.DataFrame):
                price_len = len(price_data)
            else:
                price_len = len(price_data)
                
            if len(catr_values) != price_len:
                 raise ValueError(f"価格データ長 ({price_len}) と CATRデータ長 ({len(catr_values)}) が一致しません。")

            # ハッシュチェック
            data_hash = self._get_data_hash(price_data, catr_values) # ハッシュ計算の引数を変更
            if data_hash == self._data_hash and self._values is not None:
                return self._values
            
            # ドミナントサイクルの計算 (価格データを使用)
            dc_values = self.dc_detector.calculate(price_data)
            
            # 効率比計算のソースとして CATR 値を使用
            source_values = np.asarray(catr_values, dtype=np.float64) # CATR値をソースとする
            
            # サイクル効率比を計算
            # from .efficiency_ratio import calculate_efficiency_ratio_for_period # クラス冒頭でインポート済
            
            # ドミナントサイクル値の平均値を整数として計算
            # NaNや0値を除外し、最低値として8を設定
            valid_dc = dc_values[~np.isnan(dc_values) & (dc_values > 0)]
            period = int(np.mean(valid_dc)) if len(valid_dc) > 0 else 14
            period = max(8, min(period, 34))  # 8〜34の範囲に制限
            
            # 整数ピリオドで効率比を計算 (ソースを catr_values に変更)
            er_values = calculate_efficiency_ratio_for_period(source_values, period)
            
            # 結果を保存
            self._values = er_values
            self._data_hash = data_hash
            
            return er_values
            
        except Exception as e:
            import traceback
            error_msg = f"サイクルボラティリティ効率比計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            return np.array([]) # エラー時は空配列を返す
    
    def __str__(self) -> str:
        """文字列表現"""
        # クラス名に合わせて更新
        return f"CycleVolatilityER(detector_type={self.detector_type}, cycle_part={self.cycle_part}, " \
               f"max_cycle={self.max_cycle}, min_cycle={self.min_cycle})" 