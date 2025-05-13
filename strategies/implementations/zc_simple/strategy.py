#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZCSimpleSignalGenerator


class ZCSimpleStrategy(BaseStrategy):
    """
    Zチャネル戦略（シンプル版）
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - Zチャネルによる高精度なエントリーポイント検出
    
    エントリー条件:
    - ロング: Zチャネルの買いシグナル
    - ショート: Zチャネルの売りシグナル
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル
    - ショート: Zチャネルの買いシグナル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        max_multiplier: float = 7.0,  # 固定乗数を使用する場合
        min_multiplier: float = 1.0,  # 固定乗数を使用する場合
        # 動的乗数の範囲パラメータ（固定乗数の代わりに動的乗数を使用する場合）
        max_max_multiplier: float = 9.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        smoother_type: str = 'alma',  # 'alma' または 'hyper'
        src_type: str = 'hlc3',       # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # CER用パラメータ
        cer_max_cycle: int = 144,       # CER用の最大サイクル期間
        cer_min_cycle: int = 5,         # CER用の最小サイクル期間
        cer_max_output: int = 89,       # CER用の最大出力値
        cer_min_output: int = 5,        # CER用の最小出力値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 100,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 120,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_lp_period: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用LPピリオド
        zma_max_dc_hp_period: int = 55,         # ZMA: 最大期間用ドミナントサイクル計算用HPピリオド
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_lp_period: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用LPピリオド
        zma_min_dc_hp_period: int = 34,         # ZMA: 最小期間用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.5,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 89,
        zma_slow_max_dc_min_output: int = 22,
        zma_slow_max_dc_lp_period: int = 5,      # ZMA: Slow最大用ドミナントサイクル計算用LPピリオド
        zma_slow_max_dc_hp_period: int = 55,     # ZMA: Slow最大用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.5,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 5,
        zma_slow_min_dc_max_output: int = 21,
        zma_slow_min_dc_min_output: int = 8,
        zma_slow_min_dc_lp_period: int = 5,      # ZMA: Slow最小用ドミナントサイクル計算用LPピリオド
        zma_slow_min_dc_hp_period: int = 34,     # ZMA: Slow最小用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.5,
        zma_fast_max_dc_max_cycle: int = 55,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 3,
        zma_fast_max_dc_lp_period: int = 5,      # ZMA: Fast最大用ドミナントサイクル計算用LPピリオド
        zma_fast_max_dc_hp_period: int = 21,     # ZMA: Fast最大用ドミナントサイクル計算用HPピリオド
        
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.7,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 77,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 35,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.5,   # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3,        # ZATR: 最小期間用ドミナントサイクル計算用
        band_lookback: int = 1
    ):

        """
        初期化
        
        Args:
            detector_type: 検出器タイプ（ZMAとZATRに使用）
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積（デフォルト）
                - 'dft': 離散フーリエ変換
            cer_detector_type: CER用の検出器タイプ（指定しない場合はdetector_typeと同じ）
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 55）
            cycle_part: サイクル部分の倍率（デフォルト: 0.7）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 6.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            # CER用パラメータ
            cer_max_cycle: CER用の最大サイクル期間（デフォルト: 144）
            cer_min_cycle: CER用の最小サイクル期間（デフォルト: 5）
            cer_max_output: CER用の最大出力値（デフォルト: 89）
            cer_min_output: CER用の最小出力値（デフォルト: 5）
            
            # ZMA用パラメータ
            zma_max_dc_cycle_part: ZMA最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_max_dc_max_cycle: ZMA最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 100）
            zma_max_dc_min_cycle: ZMA最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_max_dc_max_output: ZMA最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 120）
            zma_max_dc_min_output: ZMA最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 22）
            
            zma_min_dc_cycle_part: ZMA最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            zma_min_dc_max_cycle: ZMA最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            zma_min_dc_min_cycle: ZMA最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_min_dc_max_output: ZMA最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            zma_min_dc_min_output: ZMA最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            # ZMA動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part: ZMA動的Slow最大用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_slow_max_dc_max_cycle: ZMA動的Slow最大用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            zma_slow_max_dc_min_cycle: ZMA動的Slow最大用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_slow_max_dc_max_output: ZMA動的Slow最大用ドミナントサイクル計算用の最大出力値（デフォルト: 89）
            zma_slow_max_dc_min_output: ZMA動的Slow最大用ドミナントサイクル計算用の最小出力値（デフォルト: 22）
            
            # ZMA動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part: ZMA動的Slow最小用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_slow_min_dc_max_cycle: ZMA動的Slow最小用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 89）
            zma_slow_min_dc_min_cycle: ZMA動的Slow最小用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_slow_min_dc_max_output: ZMA動的Slow最小用ドミナントサイクル計算用の最大出力値（デフォルト: 21）
            zma_slow_min_dc_min_output: ZMA動的Slow最小用ドミナントサイクル計算用の最小出力値（デフォルト: 8）
            
            # ZMA動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part: ZMA動的Fast最大用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_fast_max_dc_max_cycle: ZMA動的Fast最大用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            zma_fast_max_dc_min_cycle: ZMA動的Fast最大用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_fast_max_dc_max_output: ZMA動的Fast最大用ドミナントサイクル計算用の最大出力値（デフォルト: 15）
            zma_fast_max_dc_min_output: ZMA動的Fast最大用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            zma_min_fast_period: ZMA速い移動平均の最小期間（デフォルト: 2）
            zma_hyper_smooth_period: ZMAハイパースムーサーの平滑化期間（デフォルト: 0）
            
            # ZATR用パラメータ
            zatr_max_dc_cycle_part: ZATR最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.7）
            zatr_max_dc_max_cycle: ZATR最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 77）
            zatr_max_dc_min_cycle: ZATR最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zatr_max_dc_max_output: ZATR最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 35）
            zatr_max_dc_min_output: ZATR最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 5）
            
            zatr_min_dc_cycle_part: ZATR最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zatr_min_dc_max_cycle: ZATR最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 34）
            zatr_min_dc_min_cycle: ZATR最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 3）
            zatr_min_dc_max_output: ZATR最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            zatr_min_dc_min_output: ZATR最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
        """
        super().__init__("ZCSimple")
        
        # パラメータの設定
        self._parameters = {
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # CER用パラメータ
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            
            # ZMA用パラメータ
            'zma_max_dc_cycle_part': zma_max_dc_cycle_part,
            'zma_max_dc_max_cycle': zma_max_dc_max_cycle,
            'zma_max_dc_min_cycle': zma_max_dc_min_cycle,
            'zma_max_dc_max_output': zma_max_dc_max_output,
            'zma_max_dc_min_output': zma_max_dc_min_output,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_slow_max_dc_cycle_part': zma_slow_max_dc_cycle_part,
            'zma_slow_max_dc_max_cycle': zma_slow_max_dc_max_cycle,
            'zma_slow_max_dc_min_cycle': zma_slow_max_dc_min_cycle,
            'zma_slow_max_dc_max_output': zma_slow_max_dc_max_output,
            'zma_slow_max_dc_min_output': zma_slow_max_dc_min_output,
            'zma_slow_min_dc_cycle_part': zma_slow_min_dc_cycle_part,
            'zma_slow_min_dc_max_cycle': zma_slow_min_dc_max_cycle,
            'zma_slow_min_dc_min_cycle': zma_slow_min_dc_min_cycle,
            'zma_slow_min_dc_max_output': zma_slow_min_dc_max_output,
            'zma_slow_min_dc_min_output': zma_slow_min_dc_min_output,
            'zma_fast_max_dc_cycle_part': zma_fast_max_dc_cycle_part,
            'zma_fast_max_dc_max_cycle': zma_fast_max_dc_max_cycle,
            'zma_fast_max_dc_min_cycle': zma_fast_max_dc_min_cycle,
            'zma_fast_max_dc_max_output': zma_fast_max_dc_max_output,
            'zma_fast_max_dc_min_output': zma_fast_max_dc_min_output,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # ZATR用パラメータ
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZCSimpleSignalGenerator(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # CER用パラメータ
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            
            # ZMA基本パラメータ
            zma_max_dc_cycle_part=zma_max_dc_cycle_part,
            zma_max_dc_max_cycle=zma_max_dc_max_cycle,
            zma_max_dc_min_cycle=zma_max_dc_min_cycle,
            zma_max_dc_max_output=zma_max_dc_max_output,
            zma_max_dc_min_output=zma_max_dc_min_output,
            
            zma_min_dc_cycle_part=zma_min_dc_cycle_part,
            zma_min_dc_max_cycle=zma_min_dc_max_cycle,
            zma_min_dc_min_cycle=zma_min_dc_min_cycle,
            zma_min_dc_max_output=zma_min_dc_max_output,
            zma_min_dc_min_output=zma_min_dc_min_output,
            
            # ZMA動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part=zma_slow_max_dc_cycle_part,
            zma_slow_max_dc_max_cycle=zma_slow_max_dc_max_cycle,
            zma_slow_max_dc_min_cycle=zma_slow_max_dc_min_cycle,
            zma_slow_max_dc_max_output=zma_slow_max_dc_max_output,
            zma_slow_max_dc_min_output=zma_slow_max_dc_min_output,
            
            # ZMA動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part=zma_slow_min_dc_cycle_part,
            zma_slow_min_dc_max_cycle=zma_slow_min_dc_max_cycle,
            zma_slow_min_dc_min_cycle=zma_slow_min_dc_min_cycle,
            zma_slow_min_dc_max_output=zma_slow_min_dc_max_output,
            zma_slow_min_dc_min_output=zma_slow_min_dc_min_output,
            
            # ZMA動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part=zma_fast_max_dc_cycle_part,
            zma_fast_max_dc_max_cycle=zma_fast_max_dc_max_cycle,
            zma_fast_max_dc_min_cycle=zma_fast_max_dc_min_cycle,
            zma_fast_max_dc_max_output=zma_fast_max_dc_max_output,
            zma_fast_max_dc_min_output=zma_fast_max_dc_min_output,
            
            # ZMA追加パラメータ
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
            # ZATR基本パラメータ
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            # 基本パラメータ
            'detector_type': trial.suggest_categorical('detector_type', ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'cer_detector_type': trial.suggest_categorical('cer_detector_type', ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'lp_period': trial.suggest_int('lp_period', 3, 21),
            'hp_period': trial.suggest_int('hp_period', 34, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.2, 0.9, step=0.1),
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper']),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 5.0, 10.0, step=0.5),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 3.0, 6.0, step=0.5),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0, step=0.1),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.0, 1.0, step=0.1),
            
            # CER用パラメータ
            'cer_max_cycle': trial.suggest_int('cer_max_cycle', 55, 233),
            'cer_min_cycle': trial.suggest_int('cer_min_cycle', 3, 34),
            'cer_max_output': trial.suggest_int('cer_max_output', 21, 144),
            'cer_min_output': trial.suggest_int('cer_min_output', 3, 13),
            
            # ZMA用パラメータ - すべて
            'zma_max_dc_cycle_part': trial.suggest_float('zma_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zma_max_dc_max_cycle': trial.suggest_int('zma_max_dc_max_cycle', 89, 233),
            'zma_max_dc_min_cycle': trial.suggest_int('zma_max_dc_min_cycle', 3, 13),
            'zma_max_dc_max_output': trial.suggest_int('zma_max_dc_max_output', 21, 144),
            'zma_max_dc_min_output': trial.suggest_int('zma_max_dc_min_output', 13, 34),
            
            'zma_min_dc_cycle_part': trial.suggest_float('zma_min_dc_cycle_part', 0.1, 0.5, step=0.1),
            'zma_min_dc_max_cycle': trial.suggest_int('zma_min_dc_max_cycle', 34, 89),
            'zma_min_dc_min_cycle': trial.suggest_int('zma_min_dc_min_cycle', 3, 8),
            'zma_min_dc_max_output': trial.suggest_int('zma_min_dc_max_output', 8, 21),
            'zma_min_dc_min_output': trial.suggest_int('zma_min_dc_min_output', 2, 5),
            
            'zma_slow_max_dc_cycle_part': trial.suggest_float('zma_slow_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zma_slow_max_dc_max_cycle': trial.suggest_int('zma_slow_max_dc_max_cycle', 89, 233),
            'zma_slow_max_dc_min_cycle': trial.suggest_int('zma_slow_max_dc_min_cycle', 3, 13),
            'zma_slow_max_dc_max_output': trial.suggest_int('zma_slow_max_dc_max_output', 55, 144),
            'zma_slow_max_dc_min_output': trial.suggest_int('zma_slow_max_dc_min_output', 13, 34),
            
            'zma_slow_min_dc_cycle_part': trial.suggest_float('zma_slow_min_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zma_slow_min_dc_max_cycle': trial.suggest_int('zma_slow_min_dc_max_cycle', 34, 144),
            'zma_slow_min_dc_min_cycle': trial.suggest_int('zma_slow_min_dc_min_cycle', 3, 13),
            'zma_slow_min_dc_max_output': trial.suggest_int('zma_slow_min_dc_max_output', 10, 89),
            'zma_slow_min_dc_min_output': trial.suggest_int('zma_slow_min_dc_min_output', 8, 55),
            
            'zma_fast_max_dc_cycle_part': trial.suggest_float('zma_fast_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zma_fast_max_dc_max_cycle': trial.suggest_int('zma_fast_max_dc_max_cycle', 34, 144),
            'zma_fast_max_dc_min_cycle': trial.suggest_int('zma_fast_max_dc_min_cycle', 3, 13),
            'zma_fast_max_dc_max_output': trial.suggest_int('zma_fast_max_dc_max_output', 5, 21),
            'zma_fast_max_dc_min_output': trial.suggest_int('zma_fast_max_dc_min_output', 3, 13),
            
            'zma_min_fast_period': trial.suggest_int('zma_min_fast_period', 1, 5),
            'zma_hyper_smooth_period': trial.suggest_int('zma_hyper_smooth_period', 0, 5),
            
            # ZATR用パラメータ - すべて
            'zatr_max_dc_cycle_part': trial.suggest_float('zatr_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zatr_max_dc_max_cycle': trial.suggest_int('zatr_max_dc_max_cycle', 34, 89),
            'zatr_max_dc_min_cycle': trial.suggest_int('zatr_max_dc_min_cycle', 3, 13),
            'zatr_max_dc_max_output': trial.suggest_int('zatr_max_dc_max_output', 34, 55),
            'zatr_max_dc_min_output': trial.suggest_int('zatr_max_dc_min_output', 3, 13),
            
            'zatr_min_dc_cycle_part': trial.suggest_float('zatr_min_dc_cycle_part', 0.1, 0.5, step=0.1),
            'zatr_min_dc_max_cycle': trial.suggest_int('zatr_min_dc_max_cycle', 21, 55),
            'zatr_min_dc_min_cycle': trial.suggest_int('zatr_min_dc_min_cycle', 2, 5),
            'zatr_min_dc_max_output': trial.suggest_int('zatr_min_dc_max_output', 8, 21),
            'zatr_min_dc_min_output': trial.suggest_int('zatr_min_dc_min_output', 2, 5)
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        strategy_params = {
            # 基本パラメータ
            'detector_type': params['detector_type'],
            'cer_detector_type': params['cer_detector_type'],
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            'smoother_type': params['smoother_type'],
            'src_type': params['src_type'],
            'band_lookback': int(params['band_lookback']),
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': float(params['max_max_multiplier']),
            'min_max_multiplier': float(params['min_max_multiplier']),
            'max_min_multiplier': float(params['max_min_multiplier']),
            'min_min_multiplier': float(params['min_min_multiplier']),
            
            # CER用パラメータ
            'cer_max_cycle': int(params['cer_max_cycle']),
            'cer_min_cycle': int(params['cer_min_cycle']),
            'cer_max_output': int(params['cer_max_output']),
            'cer_min_output': int(params['cer_min_output']),
            
            # ZMA用パラメータ - すべて
            'zma_max_dc_cycle_part': float(params['zma_max_dc_cycle_part']),
            'zma_max_dc_max_cycle': int(params['zma_max_dc_max_cycle']),
            'zma_max_dc_min_cycle': int(params['zma_max_dc_min_cycle']),
            'zma_max_dc_max_output': int(params['zma_max_dc_max_output']),
            'zma_max_dc_min_output': int(params['zma_max_dc_min_output']),
            
            'zma_min_dc_cycle_part': float(params['zma_min_dc_cycle_part']),
            'zma_min_dc_max_cycle': int(params['zma_min_dc_max_cycle']),
            'zma_min_dc_min_cycle': int(params['zma_min_dc_min_cycle']),
            'zma_min_dc_max_output': int(params['zma_min_dc_max_output']),
            'zma_min_dc_min_output': int(params['zma_min_dc_min_output']),
            
            'zma_slow_max_dc_cycle_part': float(params['zma_slow_max_dc_cycle_part']),
            'zma_slow_max_dc_max_cycle': int(params['zma_slow_max_dc_max_cycle']),
            'zma_slow_max_dc_min_cycle': int(params['zma_slow_max_dc_min_cycle']),
            'zma_slow_max_dc_max_output': int(params['zma_slow_max_dc_max_output']),
            'zma_slow_max_dc_min_output': int(params['zma_slow_max_dc_min_output']),
            
            'zma_slow_min_dc_cycle_part': float(params['zma_slow_min_dc_cycle_part']),
            'zma_slow_min_dc_max_cycle': int(params['zma_slow_min_dc_max_cycle']),
            'zma_slow_min_dc_min_cycle': int(params['zma_slow_min_dc_min_cycle']),
            'zma_slow_min_dc_max_output': int(params['zma_slow_min_dc_max_output']),
            'zma_slow_min_dc_min_output': int(params['zma_slow_min_dc_min_output']),
            
            'zma_fast_max_dc_cycle_part': float(params['zma_fast_max_dc_cycle_part']),
            'zma_fast_max_dc_max_cycle': int(params['zma_fast_max_dc_max_cycle']),
            'zma_fast_max_dc_min_cycle': int(params['zma_fast_max_dc_min_cycle']),
            'zma_fast_max_dc_max_output': int(params['zma_fast_max_dc_max_output']),
            'zma_fast_max_dc_min_output': int(params['zma_fast_max_dc_min_output']),
            
            'zma_min_fast_period': int(params['zma_min_fast_period']),
            'zma_hyper_smooth_period': int(params['zma_hyper_smooth_period']),
            
            # ZATR用パラメータ - すべて
            'zatr_max_dc_cycle_part': float(params['zatr_max_dc_cycle_part']),
            'zatr_max_dc_max_cycle': int(params['zatr_max_dc_max_cycle']),
            'zatr_max_dc_min_cycle': int(params['zatr_max_dc_min_cycle']),
            'zatr_max_dc_max_output': int(params['zatr_max_dc_max_output']),
            'zatr_max_dc_min_output': int(params['zatr_max_dc_min_output']),
            
            'zatr_min_dc_cycle_part': float(params['zatr_min_dc_cycle_part']),
            'zatr_min_dc_max_cycle': int(params['zatr_min_dc_max_cycle']),
            'zatr_min_dc_min_cycle': int(params['zatr_min_dc_min_cycle']),
            'zatr_min_dc_max_output': int(params['zatr_min_dc_max_output']),
            'zatr_min_dc_min_output': int(params['zatr_min_dc_min_output'])
        }
        return strategy_params 