#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, Type
import numpy as np
import pandas as pd

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from .ehlers_hody_dc import EhlersHoDyDC
from .ehlers_phac_dc import EhlersPhAcDC
from .ehlers_dudi_dc import EhlersDuDiDC
from .ehlers_dudi_dce import EhlersDuDiDCE
from .ehlers_hody_dce import EhlersHoDyDCE
from .ehlers_phac_dce import EhlersPhAcDCE


class EhlersUnifiedDC(EhlersDominantCycle):
    """
    エーラーズのサイクル検出器を統合したインジケーター
    
    このクラスは複数のエーラーズサイクル検出アルゴリズムを統合し、
    単一のインターフェースで利用可能にします。
    
    対応検出器:
    - 'hody': ホモダイン判別機 (Homodyne Discriminator)
    - 'phac': 位相累積 (Phase Accumulation)
    - 'dudi': 二重微分 (Dual Differentiator)
    - 'dudi_e': 拡張二重微分 (Enhanced Dual Differentiator)
    - 'hody_e': 拡張ホモダイン判別機 (Enhanced Homodyne Discriminator)
    - 'phac_e': 拡張位相累積 (Enhanced Phase Accumulation)
    """
    
    # 利用可能な検出器の定義
    _DETECTORS = {
        'hody': EhlersHoDyDC,
        'phac': EhlersPhAcDC,
        'dudi': EhlersDuDiDC,
        'dudi_e': EhlersDuDiDCE,
        'hody_e': EhlersHoDyDCE,
        'phac_e': EhlersPhAcDCE
    }
    
    # 検出器の説明
    _DETECTOR_DESCRIPTIONS = {
        'hody': 'ホモダイン判別機（Homodyne Discriminator）',
        'phac': '位相累積（Phase Accumulation）',
        'dudi': '二重微分（Dual Differentiator）',
        'dudi_e': '拡張二重微分（Enhanced Dual Differentiator）',
        'hody_e': '拡張ホモダイン判別機（Enhanced Homodyne Discriminator）',
        'phac_e': '拡張位相累積（Enhanced Phase Accumulation）'
    }
    
    def __init__(
        self,
        detector_type: str = 'hody',
        cycle_part: float = 0.5,
        max_cycle: int = 50,
        min_cycle: int = 6,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close',
        # 拡張検出器用のパラメータ
        lp_period: int = 5,
        hp_period: int = 55
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 使用する検出器のタイプ
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 50）
            min_cycle: 最小サイクル期間（デフォルト: 6）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            lp_period: ローパスフィルターの期間（拡張検出器用）
            hp_period: ハイパスフィルターの期間（拡張検出器用）
        """
        # 検出器名を小文字に変換して正規化
        detector_type = detector_type.lower()
        
        # 検出器が有効かチェック
        if detector_type not in self._DETECTORS:
            valid_detectors = ", ".join(self._DETECTORS.keys())
            raise ValueError(f"無効な検出器タイプです: {detector_type}。有効なオプション: {valid_detectors}")
        
        # 親クラスの初期化
        super().__init__(
            f"EhlersUnifiedDC({detector_type})",
            cycle_part,
            max_cycle,
            min_cycle,
            max_output,
            min_output
        )
        
        # 検出器タイプとパラメータを保存
        self.detector_type = detector_type
        self.src_type = src_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # 検出器の初期化
        if detector_type in ['dudi_e', 'hody_e', 'phac_e']:
            # 拡張検出器はローパスとハイパスのパラメータが必要
            self.detector = self._DETECTORS[detector_type](
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type
            )
        else:
            # 標準検出器
            self.detector = self._DETECTORS[detector_type](
                cycle_part=cycle_part,
                max_cycle=max_cycle,
                min_cycle=min_cycle,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type
            )
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        指定された検出器を使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、指定したソースタイプに必要なカラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # 選択された検出器で計算を実行
            dom_cycle = self.detector.calculate(data)
            
            # 結果を直接設定（get_result()を使用しない）
            from .ehlers_dominant_cycle import DominantCycleResult
            # raw_periodとsmooth_periodの設定（ここでは同じ値を使用）
            length = len(dom_cycle)
            raw_period = np.full(length, self.max_cycle)
            smooth_period = np.full(length, self.max_cycle)
            self._result = DominantCycleResult(dom_cycle, raw_period, smooth_period)
            self._values = dom_cycle
            
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersUnifiedDC計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    @classmethod
    def get_available_detectors(cls) -> Dict[str, str]:
        """
        利用可能な検出器とその説明を返す
        
        Returns:
            Dict[str, str]: 検出器名とその説明の辞書
        """
        return cls._DETECTOR_DESCRIPTIONS 