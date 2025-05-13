#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, Type
import numpy as np
import pandas as pd

# from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult # 相対インポートをコメントアウト
from indicators.ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult # 絶対インポートに変更
# from .indicators.ehlers_hody_dc import EhlersHoDyDC # 不正なパス
from indicators.ehlers_hody_dc import EhlersHoDyDC # 絶対インポートに変更
# from .indicators.ehlers_phac_dc import EhlersPhAcDC # 不正なパス
from indicators.ehlers_phac_dc import EhlersPhAcDC # 絶対インポートに変更
# from .indicators.ehlers_dudi_dc import EhlersDuDiDC # 不正なパス
from indicators.ehlers_dudi_dc import EhlersDuDiDC # 絶対インポートに変更
# from .indicators.ehlers_dudi_dce import EhlersDuDiDCE # 不正なパス
from indicators.ehlers_dudi_dce import EhlersDuDiDCE # 絶対インポートに変更
# from .indicators.ehlers_hody_dce import EhlersHoDyDCE # 不正なパス
from indicators.ehlers_hody_dce import EhlersHoDyDCE # 絶対インポートに変更
# from .indicators.ehlers_phac_dce import EhlersPhAcDCE # 不正なパス
from indicators.ehlers_phac_dce import EhlersPhAcDCE # 絶対インポートに変更
# from .indicators.kalman_filter import KalmanFilter # 不正なパス
from indicators.kalman_filter import KalmanFilter # 絶対インポートに変更
# from .indicators.price_source import PriceSource # 不正なパス
from indicators.price_source import PriceSource # 絶対インポートに変更


class EhlersUnifiedDC(EhlersDominantCycle):
    """
    改良版 エーラーズ統合サイクル検出器
    
    このクラスは複数のエーラーズサイクル検出アルゴリズムを統合し、
    単一のインターフェースで利用可能にします。
    
    特徴:
    - 複数のサイクル検出アルゴリズムを選択可能
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)
    - オプションで価格ソースにカルマンフィルターを適用可能
    
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
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
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
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_measurement_noise: カルマンフィルターの測定ノイズ
            kalman_process_noise: カルマンフィルターのプロセスノイズ
            kalman_n_states: カルマンフィルターの状態数
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
        name = f"EhlersUnifiedDC(det={detector_type}, src={src_type}, kalman={'Y' if use_kalman_filter else 'N'})"
        super().__init__(
            name,
            cycle_part,
            max_cycle,
            min_cycle,
            max_output,
            min_output
        )
        
        # 検出器タイプとパラメータを保存
        self.detector_type = detector_type
        self.src_type = src_type
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # PriceSourceユーティリティ
        self.price_source_extractor = PriceSource()
        
        # オプションのカルマンフィルター
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = KalmanFilter(
                price_source=self.src_type,
                measurement_noise=self.kalman_measurement_noise,
                process_noise=self.kalman_process_noise,
                n_states=self.kalman_n_states
            )
        
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
                DataFrameの場合、src_type および (オプションで) HLC カラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            original_data = data # 元データを保持
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # --- Optional Kalman Filtering ---
            calculation_input = original_data
            if self.use_kalman_filter and self.kalman_filter:
                # カルマンフィルターを適用
                filtered_source_prices = self.kalman_filter.calculate(original_data)
                if filtered_source_prices is not None and len(filtered_source_prices) > 0:
                    # 下位の検出器は通常、フィルタリングされた価格のNumPy配列を直接扱えるはず
                    # (もしDataFrameが必要な場合は、ここで一時的なDataFrameを作成する)
                    calculation_input = filtered_source_prices
                else:
                    self.logger.warning("カルマンフィルターの計算に失敗したか、空の結果が返されました。元のデータを使用します。")
            
            # --- Run Calculation ---
            # 選択された検出器で計算を実行 (フィルタリングされた、または元のデータを使用)
            dom_cycle = self.detector.calculate(calculation_input)
            
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
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(original_data) if hasattr(original_data, '__len__') else 0
            self._values = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュ無効化
            return self._values
    
    @classmethod
    def get_available_detectors(cls) -> Dict[str, str]:
        """
        利用可能な検出器とその説明を返す
        
        Returns:
            Dict[str, str]: 検出器名とその説明の辞書
        """
        return cls._DETECTOR_DESCRIPTIONS
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._data_hash = None
        if hasattr(self.detector, 'reset'):
            self.detector.reset()
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        if self.src_type == 'open':
            required_cols.add('open')
        elif self.src_type == 'high':
            required_cols.add('high')
        elif self.src_type == 'low':
            required_cols.add('low')
        elif self.src_type == 'close':
            required_cols.add('close')
        elif self.src_type == 'hl2':
            required_cols.update(['high', 'low'])
        elif self.src_type == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        elif self.src_type == 'hlcc4':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'weighted_close':
            required_cols.update(['high', 'low', 'close'])
        else:
            # 不明なsrc_typeの場合はデフォルトでcloseを使用
            required_cols.add('close')

        # EhlersのDC計算は内部でフィルタリング等を行うため、
        # 安全のためにOHLCを要求する場合がある (detectorによる)
        # ここでは、src_typeに必要なものだけをハッシュ対象とする

        if isinstance(data, pd.DataFrame):
            relevant_cols = [col for col in data.columns if col.lower() in required_cols]
            # Check if all *required* columns are present, warn if not?
            # For hashing, only use present relevant columns
            present_cols = [col for col in relevant_cols if col in data.columns]
            if len(present_cols) < len(required_cols):
                 # Log a warning maybe, but proceed with hash of available data
                 pass
            if not present_cols:
                 # If no relevant columns found, hash the whole DataFrame shape? Or raise error?
                 # Let's hash based on shape and first/last row as fallback
                 try:
                     shape_tuple = data.shape
                     first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                     last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                     data_repr_tuple = (shape_tuple, first_row, last_row)
                     data_hash_val = hash(data_repr_tuple)
                 except Exception:
                     data_hash_val = hash(str(data))
            else:
                 data_values = data[present_cols].values # Get only relevant columns
                 data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            # Determine column index based on src_type (assuming OHLC(V) order)
            col_indices = []
            if 'open' in required_cols: col_indices.append(0)
            if 'high' in required_cols: col_indices.append(1)
            if 'low' in required_cols: col_indices.append(2)
            if 'close' in required_cols: col_indices.append(3)
            # Add more mappings for hl2, hlc3, ohlc4 if needed, though they are harder with numpy

            if data.ndim == 2 and data.shape[1] > max(col_indices if col_indices else [-1]):
                data_values = data[:, col_indices]
                data_hash_val = hash(data_values.tobytes())
            else:
                data_hash_val = hash(data.tobytes()) # Fallback
        else:
            data_hash_val = hash(str(data))

        # Include relevant parameters
        param_str = (
            f"det={self.detector_type}_src={self.src_type}_"
            f"kalman={self.use_kalman_filter}_{self.kalman_measurement_noise}_{self.kalman_process_noise}_{self.kalman_n_states}_"
            f"cycPart={self.cycle_part}_maxC={self.max_cycle}_minC={self.min_cycle}_"
            f"maxOut={self.max_output}_minOut={self.min_output}_"
            f"lp={self.lp_period}_hp={self.hp_period}"
            # Add other specific detector params if they vary significantly and affect output
        )
        return f"{data_hash_val}_{param_str}" 