#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math

try:
    from ..indicator import Indicator
    from .source_calculator import calculate_source_simple
    # EhlersUnifiedDC は関数内でインポートして循環インポートを回避
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from source_calculator import calculate_source_simple
    # EhlersUnifiedDC は関数内でインポートして循環インポートを回避


@dataclass
class UltimateSmootherResult:
    """アルティメットスムーザーの計算結果"""
    values: np.ndarray                # アルティメットスムーザー値
    coefficients: np.ndarray          # 使用された係数（デバッグ用）


@njit(fastmath=True, cache=True)
def calculate_ultimate_smoother(
    price: np.ndarray,
    period: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    アルティメットスムーザーを計算する（Numba最適化版）
    
    Based on John Ehlers' "The Ultimate Smoother" paper
    
    Args:
        price: 価格配列
        period: 臨界期間（デフォルト: 20.0）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        (UltimateSmoother値, 係数配列)
    """
    length = len(price)
    
    # 係数の計算（Ehlers' formula）
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1.0 + c2 - c3) / 4.0
    
    # 結果配列の初期化
    ultimate_smoother = np.zeros(length, dtype=np.float64)
    coefficients = np.full(length, c1, dtype=np.float64)  # 係数配列（デバッグ用）
    
    # 最初の3つの値は初期値として設定
    for i in range(min(3, length)):
        ultimate_smoother[i] = price[i] if i < length else 0.0
    
    # CurrentBar >= 4の条件（インデックス3から開始、0ベースなので）
    for i in range(3, length):
        # Ultimate Smoother計算（Ehlers' formula）
        # US = (1 - c1)*Price + (2*c1 - c2)*Price[1] - (c1 + c3)*Price[2] + c2*US[1] + c3*US[2]
        if i >= 2:
            ultimate_smoother[i] = ((1.0 - c1) * price[i] + 
                                   (2.0 * c1 - c2) * price[i-1] - 
                                   (c1 + c3) * price[i-2] + 
                                   c2 * ultimate_smoother[i-1] + 
                                   c3 * ultimate_smoother[i-2])
        else:
            ultimate_smoother[i] = price[i]
    
    return ultimate_smoother, coefficients


@njit(fastmath=True, cache=True)
def calculate_ultimate_smoother_adaptive(
    price: np.ndarray,
    periods: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的適応アルティメットスムーザーを計算する（Numba最適化版）
    
    Based on John Ehlers' "The Ultimate Smoother" paper with adaptive periods
    
    Args:
        price: 価格配列
        periods: 各データポイントに対応する動的期間配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        (UltimateSmoother値, 係数配列)
    """
    length = len(price)
    
    # 結果配列の初期化
    ultimate_smoother = np.zeros(length, dtype=np.float64)
    coefficients = np.zeros(length, dtype=np.float64)
    
    # 最初の3つの値は初期値として設定
    for i in range(min(3, length)):
        ultimate_smoother[i] = price[i] if i < length else 0.0
        if i < len(periods):
            # 初期期間での係数計算
            period = max(2.0, periods[i])  # 最小期間2
            a1 = math.exp(-1.414 * math.pi / period)
            b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
            c2 = b1
            c3 = -a1 * a1
            c1 = (1.0 + c2 - c3) / 4.0
            coefficients[i] = c1
    
    # CurrentBar >= 4の条件（インデックス3から開始、0ベースなので）
    for i in range(3, length):
        if i < len(periods):
            # 動的期間から係数を計算
            period = max(2.0, periods[i])  # 最小期間2
            a1 = math.exp(-1.414 * math.pi / period)
            b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
            c2 = b1
            c3 = -a1 * a1
            c1 = (1.0 + c2 - c3) / 4.0
            coefficients[i] = c1
            
            # Ultimate Smoother計算（動的係数版）
            if i >= 2:
                ultimate_smoother[i] = ((1.0 - c1) * price[i] + 
                                       (2.0 * c1 - c2) * price[i-1] - 
                                       (c1 + c3) * price[i-2] + 
                                       c2 * ultimate_smoother[i-1] + 
                                       c3 * ultimate_smoother[i-2])
            else:
                ultimate_smoother[i] = price[i]
        else:
            # 期間データがない場合は前の値をコピー
            ultimate_smoother[i] = ultimate_smoother[i-1] if i > 0 else price[i]
            coefficients[i] = coefficients[i-1] if i > 0 else 0.0
    
    return ultimate_smoother, coefficients


class UltimateSmoother(Indicator):
    """
    アルティメットスムーザーインジケーター（John Ehlers）- 動的適応対応版
    
    パスバンドでゼロラグを持つ適応型スムージングフィルター：
    - ハイパスフィルターの応答を入力データから差し引いて構築
    - 低周波数コンポーネントを通し、高周波数コンポーネントを減衰
    - 従来のEMAやSuperSmootherより優れた応答性
    - 動的適応期間対応（EhlersUnifiedDCによるサイクル検出）
    
    特徴:
    - パスバンドでゼロラグ
    - 第2次IIRフィルター（無限インパルス応答）
    - 臨界期間による調整可能（固定または動的）
    - Ehlersサイクル検出による期間自動調整
    """
    
    def __init__(
        self,
        period: float = 20.0,                  # 臨界期間
        src_type: str = 'hlc3',               # ソースタイプ
        ukf_params: Optional[Dict] = None,     # UKFパラメータ（UKFソース使用時）
        # 動的適応パラメータ
        period_mode: str = 'fixed',            # 期間モード ('fixed' or 'dynamic')
        # サイクル検出器パラメータ
        cycle_detector_type: str = 'absolute_ultimate',
        cycle_detector_cycle_part: float = 0.5,
        cycle_detector_max_cycle: int = 120,
        cycle_detector_min_cycle: int = 5,
        cycle_period_multiplier: float = 1.0,
        cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """
        コンストラクタ
        
        Args:
            period: 臨界期間（デフォルト: 20.0）
                - パスバンドとストップバンドを分離する波長
                - 長い値：よりスムージング、より大きなラグ
                - 短い値：より敏感、より小さなラグ
            src_type: ソースタイプ
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
                alpha: UKFのalpha値（デフォルト: 0.001）
                beta: UKFのbeta値（デフォルト: 2.0）
                kappa: UKFのkappa値（デフォルト: 0.0）
                process_noise_scale: プロセスノイズスケール（デフォルト: 0.001）
                volatility_window: ボラティリティ計算ウィンドウ（デフォルト: 10）
                adaptive_noise: 適応ノイズの使用（デフォルト: True）
            
            # 動的適応パラメータ
            period_mode: 期間モード ('fixed' or 'dynamic')
            
            # サイクル検出器パラメータ
            cycle_detector_type: サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            cycle_detector_cycle_part: サイクル検出器のサイクル部分倍率（デフォルト: 1.0）
            cycle_detector_max_cycle: サイクル検出器の最大サイクル期間（デフォルト: 120）
            cycle_detector_min_cycle: サイクル検出器の最小サイクル期間（デフォルト: 5）
            cycle_period_multiplier: サイクル期間の乗数（デフォルト: 1.0）
            cycle_detector_period_range: サイクル検出器の周期範囲（デフォルト: (5, 120)）
        """
        # 指標名の作成
        indicator_name = f"UltimateSmoother(period={period}({period_mode}), {src_type}, cycle={cycle_detector_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.period = period
        self.src_type = src_type.lower()
        self.ukf_params = ukf_params
        
        # 動的適応パラメータ
        self.period_mode = period_mode.lower()
        
        # サイクル検出器パラメータ
        self.cycle_detector_type = cycle_detector_type
        self.cycle_detector_cycle_part = cycle_detector_cycle_part
        self.cycle_detector_max_cycle = cycle_detector_max_cycle
        self.cycle_detector_min_cycle = cycle_detector_min_cycle
        self.cycle_period_multiplier = cycle_period_multiplier
        self.cycle_detector_period_range = cycle_detector_period_range
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.period < 2:
            raise ValueError("periodは2以上である必要があります（フィルター安定性のため）")
        
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効なperiod_mode: {self.period_mode}. 'fixed' または 'dynamic' を指定してください。")
        
        # ソースタイプの検証
        valid_sources = ['close', 'open', 'high', 'low', 'hlc3', 'hl2', 'ohlc4']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(valid_sources)}")
        
        # 動的適応が必要な場合のみEhlersUnifiedDCを初期化
        self.cycle_detector = None
        
        if self.period_mode == 'dynamic':
            # EhlersUnifiedDCのインポート（デバッグ付き）
            EhlersUnifiedDC = None
            import_success = False
            
            try:
                # 相対インポートを試行
                from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
                import_success = True
                self.logger.debug("UltimateSmoother: EhlersUnifiedDC 相対インポート成功")
            except ImportError as e1:
                self.logger.debug(f"UltimateSmoother: EhlersUnifiedDC 相対インポート失敗: {e1}")
                try:
                    # 絶対インポートを試行（パス調整付き）
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
                    import_success = True
                    self.logger.debug("UltimateSmoother: EhlersUnifiedDC 絶対インポート成功")
                except ImportError as e2:
                    self.logger.error(f"UltimateSmoother: EhlersUnifiedDC インポート失敗 - 相対: {e1}, 絶対: {e2}")
                    import_success = False
            
            if import_success and EhlersUnifiedDC is not None:
                try:
                    self.cycle_detector = EhlersUnifiedDC(
                        detector_type=self.cycle_detector_type,
                        cycle_part=self.cycle_detector_cycle_part,
                        max_cycle=self.cycle_detector_max_cycle,
                        min_cycle=self.cycle_detector_min_cycle,
                        src_type=self.src_type,
                        period_range=self.cycle_detector_period_range
                    )
                    self.logger.info(f"UltimateSmoother: 動的適応サイクル検出器を初期化: {self.cycle_detector_type}")
                except Exception as e:
                    self.logger.error(f"UltimateSmoother: サイクル検出器の初期化に失敗: {e}")
                    # フォールバックとして固定モードに変更
                    self.period_mode = 'fixed'
                    self.logger.warning("UltimateSmoother: 動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
            else:
                self.logger.error("UltimateSmoother: EhlersUnifiedDCのインポートに失敗しました")
                # フォールバックとして固定モードに変更
                self.period_mode = 'fixed'
                self.logger.warning("UltimateSmoother: EhlersUnifiedDCインポート失敗のため、固定モードにフォールバックしました。")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的適応期間を計算する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 動的期間配列
        """
        data_length = len(data) if hasattr(data, '__len__') else 0
        
        # デフォルト値で初期化
        periods = np.full(data_length, self.period, dtype=np.float64)
        
        # 動的適応期間の計算
        if self.period_mode == 'dynamic' and self.cycle_detector is not None:
            try:
                # ドミナントサイクルを計算
                dominant_cycles = self.cycle_detector.calculate(data)
                
                if dominant_cycles is not None and len(dominant_cycles) == data_length:
                    # サイクル期間に乗数を適用
                    adjusted_cycles = dominant_cycles * self.cycle_period_multiplier
                    
                    # サイクル期間を適切な範囲にクリップ
                    periods = np.clip(adjusted_cycles, 
                                     self.cycle_detector_min_cycle, 
                                     self.cycle_detector_max_cycle)
                    
                    self.logger.debug(f"動的期間計算完了 - 期間範囲: [{np.min(periods):.1f}-{np.max(periods):.1f}]")
                else:
                    self.logger.warning("ドミナントサイクルの計算結果が無効です。固定期間を使用します。")
                    
            except Exception as e:
                self.logger.error(f"動的期間計算中にエラー: {e}")
                # エラー時は固定期間を使用
        
        return periods
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # 最小限のパラメータ情報
            ukf_sig = str(self.ukf_params) if self.ukf_params else "None"
            params_sig = (f"{self.period}_{self.period_mode}_{self.src_type}_{ukf_sig}_"
                         f"{self.cycle_detector_type}_{self.cycle_detector_cycle_part}_"
                         f"{self.cycle_detector_max_cycle}_{self.cycle_detector_min_cycle}_"
                         f"{self.cycle_period_multiplier}_{self.cycle_detector_period_range}")
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.period_mode}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateSmootherResult:
        """
        アルティメットスムーザーを計算する（動的適応対応）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            UltimateSmootherResult: アルティメットスムーザーの値と関連情報を含む結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UltimateSmootherResult(
                    values=cached_result.values.copy(),
                    coefficients=cached_result.coefficients.copy()
                )
            
            # 価格ソースの計算（シンプル版を使用、UKFは使わない）
            price_source = calculate_source_simple(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 4:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低4点以上を推奨します。")
            
            # 動的適応期間の計算
            if self.period_mode == 'dynamic':
                self.logger.debug("動的適応期間を計算中...")
                periods = self._get_dynamic_periods(data)
                
                # 動的適応アルティメットスムーザーの計算
                ultimate_smoother_values, coefficients = calculate_ultimate_smoother_adaptive(
                    price_source, periods
                )
            else:
                # 固定期間アルティメットスムーザーの計算（Numba最適化関数を使用）
                ultimate_smoother_values, coefficients = calculate_ultimate_smoother(
                    price_source, self.period
                )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = UltimateSmootherResult(
                values=ultimate_smoother_values.copy(),
                coefficients=coefficients.copy()
            )
            
            # キャッシュを更新
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = ultimate_smoother_values  # 基底クラスの要件を満たすため
            
            mode_info = f"モード:{self.period_mode}"
            if self.period_mode == 'dynamic':
                mode_info += f", サイクル検出器:{self.cycle_detector_type}"
            
            self.logger.debug(f"Ultimate Smoother 計算完了 - {mode_info}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UltimateSmoother計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = UltimateSmootherResult(
                values=np.array([]),
                coefficients=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """アルティメットスムーザー値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        使用された係数を取得する（デバッグ用）
        
        Returns:
            np.ndarray: 係数配列
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.coefficients.copy()
    
    def get_dynamic_periods_info(self) -> dict:
        """動的適応期間の情報を取得する"""
        info = {
            'period_mode': self.period_mode,
            'cycle_detector_available': self.cycle_detector is not None
        }
        
        # サイクル検出器の情報
        if self.cycle_detector is not None:
            info.update({
                'cycle_detector_type': self.cycle_detector_type,
                'cycle_detector_cycle_part': self.cycle_detector_cycle_part,
                'cycle_detector_max_cycle': self.cycle_detector_max_cycle,
                'cycle_detector_min_cycle': self.cycle_detector_min_cycle,
                'cycle_period_multiplier': self.cycle_period_multiplier,
                'cycle_detector_period_range': self.cycle_detector_period_range
            })
        
        return info
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        if self.cycle_detector is not None:
            self.cycle_detector.reset() 