#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, float64
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .ultimate_smoother import UltimateSmoother
    from .ultimate_ma import UltimateMA
    from .str import STR
    from .ultra_quantum_adaptive_trend_range_discriminator import UltraQuantumAdaptiveTrendRangeDiscriminator
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ultimate_smoother import UltimateSmoother
    from ultimate_ma import UltimateMA
    from str import STR
    from ultra_quantum_adaptive_trend_range_discriminator import UltraQuantumAdaptiveTrendRangeDiscriminator


@njit(fastmath=True, cache=True)
def calculate_dynamic_multipliers(uqatrd_values: np.ndarray) -> np.ndarray:
    """
    UQATRDの値に基づいて動的チャネル乗数を計算する（線形補間版）
    
    Args:
        uqatrd_values: UQATRD信号値配列 (0-1)
    
    Returns:
        np.ndarray: 動的乗数配列
        
    ルール（線形補間）:
    - 0.5から6の間で線形補間
    - 値が高ければ高いほど0.0に近づく（強いトレンド）
    - 値が低いほど6に近づく（強いレンジ相場）
    - 計算式: 6.0 - uqatrd_value * (6.0 - 0.0)
    """
    # 定数定義
    MAX_MULTIPLIER = 6.0
    MIN_MULTIPLIER = 0.0
    MULTIPLIER_RANGE = MAX_MULTIPLIER - MIN_MULTIPLIER  
    
    length = len(uqatrd_values)
    multipliers = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        value = uqatrd_values[i]
        
        # NaN値の処理
        if np.isnan(value):
            multipliers[i] = 3.25  # デフォルト値（中間値）
            continue
        
        # 値を0-1の範囲にクランプ
        safe_value = min(max(value, 0.0), 1.0)
        
        # 線形補間計算
        # 値が0の時は6.0、値が1の時は0.5
        multipliers[i] = MAX_MULTIPLIER - safe_value * MULTIPLIER_RANGE
    
    return multipliers


@dataclass
class UltimateChannelResult:
    """Ultimate Channelの計算結果"""
    upper_channel: np.ndarray    # 上側チャネル
    lower_channel: np.ndarray    # 下側チャネル
    center_line: np.ndarray      # 中心線（Ultimate Smoother または Ultimate MA）
    str_values: np.ndarray       # 使用されたSTR値
    midband_type: str            # 使用されたミッドバンドタイプ
    # 動的適応関連
    dynamic_multipliers: np.ndarray  # 動的乗数配列
    uqatrd_values: np.ndarray    # UQATRD信号値
    multiplier_mode: str         # 乗数モード ('fixed' または 'dynamic')


class UltimateChannel(Indicator):
    """
    Ultimate Channelインジケーター
    
    John Ehlersの論文「ULTIMATE CHANNEL and ULTIMATE BANDS」に基づく実装：
    - Keltner Channelの低遅延版
    - Ultimate SmootherとSTRを使用
    - 従来のEMAとATRよりも大幅に低遅延
    - ミッドバンドを Ultimate Smoother または Ultimate MA から選択可能
    
    特徴:
    - 超低遅延: Ultimate Smootherによる最小限の遅延
    - 高精度: STRによる真の範囲測定
    - 柔軟性: 期間とマルチプライヤーの調整可能
    - 選択可能なミッドバンド: Ultimate Smoother または Ultimate MA
    """
    
    def __init__(
        self,
        length: float = 20.0,                 # 中心線の期間
        str_length: float = 20.0,             # STR期間
        num_strs: float = 2.0,                # STRマルチプライヤー（固定モード時）
        src_type: str = 'ukf_hlc3',           # プライスソース
        midband_type: str = 'ultimate_smoother', # ミッドバンドタイプ ('ultimate_smoother' or 'ultimate_ma')
        ukf_params: Optional[Dict] = None,    # UKFパラメータ（UKFソース使用時）
        
        # 動的乗数適応パラメータ
        multiplier_mode: str = 'fixed',       # 乗数モード ('fixed' または 'dynamic')
        
        # UQATRD パラメータ（動的モード用）
        uqatrd_coherence_window: int = 21,    # 量子コヒーレンス分析窓
        uqatrd_entanglement_window: int = 34, # 量子エンタングルメント分析窓
        uqatrd_efficiency_window: int = 21,   # 量子効率スペクトラム分析窓
        uqatrd_uncertainty_window: int = 14,  # 量子不確定性分析窓
        uqatrd_str_period: float = 20.0,      # UQATRD用STR期間
        
        # Ultimate MA用サイクル検出器パラメーター
        cycle_detector_type: str = 'ehlers_unified_dc',
        cycle_detector_cycle_part: float = 1.0,
        cycle_detector_max_cycle: int = 120,
        cycle_detector_min_cycle: int = 5,
        cycle_period_multiplier: float = 1.0,
        cycle_detector_period_range: Tuple[int, int] = (5, 120),
    ):
        """
        コンストラクタ
        
        Args:
            length: 中心線（Ultimate Smoother）の期間（デフォルト: 20.0）
            str_length: STR期間（デフォルト: 20.0）
            num_strs: STRマルチプライヤー（固定モード時、デフォルト: 2.0）
            src_type: プライスソースタイプ（デフォルト: 'ukf_hlc3'）
            midband_type: ミッドバンドタイプ（デフォルト: 'ultimate_smoother'）
                'ultimate_smoother': Ultimate Smootherを使用
                'ultimate_ma': Ultimate MAを使用
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
            
            # 動的乗数適応パラメータ
            multiplier_mode: 乗数モード（デフォルト: 'fixed'）
                'fixed': 固定乗数（num_strs）を使用
                'dynamic': UQATRDに基づく動的乗数を使用
            
            # UQATRD パラメータ（動的モード用）
            uqatrd_coherence_window: 量子コヒーレンス分析窓（デフォルト: 21）
            uqatrd_entanglement_window: 量子エンタングルメント分析窓（デフォルト: 34）
            uqatrd_efficiency_window: 量子効率スペクトラム分析窓（デフォルト: 21）
            uqatrd_uncertainty_window: 量子不確定性分析窓（デフォルト: 14）
            uqatrd_str_period: UQATRD用STR期間（デフォルト: 20.0）
            
            cycle_detector_type: サイクル検出器タイプ（Ultimate MA用、デフォルト: 'ehlers_unified_dc'）
            cycle_detector_cycle_part: サイクル部分倍率（Ultimate MA用、デフォルト: 1.0）
            cycle_detector_max_cycle: 最大サイクル期間（Ultimate MA用、デフォルト: 120）
            cycle_detector_min_cycle: 最小サイクル期間（Ultimate MA用、デフォルト: 5）
            cycle_period_multiplier: サイクル期間乗数（Ultimate MA用、デフォルト: 1.0）
            cycle_detector_period_range: 周期範囲（Ultimate MA用、デフォルト: (5, 120)）
        """
        super().__init__(f"UltimateChannel(len={length}, str_len={str_length}, mult={num_strs}({multiplier_mode}), src={src_type}, midband={midband_type})")
        
        self.length = length
        self.str_length = str_length
        self.num_strs = num_strs
        self.src_type = src_type.lower()
        self.midband_type = midband_type.lower()
        self.ukf_params = ukf_params
        
        # 動的乗数適応パラメータ
        self.multiplier_mode = multiplier_mode.lower()
        
        # UQATRD パラメータ
        self.uqatrd_coherence_window = uqatrd_coherence_window
        self.uqatrd_entanglement_window = uqatrd_entanglement_window
        self.uqatrd_efficiency_window = uqatrd_efficiency_window
        self.uqatrd_uncertainty_window = uqatrd_uncertainty_window
        self.uqatrd_str_period = uqatrd_str_period
        
        # Ultimate MA用サイクル検出器パラメーター
        self.cycle_detector_type = cycle_detector_type
        self.cycle_detector_cycle_part = cycle_detector_cycle_part
        self.cycle_detector_max_cycle = cycle_detector_max_cycle
        self.cycle_detector_min_cycle = cycle_detector_min_cycle
        self.cycle_period_multiplier = cycle_period_multiplier
        self.cycle_detector_period_range = cycle_detector_period_range
        
        # パラメータの検証
        if self.length <= 0:
            raise ValueError("lengthは0より大きい必要があります")
        if self.str_length <= 0:
            raise ValueError("str_lengthは0より大きい必要があります")
        if self.num_strs < 0:
            raise ValueError("num_strsは0以上である必要があります")
        
        # ミッドバンドタイプの検証
        if self.midband_type not in ['ultimate_smoother', 'ultimate_ma']:
            raise ValueError(f"無効なミッドバンドタイプです: {midband_type}。有効なオプション: 'ultimate_smoother', 'ultimate_ma'")
        
        # 乗数モードの検証
        if self.multiplier_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効な乗数モードです: {multiplier_mode}。有効なオプション: 'fixed', 'dynamic'")
        
        # ソースタイプの検証
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # インジケーターの初期化
        self._center_smoother = None
        self._center_ma = None
        
        if self.midband_type == 'ultimate_smoother':
            self._center_smoother = UltimateSmoother(
                period=self.length, 
                src_type=self.src_type,
                ukf_params=self.ukf_params
            )
        elif self.midband_type == 'ultimate_ma':
            # Ultimate MAをサイクル検出器パラメーターで初期化
            self._center_ma = UltimateMA(
                src_type=self.src_type,
                cycle_detector_type=self.cycle_detector_type,
                cycle_detector_cycle_part=self.cycle_detector_cycle_part,
                cycle_detector_max_cycle=self.cycle_detector_max_cycle,
                cycle_detector_min_cycle=self.cycle_detector_min_cycle,
                cycle_period_multiplier=self.cycle_period_multiplier,
                cycle_detector_period_range=self.cycle_detector_period_range
            )
        
        self._str_indicator = STR(
            period=self.str_length,
            src_type=self.src_type
        )
        
        # UQATRD（動的乗数モード用）
        self._uqatrd = None
        if self.multiplier_mode == 'dynamic':
            try:
                self._uqatrd = UltraQuantumAdaptiveTrendRangeDiscriminator(
                    coherence_window=self.uqatrd_coherence_window,
                    entanglement_window=self.uqatrd_entanglement_window,
                    efficiency_window=self.uqatrd_efficiency_window,
                    uncertainty_window=self.uqatrd_uncertainty_window,
                    src_type=self.src_type,
                    str_period=self.uqatrd_str_period
                )
                self.logger.info(f"UQATRD初期化完了（動的乗数モード）")
            except Exception as e:
                self.logger.error(f"UQATRD初期化に失敗: {e}")
                # フォールバックとして固定モードに変更
                self.multiplier_mode = 'fixed'
                self.logger.warning("動的乗数モードの初期化に失敗したため、固定モードにフォールバックしました。")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
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
            
            ukf_sig = str(self.ukf_params) if self.ukf_params else "None"
            cycle_sig = f"{self.cycle_detector_type}_{self.cycle_detector_cycle_part}_{self.cycle_detector_max_cycle}_{self.cycle_detector_min_cycle}_{self.cycle_period_multiplier}_{self.cycle_detector_period_range}"
            uqatrd_sig = f"{self.multiplier_mode}_{self.uqatrd_coherence_window}_{self.uqatrd_entanglement_window}_{self.uqatrd_efficiency_window}_{self.uqatrd_uncertainty_window}_{self.uqatrd_str_period}"
            params_sig = f"{self.length}_{self.str_length}_{self.num_strs}_{self.src_type}_{self.midband_type}_{ukf_sig}_{cycle_sig}_{uqatrd_sig}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.length}_{self.str_length}_{self.num_strs}_{self.midband_type}_{self.cycle_detector_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateChannelResult:
        """
        Ultimate Channelを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close (+ src_typeに応じた追加カラム)
        
        Returns:
            UltimateChannelResult: Ultimate Channelの値を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UltimateChannelResult(
                    upper_channel=cached_result.upper_channel.copy(),
                    lower_channel=cached_result.lower_channel.copy(),
                    center_line=cached_result.center_line.copy(),
                    str_values=cached_result.str_values.copy(),
                    midband_type=cached_result.midband_type,
                    dynamic_multipliers=cached_result.dynamic_multipliers.copy(),
                    uqatrd_values=cached_result.uqatrd_values.copy(),
                    multiplier_mode=cached_result.multiplier_mode
                )
            
            # データの検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                # 追加のカラムチェック（src_typeに応じて）
                if self.src_type in ['hlc3', 'hl2', 'ohlc4'] and 'high' not in data.columns:
                    raise ValueError(f"ソースタイプ '{self.src_type}' には 'high' カラムが必要です")
                if self.src_type in ['ohlc4'] and 'open' not in data.columns:
                    raise ValueError(f"ソースタイプ '{self.src_type}' には 'open' カラムが必要です")
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 4:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低4点以上を推奨します。")
            
            # 中心線の計算（選択されたタイプに応じて）
            if self.midband_type == 'ultimate_smoother':
                center_result = self._center_smoother.calculate(data)
                center_line = center_result.values
            elif self.midband_type == 'ultimate_ma':
                center_result = self._center_ma.calculate(data)
                center_line = center_result.values
            else:
                raise ValueError(f"未サポートのミッドバンドタイプ: {self.midband_type}")
            
            # STRの計算
            str_result = self._str_indicator.calculate(data)
            str_values = str_result.values
            
            # 乗数の計算（固定または動的）
            uqatrd_values = np.zeros(data_length)  # デフォルト値
            
            if self.multiplier_mode == 'dynamic' and self._uqatrd is not None:
                try:
                    # UQATRDの計算
                    uqatrd_result = self._uqatrd.calculate(data)
                    uqatrd_values = uqatrd_result.trend_range_signal
                    
                    # 動的乗数の計算
                    dynamic_multipliers = calculate_dynamic_multipliers(uqatrd_values)
                    
                    self.logger.debug(f"動的乗数計算完了 - 平均乗数: {np.mean(dynamic_multipliers):.2f}")
                    
                except Exception as e:
                    self.logger.warning(f"UQATRD計算に失敗: {e}. 固定乗数を使用します。")
                    # フォールバック: 固定乗数を使用
                    dynamic_multipliers = np.full(data_length, self.num_strs)
            else:
                # 固定乗数
                dynamic_multipliers = np.full(data_length, self.num_strs)
            
            # チャネルの計算
            # 論文に基づく計算式（動的乗数版）:
            # UpperChnl = CenterLine + DynamicMultiplier * STR
            # LowerChnl = CenterLine - DynamicMultiplier * STR
            upper_channel = center_line + dynamic_multipliers * str_values
            lower_channel = center_line - dynamic_multipliers * str_values
            
            # 結果の保存
            result = UltimateChannelResult(
                upper_channel=upper_channel.copy(),
                lower_channel=lower_channel.copy(),
                center_line=center_line.copy(),
                str_values=str_values.copy(),
                midband_type=self.midband_type,
                dynamic_multipliers=dynamic_multipliers.copy(),
                uqatrd_values=uqatrd_values.copy(),
                multiplier_mode=self.multiplier_mode
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = center_line  # 基底クラスの要件
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UltimateChannel計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            return UltimateChannelResult(
                upper_channel=np.array([]),
                lower_channel=np.array([]),
                center_line=np.array([]),
                str_values=np.array([]),
                midband_type=self.midband_type,
                dynamic_multipliers=np.array([]),
                uqatrd_values=np.array([]),
                multiplier_mode=self.multiplier_mode
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """中心線のみを取得する（後方互換性のため）"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.center_line.copy()
    
    def get_upper_channel(self) -> Optional[np.ndarray]:
        """上側チャネルを取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.upper_channel.copy()
    
    def get_lower_channel(self) -> Optional[np.ndarray]:
        """下側チャネルを取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.lower_channel.copy()
    
    def get_str_values(self) -> Optional[np.ndarray]:
        """STR値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.str_values.copy()
    
    def get_midband_type(self) -> str:
        """使用されているミッドバンドタイプを取得する"""
        return self.midband_type
    
    def get_channel_width(self) -> Optional[np.ndarray]:
        """チャネル幅を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return (result.upper_channel - result.lower_channel).copy()
    
    def get_channel_position(self, price: np.ndarray) -> Optional[np.ndarray]:
        """
        チャネル内での価格位置を取得する（0-1の範囲）
        
        Args:
            price: 価格配列
            
        Returns:
            np.ndarray: チャネル内位置（0=下限、1=上限）
        """
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        
        # チャネル幅
        channel_width = result.upper_channel - result.lower_channel
        
        # ゼロ除算を避ける
        channel_width = np.where(channel_width == 0, 1.0, channel_width)
        
        # 位置計算（0-1の範囲にクリップ）
        position = (price - result.lower_channel) / channel_width
        return np.clip(position, 0.0, 1.0)
    
    def get_dynamic_multipliers(self) -> Optional[np.ndarray]:
        """
        動的乗数配列を取得する
        
        Returns:
            np.ndarray: 動的乗数配列
        """
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.dynamic_multipliers.copy()
    
    def get_uqatrd_values(self) -> Optional[np.ndarray]:
        """
        UQATRD信号値を取得する
        
        Returns:
            np.ndarray: UQATRD信号値配列 (0-1)
        """
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.uqatrd_values.copy()
    
    def get_multiplier_mode(self) -> str:
        """
        乗数モードを取得する
        
        Returns:
            str: 乗数モード ('fixed' または 'dynamic')
        """
        return self.multiplier_mode
    
    def get_multiplier_info(self) -> Dict[str, any]:
        """
        乗数の統計情報を取得する
        
        Returns:
            Dict: 乗数の統計情報
        """
        if not self._result_cache or not self._cache_keys:
            return {'multiplier_mode': self.multiplier_mode}
        
        result = self._result_cache[self._cache_keys[-1]]
        multipliers = result.dynamic_multipliers
        
        info = {
            'multiplier_mode': result.multiplier_mode,
            'mean_multiplier': float(np.mean(multipliers)),
            'std_multiplier': float(np.std(multipliers)),
            'min_multiplier': float(np.min(multipliers)),
            'max_multiplier': float(np.max(multipliers)),
            'current_multiplier': float(multipliers[-1]) if len(multipliers) > 0 else None
        }
        
        if self.multiplier_mode == 'fixed':
            info['fixed_multiplier'] = self.num_strs
        
        return info
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        if self._center_smoother is not None:
            self._center_smoother.reset()
        if self._center_ma is not None:
            self._center_ma.reset()
        self._str_indicator.reset()
        if self._uqatrd is not None:
            self._uqatrd.reset() 