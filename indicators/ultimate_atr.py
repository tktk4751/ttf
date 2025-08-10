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
    from .smoother.ultimate_smoother import UltimateSmoother
    from .smoother.unscented_kalman_filter import UnscentedKalmanFilter
    # EhlersUnifiedDC は関数内でインポートして循環インポートを回避
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    # EhlersUnifiedDC は関数内でインポートして循環インポートを回避


@dataclass
class UltimateATRResult:
    """アルティメットATRの計算結果"""
    values: np.ndarray           # アルティメットATR値（最終的なUltimate Smoother平滑化済み）
    true_range: np.ndarray       # 通常のTrue Range値
    raw_atr: np.ndarray          # 通常のATR値（比較用）
    ultimate_smoothed: np.ndarray # UKFフィルター済み値（中間結果）


@njit(fastmath=True, cache=True)
def calculate_standard_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    通常のTrue Range（ATR用）を計算する
    
    True Range = max(
        High - Low,
        abs(High - Close[1]),
        abs(Low - Close[1])
    )
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        np.ndarray: True Range値の配列
    """
    length = len(close)
    true_range = np.zeros(length, dtype=np.float64)
    
    # 最初の値は単純にHigh - Low
    true_range[0] = high[0] - low[0]
    
    for i in range(1, length):
        # 通常のTrue Range計算
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        
        true_range[i] = max(hl, hc, lc)
    
    return true_range


@njit(fastmath=True, cache=True)
def calculate_standard_atr(true_range: np.ndarray, period: int) -> np.ndarray:
    """
    通常のATR（単純移動平均版）を計算する
    
    Args:
        true_range: True Range値の配列
        period: ATR期間
    
    Returns:
        np.ndarray: ATR値の配列
    """
    length = len(true_range)
    atr = np.zeros(length, dtype=np.float64)
    
    # 最初の期間分はTRの平均
    for i in range(period):
        if i == 0:
            atr[i] = true_range[i]
        else:
            sum_tr = 0.0
            for j in range(i + 1):
                sum_tr += true_range[j]
            atr[i] = sum_tr / (i + 1)
    
    # 期間以降は単純移動平均
    for i in range(period, length):
        sum_tr = 0.0
        for j in range(period):
            sum_tr += true_range[i - j]
        atr[i] = sum_tr / period
    
    return atr


class UltimateATR(Indicator):
    """
    アルティメットATR（Ultimate Average True Range）インジケーター
    
    通常のATRとの違い：
    - 通常のATR: True RangeをWilder's Smoothing（指数移動平均）で平滑化
    - アルティメットATR: True RangeをUKF（無香料カルマンフィルター）でフィルタリング後、Ultimate Smootherで平滑化
    
    特徴：
    - 超低遅延: UKF+Ultimate Smootherによる最小限の遅延
    - 高精度: UKFによる非線形フィルタリングと通常のTrue Range計算による正確性
    - 適応性: 固定期間または動的期間対応
    - サイクル検出: 複数のEhlersサイクル検出器による期間自動調整
    - 2段階フィルタリング: UKFで非線形ノイズ除去、Ultimate Smootherで最終平滑化
    """
    
    def __init__(
        self,
        ultimate_smoother_period: float = 10.0,  # Ultimate Smoother期間（第1段階）
        zlema_period: int = 20,                   # ZLEMA期間（第2段階）
        src_type: str = 'hlc3',                   # プライスソース（一貫性のため）
        ukf_params: Optional[Dict] = None,        # UKFパラメータ（UKFソース使用時）
        # 動的適応パラメータ
        period_mode: str = 'fixed',               # 期間モード ('fixed' or 'dynamic')
        # サイクル検出器パラメータ
        cycle_detector_type: str = 'absolute_ultimate',
        cycle_detector_cycle_part: float = 0.5,
        cycle_detector_max_cycle: int = 55,
        cycle_detector_min_cycle: int = 5,
        cycle_period_multiplier: float = 1.0,
        cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """
        コンストラクタ
        
        Args:
            ultimate_smoother_period: Ultimate Smoother期間（第1段階、デフォルト: 10.0）
            zlema_period: ZLEMA期間（第2段階、デフォルト: 20）
            src_type: プライスソースタイプ（一貫性のため）
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
            period_mode: 期間モード ('fixed' or 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            cycle_detector_cycle_part: サイクル検出器のサイクル部分倍率
            cycle_detector_max_cycle: サイクル検出器の最大サイクル期間
            cycle_detector_min_cycle: サイクル検出器の最小サイクル期間
            cycle_period_multiplier: サイクル期間の乗数
            cycle_detector_period_range: サイクル検出器の周期範囲
        """
        # 指標名の作成
        indicator_name = f"UltimateATR(us_period={ultimate_smoother_period}, zlema_period={zlema_period}, {src_type}, cycle={cycle_detector_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.ultimate_smoother_period = ultimate_smoother_period
        self.zlema_period = zlema_period
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
        if self.ultimate_smoother_period <= 0:
            raise ValueError("ultimate_smoother_periodは0より大きい必要があります")
        if self.ultimate_smoother_period < 2:
            raise ValueError("ultimate_smoother_periodは2以上である必要があります（フィルター安定性のため）")
        if self.zlema_period <= 0:
            raise ValueError("zlema_periodは0より大きい必要があります")
        if self.zlema_period < 2:
            raise ValueError("zlema_periodは2以上である必要があります（フィルター安定性のため）")
        
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効なperiod_mode: {self.period_mode}. 'fixed' または 'dynamic' を指定してください。")
        
        # ソースタイプの検証（PriceSourceから利用可能なタイプを取得）
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # 動的適応が必要な場合のみEhlersUnifiedDCを初期化
        self.cycle_detector = None
        
        if self.period_mode == 'dynamic':
            # EhlersUnifiedDCのインポート（デバッグ付き）
            EhlersUnifiedDC = None
            import_success = False
            
            try:
                # 相対インポートを試行
                from .cycle.ehlers_unified_dc import EhlersUnifiedDC
                import_success = True
                self.logger.debug("UltimateATR: EhlersUnifiedDC 相対インポート成功")
            except ImportError as e1:
                self.logger.debug(f"UltimateATR: EhlersUnifiedDC 相対インポート失敗: {e1}")
                try:
                    # 絶対インポートを試行（パス調整付き）
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
                    import_success = True
                    self.logger.debug("UltimateATR: EhlersUnifiedDC 絶対インポート成功")
                except ImportError as e2:
                    self.logger.error(f"UltimateATR: EhlersUnifiedDC インポート失敗 - 相対: {e1}, 絶対: {e2}")
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
                    self.logger.info(f"UltimateATR: 動的適応サイクル検出器を初期化: {self.cycle_detector_type}")
                except Exception as e:
                    self.logger.error(f"UltimateATR: サイクル検出器の初期化に失敗: {e}")
                    # フォールバックとして固定モードに変更
                    self.period_mode = 'fixed'
                    self.logger.warning("UltimateATR: 動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
            else:
                self.logger.error("UltimateATR: EhlersUnifiedDCのインポートに失敗しました")
                # フォールバックとして固定モードに変更
                self.period_mode = 'fixed'
                self.logger.warning("UltimateATR: EhlersUnifiedDCインポート失敗のため、固定モードにフォールバックしました。")
        
        # Ultimate Smootherの初期化（第1段階：10期間で平滑化）
        self._ultimate_smoother = UltimateSmoother(
            period=self.ultimate_smoother_period,
            src_type='close',
            ukf_params=self.ukf_params,
            period_mode=self.period_mode,
            cycle_detector_type=self.cycle_detector_type,
            cycle_detector_cycle_part=self.cycle_detector_cycle_part,
            cycle_detector_max_cycle=self.cycle_detector_max_cycle,
            cycle_detector_min_cycle=self.cycle_detector_min_cycle,
            cycle_period_multiplier=self.cycle_period_multiplier,
            cycle_detector_period_range=self.cycle_detector_period_range
        )
        
        # 第2段階の平滑化期間を保存
        self.ema_period = self.zlema_period
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _calculate_ema(self, values: np.ndarray, period: int) -> np.ndarray:
        """
        EMAを計算する（第2段階の平滑化用）
        
        Args:
            values: 入力値の配列
            period: EMA期間
            
        Returns:
            EMA値の配列
        """
        if len(values) == 0:
            return np.array([])
        
        alpha = 2.0 / (period + 1.0)
        result = np.zeros_like(values)
        
        # 最初の有効な値を見つける
        first_valid_idx = 0
        for i in range(len(values)):
            if not np.isnan(values[i]) and values[i] > 0:
                result[i] = values[i]
                first_valid_idx = i
                break
        
        # EMAを計算
        for i in range(first_valid_idx + 1, len(values)):
            if not np.isnan(values[i]) and values[i] > 0:
                result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]  # 前の値を保持
        
        return result
    
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
            params_sig = (f"{self.ultimate_smoother_period}_{self.period_mode}_{self.src_type}_{ukf_sig}_"
                         f"{self.cycle_detector_type}_{self.cycle_detector_cycle_part}_"
                         f"{self.cycle_detector_max_cycle}_{self.cycle_detector_min_cycle}_"
                         f"{self.cycle_period_multiplier}_{self.cycle_detector_period_range}")
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.ultimate_smoother_period}_{self.period_mode}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateATRResult:
        """
        アルティメットATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close
        
        Returns:
            UltimateATRResult: アルティメットATRの値と関連情報を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UltimateATRResult(
                    values=cached_result.values.copy(),
                    true_range=cached_result.true_range.copy(),
                    raw_atr=cached_result.raw_atr.copy(),
                    ultimate_smoothed=cached_result.ultimate_smoothed.copy()
                )
            
            # データの準備
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]   # high
                low = data[:, 2]    # low  
                close = data[:, 3]  # close
            
            # NumPy配列に変換
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # データ長の検証
            data_length = len(close)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            # 通常のTrue Range値を計算
            true_range = calculate_standard_true_range(high, low, close)
            
            # 比較用に通常のATRを計算（期間は従来の14を使用）
            raw_atr = calculate_standard_atr(true_range, 14)
            
            # 第1段階：True RangeをDataFrame形式に変換してUltimate Smootherに渡す
            tr_df = pd.DataFrame({'close': true_range})
            
            # 動的期間のログ出力
            mode_info = f"モード:{self.period_mode}"
            if self.period_mode == 'dynamic':
                mode_info += f", サイクル検出器:{self.cycle_detector_type}"
            
            # 第1段階：Ultimate Smootherを適用（10期間で平滑化）
            smoother_result = self._ultimate_smoother.calculate(tr_df)
            ultimate_smoothed_values = smoother_result.values
            
            # 第2段階：Ultimate Smootherの結果をEMAで更に平滑化（20期間）
            # ZLEMAではなく、シンプルなEMAを使用
            try:
                self.logger.debug(f"Ultimate Smoother結果の形状: {ultimate_smoothed_values.shape}")
                self.logger.debug(f"Ultimate Smoother結果の最初の5値: {ultimate_smoothed_values[:5]}")
                
                final_atr_values = self._calculate_ema(ultimate_smoothed_values, self.ema_period)
                
                self.logger.debug(f"EMA結果の形状: {final_atr_values.shape}")
                self.logger.debug(f"EMA結果の最初の5値: {final_atr_values[:5]}")
                
            except Exception as e:
                self.logger.error(f"EMA計算中にエラー: {e}")
                final_atr_values = ultimate_smoothed_values
            
            # 結果の保存
            result = UltimateATRResult(
                values=final_atr_values.copy(),
                true_range=true_range.copy(),
                raw_atr=raw_atr.copy(),
                ultimate_smoothed=ultimate_smoothed_values.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = final_atr_values  # 基底クラスの要件
            
            self.logger.debug(f"UltimateATR 計算完了 - {mode_info}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UltimateATR計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            return UltimateATRResult(
                values=np.array([]),
                true_range=np.array([]),
                raw_atr=np.array([]),
                ultimate_smoothed=np.array([])
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """アルティメットATR値のみを取得する（後方互換性のため）"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.values.copy()
    
    def get_true_range(self) -> Optional[np.ndarray]:
        """True Range値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.true_range.copy()
    
    def get_raw_atr(self) -> Optional[np.ndarray]:
        """通常のATR値を取得する（比較用）"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.raw_atr.copy()
    
    def get_ultimate_smoothed(self) -> Optional[np.ndarray]:
        """Ultimate Smootherで平滑化された値を取得する（中間結果）"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.ultimate_smoothed.copy()
    
    def get_comparison_metrics(self) -> Optional[Dict[str, float]]:
        """アルティメットATRと通常のATRの比較メトリクスを取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        
        if len(result.values) == 0 or len(result.raw_atr) == 0:
            return None
        
        # 最新値の比較
        latest_ultimate = result.values[-1]
        latest_raw = result.raw_atr[-1]
        latest_ultimate_smoothed = result.ultimate_smoothed[-1] if len(result.ultimate_smoothed) > 0 else 0.0
        
        # 統計的比較
        ultimate_mean = np.mean(result.values[result.values > 0])
        raw_mean = np.mean(result.raw_atr[result.raw_atr > 0])
        ultimate_smoothed_mean = np.mean(result.ultimate_smoothed[result.ultimate_smoothed > 0])
        
        ultimate_std = np.std(result.values[result.values > 0])
        raw_std = np.std(result.raw_atr[result.raw_atr > 0])
        ultimate_smoothed_std = np.std(result.ultimate_smoothed[result.ultimate_smoothed > 0])
        
        return {
            'latest_ultimate_atr': float(latest_ultimate),
            'latest_raw_atr': float(latest_raw),
            'latest_ultimate_smoothed': float(latest_ultimate_smoothed),
            'ultimate_vs_raw_ratio': float(latest_ultimate / latest_raw) if latest_raw > 0 else 0.0,
            'ultimate_mean': float(ultimate_mean),
            'raw_mean': float(raw_mean),
            'ultimate_smoothed_mean': float(ultimate_smoothed_mean),
            'ultimate_std': float(ultimate_std),
            'raw_std': float(raw_std),
            'ultimate_smoothed_std': float(ultimate_smoothed_std),
            'smoothing_effectiveness': float(raw_std / ultimate_std) if ultimate_std > 0 else 0.0,
            'stage1_smoothing_effectiveness': float(raw_std / ultimate_smoothed_std) if ultimate_smoothed_std > 0 else 0.0,
            'stage2_smoothing_effectiveness': float(ultimate_smoothed_std / ultimate_std) if ultimate_std > 0 else 0.0
        }
    
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
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._ultimate_smoother.reset()
        if self.cycle_detector is not None:
            self.cycle_detector.reset()