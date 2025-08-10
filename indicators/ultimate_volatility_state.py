#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange
import traceback

from .indicator import Indicator
from .price_source import PriceSource
from .str import STR
from .volatility import volatility
from .smoother.ultimate_smoother import UltimateSmoother
from .zscore import ZScore


@dataclass
class UltimateVolatilityStateResult:
    """究極のボラティリティ状態判別結果"""
    state: np.ndarray           # ボラティリティ状態 (1: 高ボラティリティ, 0: 低ボラティリティ)
    probability: np.ndarray     # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray      # 生のボラティリティスコア
    components: Dict[str, np.ndarray]  # 各コンポーネントの寄与度


@njit(fastmath=True, cache=True)
def calculate_price_acceleration(prices: np.ndarray, period: int) -> np.ndarray:
    """価格の加速度を計算（2次微分）"""
    length = len(prices)
    acceleration = np.zeros(length)
    
    if length < period + 2:
        return acceleration
    
    for i in range(period + 1, length):
        if prices[i-period] > 0 and prices[i] > 0:
            vel1 = (prices[i-period//2] - prices[i-period]) / prices[i-period]
            vel2 = (prices[i] - prices[i-period//2]) / prices[i-period//2] if prices[i-period//2] > 0 else 0
            acceleration[i] = abs(vel2 - vel1)
    
    return acceleration


@njit(fastmath=True, cache=True)
def calculate_range_volatility(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """レンジベースのボラティリティを計算"""
    length = len(close)
    volatility = np.zeros(length)
    
    for i in range(period, length):
        sum_range = 0.0
        sum_close = 0.0
        count = 0
        
        for j in range(i - period + 1, i + 1):
            if high[j] > low[j] and close[j] > 0:
                sum_range += (high[j] - low[j])
                sum_close += close[j]
                count += 1
        
        if count > 0 and sum_close > 0:
            volatility[i] = (sum_range / count) / (sum_close / count)
    
    return volatility


@njit(fastmath=True, cache=True)
def calculate_entropy_volatility(returns: np.ndarray, period: int) -> np.ndarray:
    """情報エントロピーベースのボラティリティを計算"""
    length = len(returns)
    entropy = np.zeros(length)
    
    for i in range(period, length):
        window = returns[i-period+1:i+1]
        
        # ヒストグラムを作成（10ビン）
        min_val = np.min(window)
        max_val = np.max(window)
        
        if max_val > min_val:
            bins = 10
            bin_counts = np.zeros(bins)
            bin_width = (max_val - min_val) / bins
            
            for val in window:
                bin_idx = int((val - min_val) / bin_width)
                if bin_idx >= bins:
                    bin_idx = bins - 1
                bin_counts[bin_idx] += 1
            
            # エントロピーを計算
            total = np.sum(bin_counts)
            if total > 0:
                ent = 0.0
                for count in bin_counts:
                    if count > 0:
                        p = count / total
                        ent -= p * np.log(p)
                entropy[i] = ent / np.log(bins)  # 正規化
    
    return entropy


@njit(fastmath=True, cache=True)
def calculate_fractal_dimension(prices: np.ndarray, period: int) -> np.ndarray:
    """価格系列のフラクタル次元を計算"""
    length = len(prices)
    fractal_dim = np.zeros(length)
    
    for i in range(period, length):
        window = prices[i-period+1:i+1]
        
        # ボックスカウンティング法の簡易版
        n = period
        max_price = np.max(window)
        min_price = np.min(window)
        
        if max_price > min_price:
            # 価格範囲を正規化
            norm_prices = (window - min_price) / (max_price - min_price)
            
            # 軌跡の長さを計算
            path_length = 0.0
            for j in range(1, n):
                dx = 1.0 / n
                dy = abs(norm_prices[j] - norm_prices[j-1])
                path_length += np.sqrt(dx*dx + dy*dy)
            
            # フラクタル次元の推定
            if path_length > 0:
                fractal_dim[i] = np.log(path_length) / np.log(1.0/n)
    
    return fractal_dim


@njit(fastmath=True, parallel=True, cache=True)
def fuse_volatility_signals(
    str_zscore: np.ndarray,
    vol_zscore: np.ndarray,
    acceleration: np.ndarray,
    range_vol: np.ndarray,
    entropy: np.ndarray,
    fractal: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """複数のボラティリティシグナルを融合"""
    length = len(str_zscore)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    
    # 重み係数（各指標の重要度）
    w_str = 0.25
    w_vol = 0.25
    w_acc = 0.15
    w_range = 0.15
    w_entropy = 0.10
    w_fractal = 0.10
    
    for i in prange(length):
        # 各指標を正規化（0-1範囲）
        norm_str = min(max(abs(str_zscore[i]) / 3.0, 0.0), 1.0)
        norm_vol = min(max(abs(vol_zscore[i]) / 3.0, 0.0), 1.0)
        norm_acc = min(max(acceleration[i] * 100, 0.0), 1.0)
        norm_range = min(max(range_vol[i] * 10, 0.0), 1.0)
        norm_entropy = entropy[i]  # 既に0-1範囲
        norm_fractal = min(max((fractal[i] - 1.0) / 0.5, 0.0), 1.0)
        
        # 加重平均スコア
        score = (w_str * norm_str + 
                 w_vol * norm_vol + 
                 w_acc * norm_acc + 
                 w_range * norm_range + 
                 w_entropy * norm_entropy + 
                 w_fractal * norm_fractal)
        
        raw_score[i] = score
        
        # シグモイド関数で確率に変換
        k = 10.0  # 急峻さパラメータ
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - threshold)))
        
        # 状態判定
        state[i] = 1 if score > threshold else 0
    
    return state, probability, raw_score


class UltimateVolatilityState(Indicator):
    """
    究極のボラティリティ状態判別インジケーター
    
    複数の学問分野から厳選したアルゴリズムを組み合わせて、
    市場が高ボラティリティか低ボラティリティかを超高精度で判定します。
    
    使用アルゴリズム:
    1. STR (Smooth True Range) - 価格レンジの変動
    2. 統計的ボラティリティ - リターンの標準偏差
    3. 価格加速度 - 価格変化の2次微分
    4. レンジボラティリティ - High-Low比率
    5. 情報エントロピー - 価格分布の不確実性
    6. フラクタル次元 - 価格軌跡の複雑性
    
    特徴:
    - 超高精度: 多角的な分析による精密な判定
    - 超低遅延: Ultimate Smootherによる最小遅延
    - 超適応性: 市場状況に応じた動的調整
    """
    
    def __init__(
        self,
        period: int = 21,                    # 基本期間
        threshold: float = 0.5,              # 高/低ボラティリティの閾値
        zscore_period: int = 50,             # Z-Score計算期間
        src_type: str = 'hlc3',              # 価格ソース
        smoother_period: int = 5,            # スムージング期間
        adaptive_threshold: bool = True      # 適応的閾値調整
    ):
        """
        コンストラクタ
        
        Args:
            period: 基本計算期間
            threshold: ボラティリティ判定閾値 (0.0-1.0)
            zscore_period: Z-Score正規化期間
            src_type: 価格ソースタイプ
            smoother_period: 最終出力のスムージング期間
            adaptive_threshold: 閾値の動的調整を有効化
        """
        super().__init__(f"UltimateVolatilityState(period={period}, threshold={threshold})")
        
        self.period = period
        self.threshold = threshold
        self.zscore_period = zscore_period
        self.src_type = src_type.lower()
        self.smoother_period = smoother_period
        self.adaptive_threshold = adaptive_threshold
        
        # コンポーネントインジケーターの初期化
        self.str_indicator = STR(
            period=period,
            src_type=src_type,
            period_mode='dynamic'
        )
        
        self.vol_indicator = volatility(
            period_mode='adaptive',
            fixed_period=period,
            calculation_mode='return',
            return_type='log',
            smoother_type='hma',
            smoother_period=period // 2
        )
        
        self.smoother = UltimateSmoother(
            period=smoother_period,
            src_type='close'
        )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0]['close']) if length > 0 else 0.0
                last_val = float(data.iloc[-1]['close']) if length > 0 else 0.0
            else:
                length = len(data)
                first_val = float(data[0, 3]) if length > 0 else 0.0
                last_val = float(data[-1, 3]) if length > 0 else 0.0
            
            params_sig = f"{self.period}_{self.threshold}_{self.zscore_period}_{self.adaptive_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.period}_{self.threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateVolatilityStateResult:
        """
        ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            UltimateVolatilityStateResult: 判定結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                return self._result_cache[data_hash]
            
            # データ準備
            if isinstance(data, pd.DataFrame):
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                high = data[:, 1]
                low = data[:, 2]
                close = data[:, 3]
            
            length = len(close)
            if length < self.period + self.zscore_period:
                # データ不足
                empty_result = UltimateVolatilityStateResult(
                    state=np.zeros(length, dtype=np.int8),
                    probability=np.zeros(length),
                    raw_score=np.zeros(length),
                    components={}
                )
                return empty_result
            
            # 1. STRベースのボラティリティ
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            
            # STRのZ-Score
            str_zscore_calc = ZScore(period=self.zscore_period)
            str_df = pd.DataFrame({'close': str_values})
            str_zscore = str_zscore_calc.calculate(str_df)
            
            # 2. 統計的ボラティリティ
            vol_values = self.vol_indicator.calculate(data)
            
            # ボラティリティのZ-Score
            vol_zscore_calc = ZScore(period=self.zscore_period)
            vol_df = pd.DataFrame({'close': vol_values})
            vol_zscore = vol_zscore_calc.calculate(vol_df)
            
            # 3. 価格加速度
            src_prices = PriceSource.calculate_source(data, self.src_type)
            acceleration = calculate_price_acceleration(src_prices, self.period)
            
            # 4. レンジボラティリティ
            range_vol = calculate_range_volatility(high, low, close, self.period)
            
            # 5. 情報エントロピー
            returns = np.zeros(length)
            for i in range(1, length):
                if close[i-1] > 0:
                    returns[i] = np.log(close[i] / close[i-1])
            entropy = calculate_entropy_volatility(returns, self.period)
            
            # 6. フラクタル次元
            fractal = calculate_fractal_dimension(close, self.period)
            
            # 適応的閾値調整
            effective_threshold = self.threshold
            if self.adaptive_threshold:
                # 直近のスコアの平均を基に閾値を調整
                lookback = min(100, length // 4)
                if length > lookback:
                    recent_scores = []
                    for i in range(length - lookback, length):
                        if i >= self.period + self.zscore_period:
                            score = (abs(str_zscore[i]) + abs(vol_zscore[i])) / 2
                            recent_scores.append(score)
                    
                    if recent_scores:
                        median_score = np.median(recent_scores)
                        effective_threshold = 0.3 + 0.4 * (median_score / 3.0)
                        effective_threshold = np.clip(effective_threshold, 0.3, 0.7)
            
            # シグナル融合
            state, probability, raw_score = fuse_volatility_signals(
                str_zscore, vol_zscore, acceleration, range_vol,
                entropy, fractal, effective_threshold
            )
            
            # 最終的なスムージング
            state_df = pd.DataFrame({'close': state.astype(np.float64)})
            smoothed_state_result = self.smoother.calculate(state_df)
            smoothed_state = (smoothed_state_result.values > 0.5).astype(np.int8)
            
            prob_df = pd.DataFrame({'close': probability})
            smoothed_prob_result = self.smoother.calculate(prob_df)
            smoothed_probability = smoothed_prob_result.values
            
            # コンポーネントの寄与度を保存
            components = {
                'str_zscore': str_zscore.copy(),
                'vol_zscore': vol_zscore.copy(),
                'acceleration': acceleration.copy(),
                'range_volatility': range_vol.copy(),
                'entropy': entropy.copy(),
                'fractal_dimension': fractal.copy()
            }
            
            # 結果作成
            result = UltimateVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                components=components
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = smoothed_state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            # エラー時は空の結果を返す
            return UltimateVolatilityStateResult(
                state=np.zeros(len(data), dtype=np.int8),
                probability=np.zeros(len(data)),
                raw_score=np.zeros(len(data)),
                components={}
            )
    
    def get_state(self) -> Optional[np.ndarray]:
        """現在のボラティリティ状態を取得 (1: 高, 0: 低)"""
        if self._values is not None:
            return self._values.astype(np.int8)
        return None
    
    def get_probability(self) -> Optional[np.ndarray]:
        """状態の確信度を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return latest_result.probability
        return None
    
    def get_components(self) -> Optional[Dict[str, np.ndarray]]:
        """各コンポーネントの寄与度を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return latest_result.components
        return None
    
    def is_high_volatility(self) -> bool:
        """最新の状態が高ボラティリティかどうか"""
        state = self.get_state()
        if state is not None and len(state) > 0:
            return bool(state[-1] == 1)
        return False
    
    def is_low_volatility(self) -> bool:
        """最新の状態が低ボラティリティかどうか"""
        state = self.get_state()
        if state is not None and len(state) > 0:
            return bool(state[-1] == 0)
        return False
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        self.vol_indicator.reset()
        self.smoother.reset()