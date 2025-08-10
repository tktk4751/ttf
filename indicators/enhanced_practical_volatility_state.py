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
from .smoother.ultimate_smoother import UltimateSmoother


@dataclass
class EnhancedPracticalVolatilityStateResult:
    """拡張実践的ボラティリティ状態判別結果"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    raw_score: np.ndarray                 # 生のボラティリティスコア
    str_values: np.ndarray                # STR値（超低遅延）
    egarch_volatility: np.ndarray         # EGARCH条件付きボラティリティ
    returns_volatility: np.ndarray        # リターンボラティリティ
    range_expansion: np.ndarray           # レンジ拡張度
    regime_change: np.ndarray             # 体制変化検出
    volatility_clustering: np.ndarray    # ボラティリティクラスタリング
    leverage_effect: np.ndarray          # レバレッジ効果


@njit(fastmath=True, cache=True)
def calculate_egarch_volatility(returns: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    改良EGARCH(1,1) モデル - 金融時系列特化版
    金融市場の実際のボラティリティクラスタリングを捉える
    """
    length = len(returns)
    volatility = np.zeros(length)
    leverage_effect = np.zeros(length)
    
    # バランス調整EGARCHパラメータ
    omega = 0.00005     # 小さな正の定数項  
    alpha = 0.15        # 大きさ効果（適度）
    gamma = -0.1        # レバレッジ効果（適度）
    beta = 0.8          # 持続性（バランス重視）
    
    # 簡易初期化（動的応答性重視）
    if length > period:
        # 初期期間のボラティリティ推定
        initial_returns = returns[:period]
        valid_returns = initial_returns[~np.isnan(initial_returns)]
        if len(valid_returns) > 3:
            initial_vol = np.std(valid_returns) * np.sqrt(252)  # 年率ボラティリティ
            # 現実的な範囲に調整
            initial_vol = max(min(initial_vol, 1.5), 0.05)
            volatility[0] = initial_vol
        else:
            volatility[0] = 0.3  # 30%の初期ボラティリティ
    else:
        volatility[0] = 0.3
    
    # 単純化EGARCH更新式（より動的）
    for i in range(1, length):
        if not np.isnan(returns[i-1]) and volatility[i-1] > 1e-6:
            # 標準化残差
            standardized_return = returns[i-1] / volatility[i-1]
            
            # レバレッジ効果の計算
            leverage_effect[i] = standardized_return
            
            # 簡易EGARCH更新（直接ボラティリティ更新）
            abs_std_return = abs(standardized_return)
            
            # ボラティリティの変化率計算
            shock_impact = alpha * abs_std_return  # ショック効果
            leverage_impact = gamma * standardized_return  # レバレッジ効果
            
            # 新しいボラティリティ
            vol_change = omega + shock_impact + leverage_impact
            volatility[i] = beta * volatility[i-1] + (1 - beta) * volatility[i-1] * (1 + vol_change)
            
            # 現実的範囲制限
            if volatility[i] < 0.01:   # 1%最小
                volatility[i] = 0.01
            elif volatility[i] > 2.0:   # 200%最大
                volatility[i] = 2.0
        else:
            # 欠損値の処理
            volatility[i] = volatility[i-1]
            leverage_effect[i] = 0.0
    
    return volatility, leverage_effect


@njit(fastmath=True, cache=True)
def calculate_volatility_clustering(volatility_series: np.ndarray, period: int) -> np.ndarray:
    """
    ボラティリティクラスタリングの検出
    高ボラティリティが続く期間と低ボラティリティが続く期間の特定
    """
    length = len(volatility_series)
    clustering_score = np.zeros(length)
    
    for i in range(period, length):
        window = volatility_series[i-period:i]
        
        # 現在値と過去平均の比較
        current_vol = volatility_series[i]
        mean_vol = np.mean(window)
        
        if mean_vol > 0:
            # クラスタリングスコア = 現在値 / 過去平均
            clustering_score[i] = current_vol / mean_vol
            
            # 持続性の評価（過去期間の変動係数）
            std_vol = np.std(window)
            cv = std_vol / mean_vol if mean_vol > 0 else 0
            
            # 低い変動係数は高いクラスタリングを示す
            clustering_score[i] *= (1.0 + np.exp(-cv * 5))  # シグモイド重み
        else:
            clustering_score[i] = 1.0
    
    return clustering_score


@njit(fastmath=True, cache=True)
def calculate_enhanced_range_expansion(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    拡張レンジ拡張度の計算（STRベース）
    """
    length = len(high)
    expansion = np.zeros(length)
    
    for i in range(period, length):
        # 現在のTrue Range（STR風）
        if i > 0:
            th = max(high[i], close[i-1])  # True High
            tl = min(low[i], close[i-1])   # True Low
            current_range = th - tl
        else:
            current_range = high[i] - low[i]
        
        # 過去期間の平均True Range
        avg_range = 0.0
        for j in range(period):
            idx = i - j - 1
            if idx > 0:
                th_past = max(high[idx], close[idx-1])
                tl_past = min(low[idx], close[idx-1])
                avg_range += (th_past - tl_past)
            else:
                avg_range += (high[idx] - low[idx])
        avg_range /= period
        
        # 拡張率（指数変換で極値を抑制）
        if avg_range > 0:
            raw_expansion = current_range / avg_range
            # 指数変換で極値を圧縮
            expansion[i] = 1.0 + np.log(max(raw_expansion, 0.1))
        else:
            expansion[i] = 1.0
    
    return expansion


@njit(fastmath=True, cache=True)
def calculate_enhanced_regime_change(close: np.ndarray, volatility: np.ndarray, period: int) -> np.ndarray:
    """
    拡張体制変化検出（ボラティリティ考慮版）
    """
    length = len(close)
    regime_signal = np.zeros(length)
    
    for i in range(period * 2, length):
        # 価格変化率
        short_change = abs(close[i] - close[i - period // 2]) / close[i - period // 2] if close[i - period // 2] > 0 else 0
        long_change = abs(close[i] - close[i - period]) / close[i - period] if close[i - period] > 0 else 0
        
        # ボラティリティ変化率
        current_vol = volatility[i] if not np.isnan(volatility[i]) else 0
        past_vol = volatility[i - period] if not np.isnan(volatility[i - period]) else 0
        vol_change = abs(current_vol - past_vol) / (past_vol + 1e-8)
        
        # 価格とボラティリティの複合指標
        if long_change > 0:
            price_acceleration = short_change / long_change
            vol_regime = 1.0 + vol_change  # ボラティリティ変化の影響
            
            # 総合体制変化スコア
            regime_signal[i] = price_acceleration * vol_regime
        else:
            regime_signal[i] = 1.0
    
    return regime_signal


@njit(fastmath=True, cache=True)
def calculate_enhanced_percentile(values: np.ndarray, lookback_period: int) -> np.ndarray:
    """
    拡張パーセンタイル計算（外れ値に頑健）
    """
    length = len(values)
    percentiles = np.zeros(length)
    
    for i in range(lookback_period, length):
        # 過去の値を取得
        historical_values = values[i-lookback_period:i]
        
        # 外れ値除去（IQR方式）
        sorted_values = np.sort(historical_values)
        q1_idx = len(sorted_values) // 4
        q3_idx = 3 * len(sorted_values) // 4
        
        if q3_idx < len(sorted_values) and q1_idx >= 0:
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1
            
            # 外れ値の閾値
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 外れ値を除いた値でパーセンタイル計算
            filtered_values = []
            for val in sorted_values:
                if lower_bound <= val <= upper_bound:
                    filtered_values.append(val)
            
            if len(filtered_values) > 0:
                current_value = values[i]
                count_below = 0
                for val in filtered_values:
                    if val <= current_value:
                        count_below += 1
                percentiles[i] = count_below / len(filtered_values)
            else:
                percentiles[i] = 0.5
        else:
            percentiles[i] = 0.5
    
    return percentiles


@njit(fastmath=True, parallel=True, cache=True)
def enhanced_practical_volatility_fusion(
    str_values: np.ndarray,
    egarch_volatility: np.ndarray,
    returns_vol: np.ndarray,
    range_expansion: np.ndarray,
    regime_change: np.ndarray,
    volatility_clustering: np.ndarray,
    leverage_effect: np.ndarray,
    str_percentile: np.ndarray,
    egarch_percentile: np.ndarray,
    vol_percentile: np.ndarray,
    high_vol_threshold: float = 0.7,
    low_vol_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    拡張実践的ボラティリティ融合アルゴリズム
    """
    length = len(str_values)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    raw_score = np.zeros(length)
    
    for i in prange(length):
        # 各指標の正規化と重み付け
        str_score = str_percentile[i] if not np.isnan(str_percentile[i]) else 0.5
        egarch_score = egarch_percentile[i] if not np.isnan(egarch_percentile[i]) else 0.5
        vol_score = vol_percentile[i] if not np.isnan(vol_percentile[i]) else 0.5
        
        # レンジ拡張スコア（対数変換済みなので調整）
        range_score = min(max((range_expansion[i] - 1.0) / 2.0, 0.0), 1.0) if not np.isnan(range_expansion[i]) else 0.5
        
        # 体制変化スコア（ボラティリティ考慮版）
        regime_score = min(max((regime_change[i] - 1.0) / 2.0, 0.0), 1.0) if not np.isnan(regime_change[i]) else 0.0
        
        # ボラティリティクラスタリングスコア
        clustering_score = min(max((volatility_clustering[i] - 1.0) / 1.0, 0.0), 1.0) if not np.isnan(volatility_clustering[i]) else 0.5
        
        # レバレッジ効果スコア（絶対値）
        leverage_score = min(abs(leverage_effect[i]) / 2.0, 1.0) if not np.isnan(leverage_effect[i]) else 0.0
        
        # 実用性重視の重み配分（EGARCH重み削減）
        w_str = 0.40           # STR（超低遅延、メイン）
        w_egarch = 0.10        # EGARCH（補助的）
        w_returns = 0.30       # リターンボラティリティ（重要）
        w_range = 0.10         # レンジ拡張
        w_regime = 0.05        # 体制変化
        w_clustering = 0.03    # ボラティリティクラスタリング
        w_leverage = 0.02      # レバレッジ効果
        
        score = (w_str * str_score + 
                 w_egarch * egarch_score +
                 w_returns * vol_score + 
                 w_range * range_score + 
                 w_regime * regime_score +
                 w_clustering * clustering_score +
                 w_leverage * leverage_score)
        
        raw_score[i] = score
        
        # 確率計算（シグモイド変換）
        k = 8.0  # 急峻さパラメータ
        probability[i] = 1.0 / (1.0 + np.exp(-k * (score - 0.5)))
        
        # 拡張ヒステリシス判定（レバレッジ効果考慮）
        if i > 0:
            prev_state = state[i-1]
            
            # レバレッジ効果による閾値調整
            leverage_adjustment = leverage_score * 0.1  # 最大10%の調整
            
            if prev_state == 0:  # 前回が低ボラティリティ
                # レバレッジ効果が強い場合は閾値を下げる
                adjusted_high_threshold = high_vol_threshold - leverage_adjustment
                state[i] = 1 if score > adjusted_high_threshold else 0
            else:  # 前回が高ボラティリティ
                # レバレッジ効果が強い場合は閾値を上げる（継続しやすく）
                adjusted_low_threshold = low_vol_threshold + leverage_adjustment
                state[i] = 0 if score < adjusted_low_threshold else 1
        else:
            # 初回判定
            state[i] = 1 if score > (high_vol_threshold + low_vol_threshold) / 2 else 0
    
    return state, probability, raw_score


class EnhancedPracticalVolatilityState(Indicator):
    """
    拡張実践的ボラティリティ状態判別インジケーター
    
    主要改善点:
    1. ATR → STR: 超低遅延のSmooth True Range
    2. EGARCH追加: レバレッジ効果を含む高精度ボラティリティモデリング
    3. ボラティリティクラスタリング検出
    4. 拡張体制変化検出（ボラティリティ考慮）
    5. 外れ値に頑健なパーセンタイル計算
    
    特徴:
    - 更に高精度: STR + EGARCH の最強組み合わせ
    - 超低遅延: STRによる遅延最小化
    - レバレッジ効果: 非対称ボラティリティの考慮
    - 実用性維持: 現実的なボラティリティ分布
    """
    
    def __init__(
        self,
        str_period: int = 14,                 # STR計算期間
        vol_period: int = 20,                 # ボラティリティ計算期間
        egarch_period: int = 30,              # EGARCH期間
        percentile_lookback: int = 252,       # パーセンタイル計算期間
        high_vol_threshold: float = 0.75,     # 高ボラティリティ閾値
        low_vol_threshold: float = 0.25,      # 低ボラティリティ閾値
        src_type: str = 'hlc3',               # 価格ソース
        smoothing: bool = True                 # スムージングの有効化
    ):
        """
        コンストラクタ
        
        Args:
            str_period: STR計算期間
            vol_period: ボラティリティ計算期間
            egarch_period: EGARCH計算期間
            percentile_lookback: パーセンタイル計算の振り返り期間
            high_vol_threshold: 高ボラティリティ判定閾値
            low_vol_threshold: 低ボラティリティ判定閾値
            src_type: 価格ソースタイプ
            smoothing: 最終結果のスムージング
        """
        super().__init__(f"EnhancedPracticalVolatilityState(STR={str_period}, EGARCH={egarch_period})")
        
        self.str_period = str_period
        self.vol_period = vol_period
        self.egarch_period = egarch_period
        self.percentile_lookback = percentile_lookback
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.src_type = src_type.lower()
        self.smoothing = smoothing
        
        # STRインジケーター（超低遅延）
        self.str_indicator = STR(
            period=str_period,
            src_type=src_type,
            period_mode='fixed'  # 安定性重視
        )
        
        # スムージング用
        if self.smoothing:
            self.smoother = UltimateSmoother(period=3, src_type='close')
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EnhancedPracticalVolatilityStateResult:
        """
        拡張実践的ボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            EnhancedPracticalVolatilityStateResult: 判定結果
        """
        try:
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
            min_required = max(self.str_period, self.vol_period, self.egarch_period, self.percentile_lookback // 4)
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # 1. STRベースボラティリティ（超低遅延）
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            
            # 2. リターン計算
            returns = np.zeros(length)
            for i in range(1, length):
                if close[i-1] > 0:
                    returns[i] = np.log(close[i] / close[i-1])
            
            # 3. EGARCH条件付きボラティリティ（高精度）
            egarch_vol, leverage_effect = calculate_egarch_volatility(returns, self.egarch_period)
            
            # 4. 従来のリターンボラティリティ
            returns_vol = np.zeros(length)
            for i in range(self.vol_period, length):
                window_returns = returns[i-self.vol_period:i]
                valid_returns = window_returns[~np.isnan(window_returns)]
                if len(valid_returns) > 1:
                    returns_vol[i] = np.std(valid_returns) * np.sqrt(252)
            
            # 5. 拡張レンジ拡張度（STRベース）
            range_expansion = calculate_enhanced_range_expansion(high, low, close, self.str_period)
            
            # 6. 拡張体制変化検出
            regime_change = calculate_enhanced_regime_change(close, egarch_vol, self.vol_period)
            
            # 7. ボラティリティクラスタリング
            volatility_clustering = calculate_volatility_clustering(egarch_vol, self.vol_period)
            
            # 8. 拡張パーセンタイル計算
            str_percentile = calculate_enhanced_percentile(str_values, self.percentile_lookback)
            egarch_percentile = calculate_enhanced_percentile(egarch_vol, self.percentile_lookback)
            vol_percentile = calculate_enhanced_percentile(returns_vol, self.percentile_lookback)
            
            # 9. 拡張実践的融合
            state, probability, raw_score = enhanced_practical_volatility_fusion(
                str_values, egarch_vol, returns_vol, range_expansion, regime_change,
                volatility_clustering, leverage_effect,
                str_percentile, egarch_percentile, vol_percentile,
                self.high_vol_threshold, self.low_vol_threshold
            )
            
            # 10. オプショナルスムージング
            if self.smoothing:
                # 状態のスムージング
                state_df = pd.DataFrame({'close': state.astype(np.float64)})
                smoothed_state_result = self.smoother.calculate(state_df)
                smoothed_state = (smoothed_state_result.values > 0.5).astype(np.int8)
                
                # 確率のスムージング
                prob_df = pd.DataFrame({'close': probability})
                smoothed_prob_result = self.smoother.calculate(prob_df)
                smoothed_probability = smoothed_prob_result.values
            else:
                smoothed_state = state
                smoothed_probability = probability
            
            # 結果作成
            result = EnhancedPracticalVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                raw_score=raw_score,
                str_values=str_values,
                egarch_volatility=egarch_vol,
                returns_volatility=returns_vol,
                range_expansion=range_expansion,
                regime_change=regime_change,
                volatility_clustering=volatility_clustering,
                leverage_effect=leverage_effect
            )
            
            # キャッシュ管理
            data_hash = self._get_data_hash(data)
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = smoothed_state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"拡張実践的ボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> EnhancedPracticalVolatilityStateResult:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return EnhancedPracticalVolatilityStateResult(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            raw_score=empty_array,
            str_values=empty_array,
            egarch_volatility=empty_array,
            returns_volatility=empty_array,
            range_expansion=empty_array,
            regime_change=empty_array,
            volatility_clustering=empty_array,
            leverage_effect=empty_array
        )
    
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
            
            params_sig = f"{self.str_period}_{self.egarch_period}_{self.high_vol_threshold}_{self.low_vol_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.str_period}_{self.egarch_period}"
    
    def get_state(self) -> Optional[np.ndarray]:
        """現在のボラティリティ状態を取得"""
        if self._values is not None:
            return self._values.astype(np.int8)
        return None
    
    def get_enhanced_analysis(self) -> Optional[Dict[str, np.ndarray]]:
        """拡張分析結果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return {
                'str_values': latest_result.str_values,
                'egarch_volatility': latest_result.egarch_volatility,
                'returns_volatility': latest_result.returns_volatility,
                'range_expansion': latest_result.range_expansion,
                'regime_change': latest_result.regime_change,
                'volatility_clustering': latest_result.volatility_clustering,
                'leverage_effect': latest_result.leverage_effect
            }
        return None
    
    def get_current_leverage_effect(self) -> Optional[float]:
        """現在のレバレッジ効果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.leverage_effect) > 0:
                return float(latest_result.leverage_effect[-1])
        return None
    
    def is_volatility_clustering(self) -> bool:
        """現在がボラティリティクラスタリング状態か"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.volatility_clustering) > 0:
                return float(latest_result.volatility_clustering[-1]) > 1.2
        return False
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        if self.smoothing:
            self.smoother.reset()