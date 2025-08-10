#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import jit
import traceback

# インポート処理
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .smoother.ultimate_smoother import UltimateSmoother
    from .smoother.frama import FRAMA
    from .mama import MAMA
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from smoother.ultimate_smoother import UltimateSmoother
    from smoother.frama import FRAMA
    from mama import MAMA
    # EhlersUnifiedDCは動的に条件付きインポート（実行時にファンクション内で処理）


class UltimateAdaptiveMAResult(NamedTuple):
    """UltimateAdaptiveMA計算結果"""
    values: np.ndarray                  # 最終適応型移動平均値
    base_ma: np.ndarray                 # ベースとなる移動平均値
    adaptive_factor: np.ndarray         # 適応ファクター（0-1）
    frama_values: np.ndarray            # FRAMA値
    frama_alpha: np.ndarray             # FRAMAアルファ値
    mama_values: np.ndarray             # MAMA値
    mama_alpha: np.ndarray              # MAMAアルファ値
    fractal_dimension: np.ndarray       # フラクタル次元（FRAMAから）
    cycle_period: np.ndarray            # サイクル期間（MAMAから）
    trend_strength: np.ndarray          # トレンド強度
    market_regime: np.ndarray           # マーケットレジーム（0:レンジ, 1:トレンド）
    noise_level: np.ndarray             # ノイズレベル
    responsiveness: np.ndarray          # 応答性指標


@jit(nopython=True)
def calculate_adaptive_factor_numba(
    frama_alpha: np.ndarray,
    mama_alpha: np.ndarray,
    fractal_dim: np.ndarray,
    cycle_period: np.ndarray,
    price_changes: np.ndarray,
    volatility: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    適応ファクターを計算する（Numba最適化版）
    
    Args:
        frama_alpha: FRAMAアルファ値
        mama_alpha: MAMAアルファ値
        fractal_dim: フラクタル次元
        cycle_period: サイクル期間
        price_changes: 価格変化
        volatility: ボラティリティ
    
    Returns:
        Tuple: (適応ファクター, トレンド強度, マーケットレジーム, ノイズレベル)
    """
    length = len(frama_alpha)
    adaptive_factor = np.zeros(length)
    trend_strength = np.zeros(length)
    market_regime = np.zeros(length)
    noise_level = np.zeros(length)
    
    for i in range(length):
        if i < 10:  # 初期期間
            adaptive_factor[i] = 0.5
            trend_strength[i] = 0.5
            market_regime[i] = 0
            noise_level[i] = 0.5
            continue
        
        # 1. フラクタル次元に基づく適応（FRAMAロジック）
        # フラクタル次元が1に近い（トレンド）ほど高い値、2に近い（レンジ）ほど低い値
        if not np.isnan(fractal_dim[i]) and fractal_dim[i] > 0:
            fractal_factor = max(0.0, min(1.0, (2.0 - fractal_dim[i])))
        else:
            fractal_factor = 0.5
        
        # 2. サイクル期間に基づく適応（MAMAロジック）
        # サイクル期間が短い（高頻度変動）ほど応答性を高める
        if not np.isnan(cycle_period[i]) and cycle_period[i] > 0:
            cycle_factor = max(0.1, min(1.0, 20.0 / cycle_period[i]))
        else:
            cycle_factor = 0.5
        
        # 3. 価格変化に基づく動的調整
        if i >= 5:
            recent_changes = np.abs(price_changes[i-5:i+1])
            avg_change = np.mean(recent_changes)
            change_factor = max(0.1, min(1.0, avg_change * 10))
        else:
            change_factor = 0.5
        
        # 4. ボラティリティに基づく調整
        if not np.isnan(volatility[i]) and volatility[i] > 0:
            vol_factor = max(0.1, min(1.0, volatility[i] * 5))
        else:
            vol_factor = 0.5
        
        # 5. トレンド強度の計算
        # フラクタル次元とサイクル期間から総合的にトレンド強度を判定
        trend_strength[i] = (fractal_factor * 0.6 + cycle_factor * 0.4)
        
        # 6. マーケットレジームの判定（0:レンジ, 1:トレンド）
        if trend_strength[i] > 0.6:
            market_regime[i] = 1  # トレンド
        else:
            market_regime[i] = 0  # レンジ
        
        # 7. ノイズレベルの計算
        noise_level[i] = 1.0 - trend_strength[i]
        
        # 8. 総合適応ファクターの計算
        # 各要素を重み付きで統合
        weights = np.array([0.3, 0.25, 0.25, 0.2])  # fractal, cycle, change, vol
        factors = np.array([fractal_factor, cycle_factor, change_factor, vol_factor])
        
        # 重み付き平均を計算
        weighted_factor = np.sum(factors * weights)
        
        # FRAMAとMAMAのアルファ値も考慮（ただし範囲を制限）
        frama_contrib = max(0.01, min(0.99, frama_alpha[i])) if not np.isnan(frama_alpha[i]) else 0.1
        mama_contrib = max(0.01, min(0.99, mama_alpha[i])) if not np.isnan(mama_alpha[i]) else 0.1
        
        # 最終適応ファクター（より保守的な重み付け）
        adaptive_factor[i] = (
            weighted_factor * 0.6 +
            frama_contrib * 0.2 +
            mama_contrib * 0.2
        )
        
        # より安全な範囲制限（0.05-0.8の範囲）
        adaptive_factor[i] = max(0.05, min(0.8, adaptive_factor[i]))
    
    return adaptive_factor, trend_strength, market_regime, noise_level


@jit(nopython=True)
def calculate_ultimate_adaptive_ma_numba(
    prices: np.ndarray,
    base_ma: np.ndarray,
    adaptive_factor: np.ndarray,
    min_alpha: float = 0.01,
    max_alpha: float = 0.99
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ultimate Adaptive MAを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        base_ma: ベース移動平均
        adaptive_factor: 適応ファクター
        min_alpha: 最小アルファ値
        max_alpha: 最大アルファ値
    
    Returns:
        Tuple: (Ultimate Adaptive MA値, 応答性指標)
    """
    length = len(prices)
    ultimate_ma = np.zeros(length)
    responsiveness = np.zeros(length)
    
    # 初期値設定 - ベースMAまたは価格で初期化
    if length > 0:
        if not np.isnan(base_ma[0]):
            ultimate_ma[0] = base_ma[0]
        else:
            ultimate_ma[0] = prices[0]
        responsiveness[0] = 0.5
    
    for i in range(1, length):
        # 適応アルファ値の計算
        alpha = max(min_alpha, min(max_alpha, adaptive_factor[i]))
        
        # Ultimate Adaptive MAの計算
        # 修正されたロジック：ベースMAを中心に適応的に調整
        if not np.isnan(base_ma[i]):
            # 標準的なEMAとベースMAの適応的ブレンド
            # alphaが高い場合：価格により敏感に反応
            # alphaが低い場合：ベースMAにより近く
            ema_component = alpha * prices[i] + (1.0 - alpha) * ultimate_ma[i-1]
            base_component = base_ma[i]
            
            # 適応ファクターに基づいてEMAとベースMAをブレンド
            # alphaが高い場合はEMA寄り、低い場合はベースMA寄り
            blend_ratio = alpha
            ultimate_ma[i] = blend_ratio * ema_component + (1.0 - blend_ratio) * base_component
        else:
            # ベースMAが無効な場合は標準的なEMA
            ultimate_ma[i] = alpha * prices[i] + (1.0 - alpha) * ultimate_ma[i-1]
        
        # 異常値チェック：価格から極端に乖離した場合は補正
        if not np.isnan(prices[i]) and abs(ultimate_ma[i] - prices[i]) > prices[i] * 0.5:
            # 50%以上乖離している場合は、価格とベースMAの中間値に補正
            if not np.isnan(base_ma[i]):
                ultimate_ma[i] = (prices[i] + base_ma[i]) * 0.5
            else:
                ultimate_ma[i] = prices[i]
        
        # 応答性指標の計算
        if i >= 2:
            price_momentum = abs(prices[i] - prices[i-1])
            ma_momentum = abs(ultimate_ma[i] - ultimate_ma[i-1])
            if price_momentum > 0:
                responsiveness[i] = min(1.0, ma_momentum / price_momentum)
            else:
                responsiveness[i] = responsiveness[i-1]
        else:
            responsiveness[i] = alpha
    
    return ultimate_ma, responsiveness


class UltimateAdaptiveMA(Indicator):
    """
    🚀 **Ultimate Adaptive Moving Average - 究極適応型移動平均**
    
    🎯 **革新的特徴:**
    - **FRAMAロジック**: フラクタル次元による適応
    - **MAMAロジック**: サイクル期間による適応  
    - **UltimateMAベース**: 6段階フィルタリングシステム
    - **動的適応**: 相場状況に応じたリアルタイム調整
    - **マルチファクター**: 複数の適応要素の統合
    - **ノイズ除去**: 高精度フィルタリング
    - **トレンド検出**: マーケットレジーム自動判定
    
    🏆 **適応メカニズム:**
    1. **フラクタル適応**: トレンド/レンジの強度判定
    2. **サイクル適応**: 市場サイクルに応じた期間調整
    3. **ボラティリティ適応**: 価格変動に応じた応答性調整
    4. **ノイズ適応**: ノイズレベルに応じたフィルタリング
    """
    
    def __init__(self,
                 # ベースMA設定
                 base_period: int = 21,
                 src_type: str = 'hlc3',
                 # FRAMA設定
                 frama_period: int = 16,
                 frama_fc: int = 1,
                 frama_sc: int = 198,
                 # MAMA設定
                 mama_fast_limit: float = 0.5,
                 mama_slow_limit: float = 0.05,
                 # Ultimate Smoother設定
                 smoother_period: float = 5.0,
                 # 適応パラメータ
                 adaptation_strength: float = 0.8,
                 min_alpha: float = 0.05,
                 max_alpha: float = 0.8,
                 volatility_period: int = 14,
                 # 動的期間パラメータ
                 use_dynamic_periods: bool = True,
                 cycle_detector_type: str = 'hody_e'):
        """
        コンストラクタ
        
        Args:
            base_period: ベース移動平均期間
            src_type: 価格ソース
            frama_period: FRAMA期間
            frama_fc: FRAMA高速定数
            frama_sc: FRAMA低速定数
            mama_fast_limit: MAMA高速リミット
            mama_slow_limit: MAMA低速リミット
            smoother_period: Ultimate Smoother期間
            adaptation_strength: 適応強度（0.1-2.0）
            min_alpha: 最小アルファ値
            max_alpha: 最大アルファ値
            volatility_period: ボラティリティ計算期間
            use_dynamic_periods: 動的期間使用
            cycle_detector_type: サイクル検出器タイプ
        """
        name = f"UltimateAdaptiveMA(base={base_period},frama={frama_period},adapt={adaptation_strength:.1f})"
        super().__init__(name)
        
        # パラメータ保存
        self.base_period = base_period
        self.src_type = src_type
        self.frama_period = frama_period if frama_period % 2 == 0 else frama_period + 1  # 偶数に調整
        self.frama_fc = frama_fc
        self.frama_sc = frama_sc
        self.mama_fast_limit = mama_fast_limit
        self.mama_slow_limit = mama_slow_limit
        self.smoother_period = smoother_period
        self.adaptation_strength = max(0.1, min(2.0, adaptation_strength))
        self.min_alpha = max(0.01, min(0.5, min_alpha))
        self.max_alpha = max(0.1, min(0.9, max_alpha))
        
        # min_alphaがmax_alphaより大きい場合の補正
        if self.min_alpha >= self.max_alpha:
            self.min_alpha = 0.05
            self.max_alpha = 0.8
        self.volatility_period = volatility_period
        self.use_dynamic_periods = use_dynamic_periods
        self.cycle_detector_type = cycle_detector_type
        
        # 子インジケーター初期化
        self.ultimate_smoother = UltimateSmoother(period=self.smoother_period, src_type=self.src_type)
        self.frama = FRAMA(
            period=self.frama_period,
            src_type=self.src_type,
            fc=self.frama_fc,
            sc=self.frama_sc,
            period_mode='dynamic' if self.use_dynamic_periods else 'fixed',
            cycle_detector_type=self.cycle_detector_type
        )
        self.mama = MAMA(
            fast_limit=self.mama_fast_limit,
            slow_limit=self.mama_slow_limit,
            src_type=self.src_type
        )
        
        # キャッシュ
        self._cache = {}
        self._result: Optional[UltimateAdaptiveMAResult] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateAdaptiveMAResult:
        """
        Ultimate Adaptive MAを計算する
        
        Args:
            data: 価格データ
            
        Returns:
            UltimateAdaptiveMAResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                return self._result
            
            # 価格データの取得
            prices = PriceSource.calculate_source(data, self.src_type)
            prices = np.asarray(prices, dtype=np.float64)
            
            data_length = len(prices)
            if data_length == 0:
                return self._create_empty_result()
            
            self.logger.info(f"🚀 Ultimate Adaptive MA計算開始 - データ長: {data_length}")
            
            # 1. ベース移動平均の計算（Ultimate Smoother使用）
            self.logger.debug("📊 ベース移動平均計算中...")
            smoother_result = self.ultimate_smoother.calculate(data)
            base_ma = smoother_result.values
            
            # 2. FRAMA計算
            self.logger.debug("🔍 FRAMA計算中...")
            frama_result = self.frama.calculate(data)
            frama_values = frama_result.values
            frama_alpha = frama_result.alpha
            fractal_dimension = frama_result.fractal_dimension
            
            # 3. MAMA計算
            self.logger.debug("🌊 MAMA計算中...")
            mama_result = self.mama.calculate(data)
            mama_values = mama_result.mama_values
            mama_alpha = mama_result.alpha_values
            cycle_period = mama_result.period_values
            
            # 4. ボラティリティ計算
            self.logger.debug("📈 ボラティリティ計算中...")
            volatility = self._calculate_volatility(prices, self.volatility_period)
            price_changes = np.abs(np.diff(np.concatenate([[prices[0]], prices])))
            
            # 5. 適応ファクター計算
            self.logger.debug("⚡ 適応ファクター計算中...")
            adaptive_factor, trend_strength, market_regime, noise_level = calculate_adaptive_factor_numba(
                frama_alpha * self.adaptation_strength,
                mama_alpha * self.adaptation_strength,
                fractal_dimension,
                cycle_period,
                price_changes,
                volatility
            )
            
            # 6. Ultimate Adaptive MA計算
            self.logger.debug("🎯 Ultimate Adaptive MA計算中...")
            ultimate_ma, responsiveness = calculate_ultimate_adaptive_ma_numba(
                prices, base_ma, adaptive_factor, self.min_alpha, self.max_alpha
            )
            
            # 結果作成
            result = UltimateAdaptiveMAResult(
                values=ultimate_ma,
                base_ma=base_ma,
                adaptive_factor=adaptive_factor,
                frama_values=frama_values,
                frama_alpha=frama_alpha,
                mama_values=mama_values,
                mama_alpha=mama_alpha,
                fractal_dimension=fractal_dimension,
                cycle_period=cycle_period,
                trend_strength=trend_strength,
                market_regime=market_regime,
                noise_level=noise_level,
                responsiveness=responsiveness
            )
            
            self._result = result
            self._cache[data_hash] = result
            
            # 統計情報
            trend_ratio = np.mean(market_regime[market_regime >= 0]) if len(market_regime[market_regime >= 0]) > 0 else 0
            avg_responsiveness = np.mean(responsiveness[~np.isnan(responsiveness)])
            
            self.logger.info(f"✅ Ultimate Adaptive MA計算完了")
            self.logger.info(f"📊 トレンド比率: {trend_ratio:.1%}, 平均応答性: {avg_responsiveness:.3f}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Ultimate Adaptive MA計算エラー: {error_msg}\n{stack_trace}")
            return self._create_empty_result()
    
    def _calculate_volatility(self, prices: np.ndarray, period: int) -> np.ndarray:
        """ボラティリティを計算する"""
        volatility = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            price_slice = prices[i-period:i]
            volatility[i] = np.std(price_slice) / np.mean(price_slice) if np.mean(price_slice) > 0 else 0
        
        # 初期期間は最初の有効値で埋める
        if period < len(volatility):
            volatility[:period] = volatility[period]
        
        return volatility
    
    def _create_empty_result(self) -> UltimateAdaptiveMAResult:
        """空の結果を作成する"""
        empty = np.array([], dtype=np.float64)
        return UltimateAdaptiveMAResult(
            values=empty, base_ma=empty, adaptive_factor=empty,
            frama_values=empty, frama_alpha=empty, mama_values=empty,
            mama_alpha=empty, fractal_dimension=empty, cycle_period=empty,
            trend_strength=empty, market_regime=empty, noise_level=empty,
            responsiveness=empty
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """Ultimate Adaptive MA値を取得する"""
        if self._result is not None:
            return self._result.values.copy()
        return None
    
    def get_base_ma(self) -> Optional[np.ndarray]:
        """ベース移動平均値を取得する"""
        if self._result is not None:
            return self._result.base_ma.copy()
        return None
    
    def get_adaptive_factor(self) -> Optional[np.ndarray]:
        """適応ファクターを取得する"""
        if self._result is not None:
            return self._result.adaptive_factor.copy()
        return None
    
    def get_trend_strength(self) -> Optional[np.ndarray]:
        """トレンド強度を取得する"""
        if self._result is not None:
            return self._result.trend_strength.copy()
        return None
    
    def get_market_regime(self) -> Optional[np.ndarray]:
        """マーケットレジームを取得する"""
        if self._result is not None:
            return self._result.market_regime.copy()
        return None
    
    def get_responsiveness(self) -> Optional[np.ndarray]:
        """応答性指標を取得する"""
        if self._result is not None:
            return self._result.responsiveness.copy()
        return None
    
    def get_frama_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """FRAMAデータを取得する（値、アルファ、フラクタル次元）"""
        if self._result is not None:
            return (
                self._result.frama_values.copy(),
                self._result.frama_alpha.copy(),
                self._result.fractal_dimension.copy()
            )
        return None
    
    def get_mama_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """MAMAデータを取得する（値、アルファ、サイクル期間）"""
        if self._result is not None:
            return (
                self._result.mama_values.copy(),
                self._result.mama_alpha.copy(),
                self._result.cycle_period.copy()
            )
        return None
    
    def reset(self) -> None:
        """状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if hasattr(self, 'ultimate_smoother'):
            self.ultimate_smoother.reset()
        if hasattr(self, 'frama'):
            self.frama.reset()
        if hasattr(self, 'mama'):
            self.mama.reset()
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュを計算する"""
        if isinstance(data, pd.DataFrame):
            data_hash = hash(data.values.tobytes())
        else:
            data_hash = hash(data.tobytes())
        
        params = f"{self.base_period}_{self.frama_period}_{self.adaptation_strength}_{self.src_type}"
        return f"{data_hash}_{hash(params)}"