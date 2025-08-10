#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper Adaptive Channel シグナルファクトリー

ハイパーアダプティブチャネルシグナルを簡単に作成するためのファクトリー関数
"""

from typing import Dict, Any, Optional, List, Tuple
from .breakout_entry import HyperAdaptiveChannelBreakoutEntrySignal


def create_hyper_frama_breakout_signal(
    period: int = 14,
    band_lookback: int = 1,
    multiplier_mode: str = "dynamic",
    fixed_multiplier: float = 2.5,
    # HyperFRAMA専用カスタマイズ
    alpha_multiplier: float = 0.5,
    enable_dynamic_adaptation: bool = True,
    fc_range: Tuple[float, float] = (1.0, 8.0),
    sc_range: Tuple[float, float] = (50.0, 250.0),
    period_range: Tuple[int, int] = (4, 88),
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    HyperFRAMA ベースのブレイクアウトシグナルを作成
    
    Args:
        period: 基本期間
        band_lookback: バンド参照期間
        multiplier_mode: 乗数モード
        fixed_multiplier: 固定乗数
        alpha_multiplier: アルファ調整係数
        enable_dynamic_adaptation: 動的適応有効化
        fc_range: FC範囲 (min, max)
        sc_range: SC範囲 (min, max)
        period_range: 期間範囲 (min, max)
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="hyper_frama",
        multiplier_mode=multiplier_mode,
        fixed_multiplier=fixed_multiplier,
        
        # HyperFRAMA設定
        hyper_frama_period=period,
        hyper_frama_alpha_multiplier=alpha_multiplier,
        hyper_frama_enable_indicator_adaptation=enable_dynamic_adaptation,
        hyper_frama_fc_min=fc_range[0],
        hyper_frama_fc_max=fc_range[1],
        hyper_frama_sc_min=sc_range[0],
        hyper_frama_sc_max=sc_range[1],
        hyper_frama_period_min=period_range[0],
        hyper_frama_period_max=period_range[1],
        
        **kwargs
    )


def create_ultimate_ma_breakout_signal(
    period: int = 14,
    band_lookback: int = 1,
    multiplier_mode: str = "dynamic",
    fixed_multiplier: float = 2.5,
    # UltimateMA専用カスタマイズ
    ultimate_smoother_period: float = 5.0,
    zero_lag_period: int = 21,
    realtime_window: int = 89,
    use_adaptive_kalman: bool = True,
    dynamic_periods: bool = True,
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    UltimateMA ベースのブレイクアウトシグナルを作成
    
    Args:
        period: 基本期間
        band_lookback: バンド参照期間
        multiplier_mode: 乗数モード
        fixed_multiplier: 固定乗数
        ultimate_smoother_period: アルティメットスムーザー期間
        zero_lag_period: ゼロラグ期間
        realtime_window: リアルタイムウィンドウ
        use_adaptive_kalman: 適応カルマンフィルター使用
        dynamic_periods: 動的期間使用
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="ultimate_ma",
        multiplier_mode=multiplier_mode,
        fixed_multiplier=fixed_multiplier,
        
        # UltimateMA設定
        ultimate_ma_ultimate_smoother_period=ultimate_smoother_period,
        ultimate_ma_zero_lag_period=zero_lag_period,
        ultimate_ma_realtime_window=realtime_window,
        ultimate_ma_use_adaptive_kalman=use_adaptive_kalman,
        ultimate_ma_zero_lag_period_mode="dynamic" if dynamic_periods else "fixed",
        ultimate_ma_realtime_window_mode="dynamic" if dynamic_periods else "fixed",
        ultimate_ma_src_type="hlc3",  # ukf_hlc3は問題があるため
        
        **kwargs
    )


def create_laguerre_breakout_signal(
    period: int = 14,
    band_lookback: int = 1,
    multiplier_mode: str = "dynamic",
    fixed_multiplier: float = 2.5,
    # Laguerre専用カスタマイズ
    gamma: float = 0.5,
    order: int = 4,
    src_type: str = "close",
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    LaguerreFilter ベースのブレイクアウトシグナルを作成
    
    Args:
        period: 基本期間
        band_lookback: バンド参照期間
        multiplier_mode: 乗数モード
        fixed_multiplier: 固定乗数
        gamma: ダンピングファクター
        order: フィルターオーダー
        src_type: 価格ソース
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="laguerre_filter",
        multiplier_mode=multiplier_mode,
        fixed_multiplier=fixed_multiplier,
        
        # Laguerre設定
        laguerre_gamma=gamma,
        laguerre_order=order,
        laguerre_src_type=src_type,
        
        **kwargs
    )


def create_super_smoother_breakout_signal(
    period: int = 14,
    band_lookback: int = 1,
    multiplier_mode: str = "dynamic",
    fixed_multiplier: float = 2.5,
    # SuperSmoother専用カスタマイズ
    length: int = 15,
    num_poles: int = 2,
    src_type: str = "hlc3",
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    SuperSmoother ベースのブレイクアウトシグナルを作成
    
    Args:
        period: 基本期間
        band_lookback: バンド参照期間
        multiplier_mode: 乗数モード
        fixed_multiplier: 固定乗数
        length: フィルター長
        num_poles: 極数
        src_type: 価格ソース
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="super_smoother",
        multiplier_mode=multiplier_mode,
        fixed_multiplier=fixed_multiplier,
        
        # SuperSmoother設定
        super_smoother_length=length,
        super_smoother_num_poles=num_poles,
        super_smoother_src_type=src_type,
        
        **kwargs
    )


def create_z_adaptive_breakout_signal(
    period: int = 14,
    band_lookback: int = 1,
    multiplier_mode: str = "dynamic", 
    fixed_multiplier: float = 2.5,
    # ZAdaptive専用カスタマイズ
    fast_period: int = 2,
    slow_period: int = 120,
    src_type: str = "hlc3",
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    ZAdaptiveMA ベースのブレイクアウトシグナルを作成
    
    Args:
        period: 基本期間
        band_lookback: バンド参照期間
        multiplier_mode: 乗数モード
        fixed_multiplier: 固定乗数
        fast_period: 高速期間
        slow_period: 低速期間
        src_type: 価格ソース
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="z_adaptive_ma",
        multiplier_mode=multiplier_mode,
        fixed_multiplier=fixed_multiplier,
        
        # ZAdaptive設定
        z_adaptive_fast_period=fast_period,
        z_adaptive_slow_period=slow_period,
        z_adaptive_src_type=src_type,
        
        **kwargs
    )


def create_custom_atr_breakout_signal(
    period: int = 14,
    band_lookback: int = 1,
    midline_smoother: str = "super_smoother",
    multiplier_mode: str = "dynamic",
    fixed_multiplier: float = 2.5,
    # X_ATR専用カスタマイズ
    atr_period: float = 12.0,
    tr_method: str = "str",
    enable_kalman: bool = True,
    smoother_type: str = "frama",
    enable_percentile_analysis: bool = True,
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    カスタム X_ATR 設定のブレイクアウトシグナルを作成
    
    Args:
        period: 基本期間
        band_lookback: バンド参照期間
        midline_smoother: ミッドラインスムーザー
        multiplier_mode: 乗数モード
        fixed_multiplier: 固定乗数
        atr_period: ATR期間
        tr_method: TR計算方法
        enable_kalman: カルマンフィルター有効化
        smoother_type: スムーサータイプ
        enable_percentile_analysis: パーセンタイル分析有効化
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother=midline_smoother,
        multiplier_mode=multiplier_mode,
        fixed_multiplier=fixed_multiplier,
        
        # X_ATR設定
        x_atr_period=atr_period,
        x_atr_tr_method=tr_method,
        x_atr_enable_kalman=enable_kalman,
        x_atr_smoother_type=smoother_type,
        x_atr_enable_percentile_analysis=enable_percentile_analysis,
        
        **kwargs
    )


def create_high_sensitivity_signal(
    period: int = 10,
    band_lookback: int = 1,
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    高感度シグナル（より多くのシグナルを生成）
    
    Args:
        period: 基本期間（短め）
        band_lookback: バンド参照期間
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="laguerre_filter",
        multiplier_mode="dynamic",
        fixed_multiplier=1.8,  # 低めの乗数
        
        # 高感度設定
        laguerre_gamma=0.8,  # 高感度
        x_atr_period=8.0,    # 短いATR期間
        hyper_er_period=6,   # 短いER期間
        
        **kwargs
    )


def create_low_sensitivity_signal(
    period: int = 20,
    band_lookback: int = 2,
    **kwargs
) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    低感度シグナル（より確実なシグナルのみ）
    
    Args:
        period: 基本期間（長め）
        band_lookback: バンド参照期間（長め）
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    """
    
    return HyperAdaptiveChannelBreakoutEntrySignal(
        band_lookback=band_lookback,
        period=period,
        midline_smoother="hyper_frama", 
        multiplier_mode="dynamic",
        fixed_multiplier=3.0,  # 高めの乗数
        
        # 低感度設定
        hyper_frama_alpha_multiplier=0.3,  # 低感度
        x_atr_period=18.0,  # 長いATR期間
        hyper_er_period=15, # 長いER期間
        
        **kwargs
    )


# プリセット関数の辞書
SIGNAL_PRESETS = {
    "hyper_frama": create_hyper_frama_breakout_signal,
    "ultimate_ma": create_ultimate_ma_breakout_signal,
    "laguerre": create_laguerre_breakout_signal,
    "super_smoother": create_super_smoother_breakout_signal,
    "z_adaptive": create_z_adaptive_breakout_signal,
    "custom_atr": create_custom_atr_breakout_signal,
    "high_sensitivity": create_high_sensitivity_signal,
    "low_sensitivity": create_low_sensitivity_signal,
}


def create_signal_by_preset(preset_name: str, **kwargs) -> HyperAdaptiveChannelBreakoutEntrySignal:
    """
    プリセット名でシグナルを作成
    
    Args:
        preset_name: プリセット名
        **kwargs: 追加パラメータ
    
    Returns:
        HyperAdaptiveChannelBreakoutEntrySignal
    
    Raises:
        ValueError: 無効なプリセット名
    """
    
    if preset_name not in SIGNAL_PRESETS:
        available = ", ".join(SIGNAL_PRESETS.keys())
        raise ValueError(f"無効なプリセット名: {preset_name}. 利用可能: {available}")
    
    return SIGNAL_PRESETS[preset_name](**kwargs)