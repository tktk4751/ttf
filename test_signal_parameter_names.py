#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from signals.implementations.ehlers_instantaneous_trendline.entry import (
    EhlersInstantaneousTrendlinePositionEntrySignal,
    EhlersInstantaneousTrendlineCrossoverEntrySignal
)

def test_signal_name_display():
    """シグナル名の表示テスト"""
    print("=== Ehlers Instantaneous Trendlineシグナル名表示テスト ===")
    
    # 1. 基本設定（平滑化なし）
    signal1 = EhlersInstantaneousTrendlinePositionEntrySignal(
        alpha=0.07,
        src_type='hl2',
        enable_hyper_er_adaptation=False,
        smoothing_mode='none'
    )
    print(f"1. 基本設定: {signal1.name}")
    
    # 2. HyperER動的適応
    signal2 = EhlersInstantaneousTrendlinePositionEntrySignal(
        alpha=0.07,
        src_type='hl2',
        enable_hyper_er_adaptation=True,
        hyper_er_period=14,
        hyper_er_midline_period=100,
        alpha_min=0.04,
        alpha_max=0.15,
        smoothing_mode='none'
    )
    print(f"2. HyperER動的適応: {signal2.name}")
    
    # 3. カルマンフィルター平滑化
    signal3 = EhlersInstantaneousTrendlinePositionEntrySignal(
        alpha=0.07,
        src_type='hl2',
        enable_hyper_er_adaptation=False,
        smoothing_mode='kalman',
        kalman_filter_type='simple'
    )
    print(f"3. カルマン平滑化: {signal3.name}")
    
    # 4. アルティメットスムーサー平滑化
    signal4 = EhlersInstantaneousTrendlinePositionEntrySignal(
        alpha=0.07,
        src_type='hl2',
        enable_hyper_er_adaptation=False,
        smoothing_mode='ultimate',
        ultimate_smoother_period=10
    )
    print(f"4. アルティメット平滑化: {signal4.name}")
    
    # 5. カルマン + アルティメットスムーサー組み合わせ
    signal5 = EhlersInstantaneousTrendlinePositionEntrySignal(
        alpha=0.07,
        src_type='hl2',
        enable_hyper_er_adaptation=False,
        smoothing_mode='kalman_ultimate',
        kalman_filter_type='simple',
        ultimate_smoother_period=10
    )
    print(f"5. カルマン+アルティメット: {signal5.name}")
    
    # 6. フル機能（HyperER + カルマン+アルティメット）
    signal6 = EhlersInstantaneousTrendlinePositionEntrySignal(
        alpha=0.07,
        src_type='hl2',
        enable_hyper_er_adaptation=True,
        hyper_er_period=14,
        hyper_er_midline_period=100,
        alpha_min=0.04,
        alpha_max=0.15,
        smoothing_mode='kalman_ultimate',
        kalman_filter_type='unscented',
        ultimate_smoother_period=10
    )
    print(f"6. フル機能: {signal6.name}")
    
    # 7. クロスオーバーシグナル
    signal7 = EhlersInstantaneousTrendlineCrossoverEntrySignal(
        alpha=0.07,
        src_type='close',
        enable_hyper_er_adaptation=True,
        hyper_er_period=12,
        hyper_er_midline_period=80,
        alpha_min=0.05,
        alpha_max=0.12,
        smoothing_mode='kalman_ultimate',
        kalman_filter_type='extended',
        ultimate_smoother_period=8
    )
    print(f"7. クロスオーバー（フル機能）: {signal7.name}")

if __name__ == "__main__":
    test_signal_name_display()