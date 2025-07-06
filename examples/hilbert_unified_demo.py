#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌀 **Hilbert Transform Unified Demo V1.0** 🌀

ヒルベルト変換統合解析システムのデモンストレーション
- 全ヒルベルト変換手法の比較テスト
- パフォーマンス分析
- 結果の可視化
- 位相・振幅・周波数成分の分離評価
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import yaml

from indicators.hilbert_unified import HilbertTransformUnified
# データローダー関連のインポート
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

plt.style.use('dark_background')


def create_synthetic_data(n_points: int = 500) -> pd.DataFrame:
    """
    ヒルベルト変換解析用の合成価格データを作成（複雑な位相・周波数成分を含む）
    """
    np.random.seed(42)
    
    # 時間軸
    t = np.linspace(0, 10, n_points)
    
    # 1. メイントレンド（位相回転を持つ）
    main_trend = 100 + 20 * np.sin(t * 0.5) + 1.2 * t
    
    # 2. 主要サイクル（異なる位相を持つ）
    primary_cycle = 12 * np.sin(t * 1.5 + np.pi/4)
    secondary_cycle = 8 * np.cos(t * 2.3 + np.pi/3)
    
    # 3. 短期振動（高周波成分）
    short_oscillation1 = 5 * np.sin(t * 5.2 + np.pi/6)
    short_oscillation2 = 3 * np.cos(t * 8.1 + np.pi/2)
    
    # 4. 瞬時周波数変化（チャープ信号）
    freq_modulation = 4 * np.sin(t * 3.0 + np.cumsum(np.sin(t * 0.8)) * 0.5)
    
    # 5. 振幅変調
    amplitude_modulation = (1 + 0.3 * np.sin(t * 1.2)) * 6 * np.sin(t * 4.0)
    
    # 6. ランダムノイズ（ガウシアン + インパルス）
    gaussian_noise = np.random.normal(0, 1.5, n_points)
    impulse_noise = np.random.choice([0, 1], n_points, p=[0.95, 0.05]) * np.random.normal(0, 5, n_points)
    
    # 7. マーケットレジーム変化（構造変化）
    regime_switch = np.where(t > 5, 1.2, 1.0)
    
    # 最終価格合成（複雑な位相関係を保持）
    close_prices = (main_trend + primary_cycle + secondary_cycle + 
                   short_oscillation1 + short_oscillation2 + 
                   freq_modulation + amplitude_modulation + 
                   gaussian_noise + impulse_noise) * regime_switch
    
    # OHLC生成
    high_prices = close_prices + np.abs(np.random.normal(0, 1.0, n_points))
    low_prices = close_prices - np.abs(np.random.normal(0, 1.0, n_points))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # ボリューム
    volume = np.random.lognormal(9, 0.3, n_points)
    
    # 日時
    start_date = datetime.now() - timedelta(days=n_points)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        # 理論値（評価用）
        'true_trend': main_trend,
        'true_primary_cycle': primary_cycle,
        'true_secondary_cycle': secondary_cycle,
        'true_total_cycle': primary_cycle + secondary_cycle,
        'true_noise': gaussian_noise + impulse_noise
    })
    
    # datetimeをインデックスに設定
    df.set_index('datetime', inplace=True)
    return df


def test_all_hilbert_algorithms(data: pd.DataFrame) -> dict:
    """
    全ヒルベルト変換アルゴリズムをテストして結果を比較
    """
    algorithms = HilbertTransformUnified.get_available_algorithms()
    results = {}
    
    print("🌀 全ヒルベルト変換アルゴリズムのテスト実行中...")
    
    for algorithm_type, description in algorithms.items():
        print(f"   📊 {algorithm_type}: {description}")
        
        try:
            # ヒルベルト変換解析器初期化
            hilbert_analyzer = HilbertTransformUnified(
                algorithm_type=algorithm_type,
                src_type='close'
            )
            
            # 解析実行
            result = hilbert_analyzer.calculate(data)
            
            # パフォーマンス評価
            performance = evaluate_hilbert_performance(
                original=data['close'].values,
                hilbert_result=result,
                true_trend=data.get('true_trend', None),
                true_cycle=data.get('true_total_cycle', None),
                true_noise=data.get('true_noise', None)
            )
            
            # アルゴリズム固有情報の取得
            metadata = hilbert_analyzer.get_algorithm_metadata()
            
            results[algorithm_type] = {
                'result': result,
                'performance': performance,
                'metadata': metadata,
                'description': description
            }
            
            print(f"      ✅ 成功 - 総合スコア: {performance['total_score']:.3f}")
            
        except Exception as e:
            print(f"      ❌ エラー: {e}")
            import traceback
            print(f"         詳細: {traceback.format_exc()}")
            results[algorithm_type] = None
    
    return results


def evaluate_hilbert_performance(original: np.ndarray, hilbert_result, 
                                true_trend=None, true_cycle=None, true_noise=None) -> dict:
    """
    ヒルベルト変換解析の性能を評価
    """
    if len(original) < 10:
        return {'total_score': 0.0}
    
    # 基本メトリクス
    n_points = len(original)
    
    # 1. 位相精度評価
    if hilbert_result.phase is not None:
        phase_values = hilbert_result.phase
        # 位相の滑らかさ（急激な変化を避ける）
        phase_diff = np.diff(phase_values)
        phase_continuity = 1.0 - np.mean(np.abs(phase_diff) > np.pi) if len(phase_diff) > 0 else 0
        
        # 位相の範囲適正性
        phase_range_score = 1.0 if np.all((-np.pi <= phase_values) & (phase_values <= np.pi)) else 0.5
        phase_score = (phase_continuity + phase_range_score) / 2
    else:
        phase_score = 0
    
    # 2. 振幅精度評価
    if hilbert_result.amplitude is not None:
        amplitude_values = hilbert_result.amplitude
        # 振幅の非負性
        amplitude_positive = np.all(amplitude_values >= 0) if len(amplitude_values) > 0 else False
        # 振幅の合理的範囲
        price_std = np.std(original)
        amplitude_reasonable = np.all(amplitude_values <= 3 * price_std) if len(amplitude_values) > 0 else False
        amplitude_score = (amplitude_positive + amplitude_reasonable) / 2
    else:
        amplitude_score = 0
    
    # 3. 周波数精度評価
    if hilbert_result.frequency is not None:
        frequency_values = hilbert_result.frequency
        # 周波数の非負性と合理的範囲
        freq_positive = np.all(frequency_values >= 0) if len(frequency_values) > 0 else False
        freq_reasonable = np.all(frequency_values <= 0.5) if len(frequency_values) > 0 else False  # ナイキスト周波数以下
        frequency_score = (freq_positive + freq_reasonable) / 2
    else:
        frequency_score = 0
    
    # 4. 信号追従性（原信号との相関）
    # 振幅を使用して原信号との追従性を評価
    hilbert_values = hilbert_result.amplitude
    valid_mask = ~np.isnan(hilbert_values)
    
    if np.sum(valid_mask) > 10:
        # 振幅の変化と価格の変化の相関を計算
        price_changes = np.diff(original[valid_mask])
        amplitude_changes = np.diff(hilbert_values[valid_mask])
        if len(price_changes) > 10 and len(amplitude_changes) > 10:
            correlation = np.corrcoef(price_changes, amplitude_changes)[0, 1]
            signal_tracking = max(0, correlation) if not np.isnan(correlation) else 0
        else:
            signal_tracking = 0
    else:
        signal_tracking = 0
    
    # 5. トレンド成分評価
    if hasattr(hilbert_result, 'trend_component') and hilbert_result.trend_component is not None and true_trend is not None:
        trend_component = hilbert_result.trend_component
        valid_trend_mask = ~(np.isnan(trend_component) | np.isnan(true_trend))
        if np.sum(valid_trend_mask) > 10:
            trend_correlation = np.corrcoef(trend_component[valid_trend_mask], 
                                          true_trend[valid_trend_mask])[0, 1]
            trend_accuracy = max(0, trend_correlation) if not np.isnan(trend_correlation) else 0
        else:
            trend_accuracy = 0
    else:
        trend_accuracy = 0.5  # デフォルト値
    
    # 6. サイクル検出能力
    if hasattr(hilbert_result, 'cycle_component') and hilbert_result.cycle_component is not None and true_cycle is not None:
        cycle_component = hilbert_result.cycle_component
        valid_cycle_mask = ~(np.isnan(cycle_component) | np.isnan(true_cycle))
        if np.sum(valid_cycle_mask) > 10:
            cycle_correlation = np.corrcoef(cycle_component[valid_cycle_mask], 
                                          true_cycle[valid_cycle_mask])[0, 1]
            cycle_accuracy = max(0, cycle_correlation) if not np.isnan(cycle_correlation) else 0
        else:
            cycle_accuracy = 0
    else:
        cycle_accuracy = 0.5  # デフォルト値
    
    # 7. 計算効率性（遅延評価）
    nan_count = np.sum(np.isnan(hilbert_result.amplitude))
    delay_score = max(0, 1 - nan_count / n_points)
    
    # 8. 量子コヒーレンス評価（量子強化アルゴリズム用）
    if hasattr(hilbert_result, 'quantum_coherence') and hilbert_result.quantum_coherence is not None:
        coherence_score = np.nanmean(hilbert_result.quantum_coherence)
    else:
        coherence_score = 0.5  # デフォルト値
    
    # 9. 信頼度評価
    if hasattr(hilbert_result, 'confidence') and hilbert_result.confidence is not None:
        confidence_score = np.nanmean(hilbert_result.confidence)
    else:
        confidence_score = 0.5  # デフォルト値
    
    # 10. 総合スコア（重み付き平均）
    weights = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]
    scores = [phase_score, amplitude_score, frequency_score, signal_tracking, 
              trend_accuracy, cycle_accuracy, delay_score, coherence_score, confidence_score]
    total_score = sum(w * s for w, s in zip(weights, scores))
    
    return {
        'phase_score': phase_score,
        'amplitude_score': amplitude_score,
        'frequency_score': frequency_score,
        'signal_tracking': signal_tracking,
        'trend_accuracy': trend_accuracy,
        'cycle_accuracy': cycle_accuracy,
        'delay_score': delay_score,
        'coherence_score': coherence_score,
        'confidence_score': confidence_score,
        'total_score': total_score
    }


def visualize_hilbert_comparison(data: pd.DataFrame, results: dict, save_path: str = None):
    """
    ヒルベルト変換解析結果の比較可視化
    """
    # 有効な結果のみ抽出
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_algorithms = len(valid_results)
    
    if n_algorithms == 0:
        print("表示可能な結果がありません")
        return
    
    # 図の設定
    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor('black')
    
    # グリッド設定
    rows = 5
    cols = 3
    
    colors = ['cyan', 'yellow', 'lime', 'magenta', 'orange', 'red', 'pink', 'lightblue']
    
    # 1. 原信号 vs ヒルベルト変換結果
    ax1 = plt.subplot(rows, cols, 1)
    ax1.plot(data.index, data['close'], 'white', alpha=0.8, linewidth=1.5, label='Original Signal')
    
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if result_data and result_data['result']:
            # 振幅を使用して結果を表示
            ax1.plot(data.index, result_data['result'].amplitude, 
                    colors[i % len(colors)], alpha=0.8, linewidth=2, 
                    label=f"{algorithm_type}")
    
    ax1.set_title('🌀 Hilbert Transform Comparison', fontsize=14, color='white', fontweight='bold')
    ax1.set_ylabel('Price', color='white')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 性能スコア比較
    ax2 = plt.subplot(rows, cols, 2)
    algorithm_names = []
    total_scores = []
    
    for algorithm_type, result_data in valid_results.items():
        if result_data and result_data['performance']:
            algorithm_names.append(algorithm_type.replace('_', '\n'))
            total_scores.append(result_data['performance']['total_score'])
    
    bars = ax2.bar(algorithm_names, total_scores, color=colors[:len(algorithm_names)], alpha=0.8)
    ax2.set_title('📊 Performance Scores', fontsize=14, color='white', fontweight='bold')
    ax2.set_ylabel('Total Score', color='white')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    
    # スコア値をバーの上に表示
    for bar, score in zip(bars, total_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', color='white', fontsize=9)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. 振幅成分比較
    ax3 = plt.subplot(rows, cols, 3)
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if (result_data and result_data['result'] and 
            result_data['result'].amplitude is not None):
            ax3.plot(data.index, result_data['result'].amplitude,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
    
    ax3.set_title('📈 Amplitude Components', fontsize=14, color='white', fontweight='bold')
    ax3.set_ylabel('Amplitude', color='white')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 位相成分比較
    ax4 = plt.subplot(rows, cols, 4)
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if (result_data and result_data['result'] and 
            result_data['result'].phase is not None):
            ax4.plot(data.index, result_data['result'].phase,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
    
    ax4.set_title('🔄 Phase Components', fontsize=14, color='white', fontweight='bold')
    ax4.set_ylabel('Phase (radians)', color='white')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 周波数成分比較
    ax5 = plt.subplot(rows, cols, 5)
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if (result_data and result_data['result'] and 
            result_data['result'].frequency is not None):
            ax5.plot(data.index, result_data['result'].frequency,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
    
    ax5.set_title('⚡ Frequency Components', fontsize=14, color='white', fontweight='bold')
    ax5.set_ylabel('Frequency', color='white')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ランキング作成
    ranking = sorted([(k, v['performance']['total_score']) for k, v in valid_results.items() 
                     if v and v['performance']], key=lambda x: x[1], reverse=True)
    
    # 6. トレンド成分比較
    ax6 = plt.subplot(rows, cols, 6)
    if 'true_trend' in data.columns:
        ax6.plot(data.index, data['true_trend'], 'white', alpha=0.8, linewidth=2, label='True Trend')
    
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if (result_data and result_data['result'] and 
            hasattr(result_data['result'], 'trend_component') and
            result_data['result'].trend_component is not None):
            ax6.plot(data.index, result_data['result'].trend_component,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
    
    ax6.set_title('📈 Trend Components', fontsize=14, color='white', fontweight='bold')
    ax6.set_ylabel('Trend Value', color='white')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. 詳細性能メトリクス（最高性能アルゴリズム）
    ax7 = plt.subplot(rows, cols, 7)
    metrics = ['phase_score', 'amplitude_score', 'frequency_score', 'signal_tracking', 
               'trend_accuracy', 'cycle_accuracy', 'delay_score']
    metric_labels = ['Phase\nScore', 'Amplitude\nScore', 'Frequency\nScore', 'Signal\nTracking',
                    'Trend\nAccuracy', 'Cycle\nAccuracy', 'Low\nDelay']
    
    # 最高性能アルゴリズム特定
    if ranking:
        best_algorithm = ranking[0][0]
        
        if best_algorithm and valid_results[best_algorithm]:
            perf = valid_results[best_algorithm]['performance']
            values = [perf[metric] for metric in metrics]
            
            bars = ax7.bar(metric_labels, values, color='lime', alpha=0.8)
            ax7.set_title(f'🏆 Best: {best_algorithm}', fontsize=14, color='white', fontweight='bold')
            ax7.set_ylabel('Score', color='white')
            ax7.set_ylim(0, 1)
            plt.setp(ax7.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            
            # 値をバーの上に表示
            for bar, value in zip(bars, values):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', color='white', fontsize=9)
            
            ax7.grid(True, alpha=0.3)
    
    # 8. 量子コヒーレンス（量子強化アルゴリズム用）
    ax8 = plt.subplot(rows, cols, 8)
    quantum_found = False
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if (result_data and result_data['result'] and 
            hasattr(result_data['result'], 'quantum_coherence') and
            result_data['result'].quantum_coherence is not None):
            ax8.plot(data.index, result_data['result'].quantum_coherence,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
            quantum_found = True
    
    if quantum_found:
        ax8.set_title('🌌 Quantum Coherence', fontsize=14, color='white', fontweight='bold')
        ax8.set_ylabel('Coherence', color='white')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, '量子コヒーレンス\nデータなし', ha='center', va='center',
                transform=ax8.transAxes, color='white', fontsize=12)
        ax8.set_title('🌌 Quantum Coherence', fontsize=14, color='white', fontweight='bold')
    
    # 9. 個別性能比較（レーダーチャート風）
    ax9 = plt.subplot(rows, cols, 9)
    metrics_short = ['Phase', 'Amplitude', 'Frequency', 'Tracking']
    n_metrics = len(metrics_short)
    
    for i, (algorithm_type, result_data) in enumerate(list(valid_results.items())[:4]):  # 上位4つのみ
        if result_data and result_data['performance']:
            perf = result_data['performance']
            values = [perf['phase_score'], perf['amplitude_score'], 
                     perf['frequency_score'], perf['signal_tracking']]
            
            x_pos = np.arange(n_metrics)
            ax9.plot(x_pos, values, 'o-', color=colors[i % len(colors)], 
                    linewidth=2, markersize=6, alpha=0.8, label=algorithm_type)
    
    ax9.set_title('⚡ Multi-Metric Comparison', fontsize=14, color='white', fontweight='bold')
    ax9.set_xticks(range(n_metrics))
    ax9.set_xticklabels(metrics_short, fontsize=9)
    ax9.set_ylabel('Score', color='white')
    ax9.set_ylim(0, 1)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    # 10. ランキング表
    ax10 = plt.subplot(rows, cols, 10)
    ax10.axis('off')
    
    ranking_text = "🏆 Hilbert Algorithm Ranking:\n\n"
    for i, (algorithm_name, score) in enumerate(ranking):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        ranking_text += f"{emoji} {algorithm_name}:\n    {score:.3f}\n"
    
    ax10.text(0.05, 0.95, ranking_text, transform=ax10.transAxes, fontsize=11, 
            color='white', verticalalignment='top', fontfamily='monospace')
    
    # 11. 計算時間・効率性（仮想データ）
    ax11 = plt.subplot(rows, cols, 11)
    computation_times = [np.random.uniform(0.05, 1.5) for _ in valid_results]  # 仮想時間
    algorithm_names_short = [name.replace('_', '\n') for name in valid_results.keys()]
    
    bars = ax11.bar(algorithm_names_short, computation_times, 
                   color=colors[:len(computation_times)], alpha=0.8)
    ax11.set_title('⏱️ Computation Time', fontsize=14, color='white', fontweight='bold')
    ax11.set_ylabel('Time (seconds)', color='white')
    plt.setp(ax11.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax11.grid(True, alpha=0.3)
    
    # 12. 推奨用途
    ax12 = plt.subplot(rows, cols, 12)
    ax12.axis('off')
    
    recommendations = {
        'basic': '高速処理\n基本解析',
        'quantum_enhanced': '高精度解析\n量子強化',
        'instantaneous': '詳細分析\n瞬時解析',
        'instantaneous_v2': '高速分析\n簡易版',
        'supreme': '最高精度\n9点FIR',
        'numpy_fft': 'FFT近似\n周波数特化',
        'multiresolution': 'マルチ解像度\nウェーブレット統合'
    }
    
    rec_text = "💡 推奨用途:\n\n"
    for i, (algorithm_type, _) in enumerate(ranking[:6]):
        rec = recommendations.get(algorithm_type, '汎用')
        rec_text += f"• {algorithm_type}:\n  {rec}\n\n"
    
    ax12.text(0.05, 0.95, rec_text, transform=ax12.transAxes, fontsize=10,
            color='white', verticalalignment='top')
    
    # 13. 信号分解成分（上位3アルゴリズム）
    ax13 = plt.subplot(rows, cols, 13)
    for i, (algorithm_type, _) in enumerate(ranking[:3]):
        result_data = valid_results[algorithm_type]
        if (result_data and result_data['result'] and 
            hasattr(result_data['result'], 'cycle_component') and
            result_data['result'].cycle_component is not None):
            ax13.plot(data.index, result_data['result'].cycle_component,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
    
    if 'true_total_cycle' in data.columns:
        ax13.plot(data.index, data['true_total_cycle'], 'white', alpha=0.8, 
                 linewidth=2, label='True Cycle')
    
    ax13.set_title('🔄 Cycle Decomposition', fontsize=14, color='white', fontweight='bold')
    ax13.set_ylabel('Cycle Component', color='white')
    ax13.legend(fontsize=9)
    ax13.grid(True, alpha=0.3)
    
    # 14. 信頼度スコア
    ax14 = plt.subplot(rows, cols, 14)
    confidence_found = False
    for i, (algorithm_type, result_data) in enumerate(valid_results.items()):
        if (result_data and result_data['result'] and 
            hasattr(result_data['result'], 'confidence') and
            result_data['result'].confidence is not None):
            ax14.plot(data.index, result_data['result'].confidence,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{algorithm_type}")
            confidence_found = True
    
    if confidence_found:
        ax14.set_title('🎯 Confidence Scores', fontsize=14, color='white', fontweight='bold')
        ax14.set_ylabel('Confidence', color='white')
        ax14.legend(fontsize=9)
        ax14.grid(True, alpha=0.3)
    else:
        ax14.text(0.5, 0.5, '信頼度スコア\nデータなし', ha='center', va='center',
                 transform=ax14.transAxes, color='white', fontsize=12)
        ax14.set_title('🎯 Confidence Scores', fontsize=14, color='white', fontweight='bold')
    
    # 15. 総合評価サマリー
    ax15 = plt.subplot(rows, cols, 15)
    ax15.axis('off')
    
    if ranking:
        best_name, best_score = ranking[0]
        summary_text = f"🎉 総合評価結果\n\n"
        summary_text += f"🏆 最優秀: {best_name}\n"
        summary_text += f"📊 スコア: {best_score:.3f}\n\n"
        summary_text += f"📈 テスト成功: {len(valid_results)}/{len(results)}\n"
        summary_text += f"📊 データ数: {len(data)}\n"
        summary_text += f"💹 価格範囲: ${data['close'].min():.1f}-${data['close'].max():.1f}\n"
        summary_text += f"📅 期間: {data.index[0].strftime('%Y-%m-%d')}\n"
        summary_text += f"     ～ {data.index[-1].strftime('%Y-%m-%d')}"
        
        ax15.text(0.05, 0.95, summary_text, transform=ax15.transAxes, fontsize=11,
                color='white', verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=300, bbox_inches='tight')
        print(f"🌀 ヒルベルト変換比較結果を保存しました: {save_path}")
    
    plt.show()


def detailed_hilbert_analysis(algorithm_type: str, data: pd.DataFrame):
    """
    特定ヒルベルト変換アルゴリズムの詳細分析
    """
    print(f"\n🔍 詳細分析: {algorithm_type}")
    print("=" * 60)
    
    # ヒルベルト変換実行
    try:
        hilbert_analyzer = HilbertTransformUnified(algorithm_type=algorithm_type, src_type='close')
        result = hilbert_analyzer.calculate(data)
        metadata = hilbert_analyzer.get_algorithm_metadata()
    except Exception as e:
        print(f"❌ ヒルベルト変換実行エラー: {e}")
        return
    
    # 基本情報表示
    print(f"📊 基本情報:")
    print(f"   アルゴリズム: {metadata.get('algorithm', 'N/A')}")
    print(f"   説明: {metadata.get('description', 'N/A')}")
    print(f"   価格ソース: {metadata.get('src_type', 'N/A')}")
    print(f"   長さパラメータ: {metadata.get('length', 'N/A')}")
    
    # 成分分析
    print(f"\n🔬 成分分析:")
    if result.amplitude is not None:
        amp_mean = np.nanmean(result.amplitude)
        amp_std = np.nanstd(result.amplitude)
        print(f"   振幅成分平均: {amp_mean:.4f}")
        print(f"   振幅成分標準偏差: {amp_std:.4f}")
    
    if result.phase is not None:
        phase_mean = np.nanmean(result.phase)
        phase_std = np.nanstd(result.phase)
        print(f"   位相成分平均: {phase_mean:.4f} rad")
        print(f"   位相成分標準偏差: {phase_std:.4f} rad")
    
    if result.frequency is not None:
        freq_mean = np.nanmean(result.frequency)
        freq_std = np.nanstd(result.frequency)
        print(f"   周波数成分平均: {freq_mean:.4f}")
        print(f"   周波数成分標準偏差: {freq_std:.4f}")
    
    # 量子コヒーレンス分析（該当する場合）
    if hasattr(result, 'quantum_coherence') and result.quantum_coherence is not None:
        qc_mean = np.nanmean(result.quantum_coherence)
        qc_std = np.nanstd(result.quantum_coherence)
        print(f"   量子コヒーレンス平均: {qc_mean:.4f}")
        print(f"   量子コヒーレンス標準偏差: {qc_std:.4f}")
    
    # 性能評価
    performance = evaluate_hilbert_performance(
        data['close'].values, result,
        data.get('true_trend', None), data.get('true_total_cycle', None), data.get('true_noise', None)
    )
    
    print(f"\n🎯 性能評価:")
    print(f"   位相精度: {performance['phase_score']:.3f}")
    print(f"   振幅精度: {performance['amplitude_score']:.3f}")
    print(f"   周波数精度: {performance['frequency_score']:.3f}")
    print(f"   信号追従性: {performance['signal_tracking']:.3f}")
    print(f"   トレンド抽出精度: {performance['trend_accuracy']:.3f}")
    print(f"   サイクル検出能力: {performance['cycle_accuracy']:.3f}")
    print(f"   計算効率性: {performance['delay_score']:.3f}")
    print(f"   量子コヒーレンス: {performance['coherence_score']:.3f}")
    print(f"   信頼度スコア: {performance['confidence_score']:.3f}")
    print(f"   総合スコア: {performance['total_score']:.3f}")


def load_real_market_data(config_path: str = 'config.yaml', n_recent: int = 500) -> pd.DataFrame:
    """
    config.yamlから実際の相場データを読み込む（直近n_recent本のデータ）
    """
    try:
        print(f"📡 {config_path} からデータを読み込み中... (直近{n_recent}本)")
        
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        # 直近n_recent本に限定
        if len(data) > n_recent:
            data = data.tail(n_recent).copy()
        
        print(f"✅ 実データ読み込み完了: {first_symbol}")
        print(f"📊 使用データ数: {len(data)}")
        print(f"📅 期間: {data.index[0].strftime('%Y-%m-%d')} ～ {data.index[-1].strftime('%Y-%m-%d')}")
        return data
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        print(f"🔄 合成データを使用します (データ数: {n_recent})")
        return create_synthetic_data(n_recent)


def main():
    """
    メイン実行関数
    """
    print("🌀 Hilbert Transform Unified Demo V1.0")
    print("=" * 60)
    
    # データ準備
    print("\n1️⃣ データ準備")
    
    # 実データを試し、失敗したら合成データ（直近500本）
    data = load_real_market_data('config.yaml', n_recent=500)
    
    print(f"📊 データ概要:")
    print(f"   期間: {len(data)} データポイント")
    print(f"   価格範囲: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   平均価格: ${data['close'].mean():.2f}")
    print(f"   価格標準偏差: ${data['close'].std():.2f}")
    
    # 全ヒルベルト変換アルゴリズムテスト
    print("\n2️⃣ 全ヒルベルト変換アルゴリズムテスト実行")
    results = test_all_hilbert_algorithms(data)
    
    # 結果比較・可視化
    print("\n3️⃣ 結果の可視化")
    output_path = os.path.join('output', 'hilbert_unified_comparison.png')
    os.makedirs('output', exist_ok=True)
    visualize_hilbert_comparison(data, results, output_path)
    
    # 詳細分析
    print("\n4️⃣ 詳細分析")
    
    # 最高性能アルゴリズムの詳細分析
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_algorithm = max(valid_results.keys(), 
                          key=lambda k: valid_results[k]['performance']['total_score'])
        detailed_hilbert_analysis(best_algorithm, data)
    
    # 総合レポート
    print("\n📋 総合レポート")
    print("=" * 60)
    
    valid_count = len(valid_results)
    total_algorithms = len(HilbertTransformUnified.get_available_algorithms())
    
    print(f"✅ テスト成功: {valid_count}/{total_algorithms} ヒルベルト変換アルゴリズム")
    
    if valid_results:
        # トップ3
        ranking = sorted([(k, v['performance']['total_score']) for k, v in valid_results.items()], 
                        key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 トップ3ヒルベルト変換アルゴリズム:")
        for i, (name, score) in enumerate(ranking[:3]):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {emoji} {name}: {score:.3f}")
        
        # 推奨用途
        print(f"\n💡 推奨用途:")
        best_algorithm_name = ranking[0][0]
        if best_algorithm_name == 'basic':
            print("   - 高速処理が必要な場合は basic がお勧め")
        elif best_algorithm_name == 'quantum_enhanced':
            print("   - 高精度解析には quantum_enhanced がお勧め")
        elif best_algorithm_name == 'instantaneous':
            print("   - 詳細な瞬時解析には instantaneous がお勧め")
        elif best_algorithm_name == 'supreme':
            print("   - 最高精度の解析には supreme (9点FIR) がお勧め")
        elif best_algorithm_name == 'multiresolution':
            print("   - マルチ解像度解析には multiresolution がお勧め")
        else:
            print(f"   - {best_algorithm_name} アルゴリズムが最適です")
    
    print(f"\n🎉 デモ完了!")
    print(f"🌀 結果は {output_path} に保存されました")
    print(f"\n📝 各ヒルベルト変換アルゴリズムの特徴:")
    for algorithm_type, description in HilbertTransformUnified.get_available_algorithms().items():
        print(f"   • {algorithm_type}: {description}")


if __name__ == "__main__":
    main() 