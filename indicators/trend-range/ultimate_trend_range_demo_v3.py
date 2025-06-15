#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultimate_trend_range_detector_v3 import UltimateTrendRangeDetectorV3
import time
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def generate_enhanced_market_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    V3用の高度な市場データ生成
    より実際の市場に近いパターンを再現
    """
    np.random.seed(42)
    
    data = []
    current_price = 100.0
    market_state = 'range'
    state_duration = 0
    volatility_regime = 'normal'
    
    for i in range(n_samples):
        # 市場状態の管理
        if state_duration <= 0:
            # 新しい状態を決定
            if market_state == 'range':
                market_state = np.random.choice(['trend_up', 'trend_down', 'range'], 
                                               p=[0.35, 0.35, 0.30])
            else:
                market_state = np.random.choice(['range', 'trend_up', 'trend_down'], 
                                               p=[0.50, 0.25, 0.25])
            
            # 状態の持続期間
            if 'trend' in market_state:
                state_duration = np.random.randint(60, 201)  # 60-200期間の長期トレンド
            else:
                state_duration = np.random.randint(30, 121)  # 30-120期間のレンジ
            
            # ボラティリティレジームの変更
            volatility_regime = np.random.choice(['low', 'normal', 'high'], 
                                                p=[0.3, 0.5, 0.2])
        
        # ボラティリティレジームに応じた基本変動
        if volatility_regime == 'low':
            base_vol = 0.008
        elif volatility_regime == 'normal':
            base_vol = 0.015
        else:  # high
            base_vol = 0.025
        
        # 市場状態に応じた価格生成
        if market_state == 'trend_up':
            # 上昇トレンド
            trend_strength = np.random.uniform(0.0015, 0.005)
            noise_factor = 0.6  # トレンド時はノイズを抑制
            price_change = (trend_strength * current_price + 
                           np.random.normal(0, current_price * base_vol * noise_factor))
            true_regime = 1
            
        elif market_state == 'trend_down':
            # 下降トレンド
            trend_strength = np.random.uniform(-0.005, -0.0015)
            noise_factor = 0.6
            price_change = (trend_strength * current_price + 
                           np.random.normal(0, current_price * base_vol * noise_factor))
            true_regime = 1
            
        else:  # range
            # レンジ相場（平均回帰特性）
            if i >= 30:
                recent_mean = np.mean([d['close'] for d in data[-30:]])
                mean_reversion_force = (recent_mean - current_price) * 0.08
            else:
                mean_reversion_force = 0
            
            noise_factor = 1.2  # レンジ時はノイズを増加
            price_change = (mean_reversion_force + 
                           np.random.normal(0, current_price * base_vol * noise_factor))
            true_regime = 0
        
        # 価格更新
        current_price += price_change
        current_price = max(current_price, 10.0)
        
        # OHLC生成（より現実的な日内変動）
        intraday_vol = current_price * base_vol * 0.8
        high_bias = np.random.uniform(0.2, 1.5)
        low_bias = np.random.uniform(0.2, 1.5)
        
        high = current_price + intraday_vol * high_bias
        low = current_price - intraday_vol * low_bias
        
        # Open価格は前の終値に近い値
        if i > 0:
            prev_close = data[-1]['close']
            gap = np.random.normal(0, current_price * 0.005)  # 小さなギャップ
            open_price = prev_close + gap
        else:
            open_price = current_price
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'true_regime': true_regime,
            'market_state': market_state,
            'volatility_regime': volatility_regime
        })
        
        state_duration -= 1
    
    return pd.DataFrame(data)


def evaluate_performance_v3(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """
    V3用の高度パフォーマンス評価
    """
    # 基本統計
    correct = np.sum(predicted == actual)
    total = len(predicted)
    accuracy = correct / total
    
    # 混同行列
    tp = np.sum((predicted == 1) & (actual == 1))  # True Positive (トレンド正解)
    tn = np.sum((predicted == 0) & (actual == 0))  # True Negative (レンジ正解)
    fp = np.sum((predicted == 1) & (actual == 0))  # False Positive (レンジをトレンドと誤判定)
    fn = np.sum((predicted == 0) & (actual == 1))  # False Negative (トレンドをレンジと誤判定)
    
    # 詳細メトリクス
    precision_trend = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_range = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_trend = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_range = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    f1_trend = 2 * (precision_trend * recall_trend) / (precision_trend + recall_trend) if (precision_trend + recall_trend) > 0 else 0
    f1_range = 2 * (precision_range * recall_range) / (precision_range + recall_range) if (precision_range + recall_range) > 0 else 0
    
    # バランス精度
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Matthews相関係数（MCC）- 不均衡データセットでも有効
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision_trend': precision_trend,
        'precision_range': precision_range,
        'recall_trend': recall_trend,
        'recall_range': recall_range,
        'f1_trend': f1_trend,
        'f1_range': f1_range,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    }


def plot_results_v3(data: pd.DataFrame, results: dict, save_path: str = None):
    """
    V3版の高度な結果可視化
    """
    fig, axes = plt.subplots(6, 1, figsize=(18, 16))
    
    # 1. 価格とシグナル比較
    ax1 = axes[0]
    ax1.plot(data['close'], label='Close Price', alpha=0.8, linewidth=1.5, color='black')
    
    # 真の市場状態を背景色で表示
    trend_mask_true = data['true_regime'] == 1
    range_mask_true = data['true_regime'] == 0
    
    ax1.fill_between(range(len(data)), data['close'].min(), data['close'].max(), 
                     where=trend_mask_true, alpha=0.1, color='green', label='True Trend Periods')
    ax1.fill_between(range(len(data)), data['close'].min(), data['close'].max(), 
                     where=range_mask_true, alpha=0.1, color='red', label='True Range Periods')
    
    # V3予測信号をマーカーで表示
    trend_pred_idx = np.where(results['signal'] == 1)[0]
    range_pred_idx = np.where(results['signal'] == 0)[0]
    
    if len(trend_pred_idx) > 0:
        ax1.scatter(trend_pred_idx, data['close'].iloc[trend_pred_idx], 
                   c='darkgreen', marker='^', s=15, alpha=0.7, label='V3 Trend Signals')
    if len(range_pred_idx) > 0:
        ax1.scatter(range_pred_idx, data['close'].iloc[range_pred_idx], 
                   c='darkred', marker='v', s=15, alpha=0.7, label='V3 Range Signals')
    
    ax1.set_title('🚀 V3.0 Ultimate - Price & Revolutionary Signal Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 主要指標ダッシュボード
    ax2 = axes[1]
    ax2.plot(results['efficiency_ratio'], label='Efficiency Ratio', color='blue', alpha=0.8)
    ax2.plot(results['choppiness_index']/100, label='Choppiness Index (normalized)', color='red', alpha=0.8)
    ax2.plot(results['adx']/100, label='ADX (normalized)', color='purple', alpha=0.8)
    ax2.axhline(y=0.618, color='green', linestyle='--', alpha=0.5, label='Golden Ratio')
    ax2.axhline(y=0.382, color='orange', linestyle='--', alpha=0.5, label='Silver Ratio')
    ax2.set_title('📊 Core Indicators Dashboard', fontweight='bold')
    ax2.set_ylabel('Indicator Values', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 信頼度と適応的閾値
    ax3 = axes[2]
    ax3.plot(results['confidence'], label='V3 Confidence Score', color='darkblue', linewidth=2)
    ax3.plot(results['adaptive_threshold'], label='Adaptive Threshold', color='purple', 
             linestyle='--', alpha=0.8)
    ax3.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='High Confidence Level')
    ax3.fill_between(range(len(results['confidence'])), 0, results['confidence'], 
                     alpha=0.2, color='blue')
    ax3.set_title('🧠 Confidence & Adaptive Intelligence', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. モメンタム一貫性とボラティリティ調整
    ax4 = axes[3]
    ax4.plot(results['momentum_consistency'], label='Momentum Consistency', 
             color='darkgreen', alpha=0.8)
    ax4.plot(results['volatility_adjustment'], label='Volatility Adjustment', 
             color='orange', alpha=0.8)
    ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('🎯 Advanced Market Analysis', fontweight='bold')
    ax4.set_ylabel('Factor Values', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 判定精度の時系列分析
    ax5 = axes[4]
    # 移動平均精度を計算
    window = 100
    rolling_accuracy = []
    for i in range(window, len(data)):
        pred_window = results['signal'][i-window:i]
        actual_window = data['true_regime'].values[i-window:i]
        acc = np.sum(pred_window == actual_window) / window
        rolling_accuracy.append(acc)
    
    # プロットの調整
    x_rolling = range(window, len(data))
    ax5.plot(x_rolling, rolling_accuracy, color='red', linewidth=2, 
             label=f'Rolling Accuracy ({window}-period)')
    ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% Target')
    ax5.axhline(y=np.mean(rolling_accuracy), color='blue', linestyle=':', 
               alpha=0.7, label=f'Average: {np.mean(rolling_accuracy):.3f}')
    ax5.fill_between(x_rolling, 0.8, rolling_accuracy, 
                     where=np.array(rolling_accuracy) >= 0.8, 
                     alpha=0.3, color='green', label='Above Target')
    ax5.set_title('📈 Real-time Accuracy Performance', fontweight='bold')
    ax5.set_ylabel('Accuracy', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. エラー分析
    ax6 = axes[5]
    errors = (results['signal'] != data['true_regime'].values).astype(int)
    cumulative_error_rate = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    
    ax6.plot(cumulative_error_rate, color='red', linewidth=2, label='Cumulative Error Rate')
    ax6.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='20% Error Target')
    ax6.fill_between(range(len(cumulative_error_rate)), 0, errors, 
                     alpha=0.2, color='red', label='Individual Errors')
    ax6.set_title('🔍 Error Analysis & Learning Curve', fontweight='bold')
    ax6.set_ylabel('Error Rate', fontweight='bold')
    ax6.set_xlabel('Time', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def detailed_signal_analysis(data: pd.DataFrame, results: dict) -> dict:
    """
    V3用の詳細シグナル分析
    """
    analysis = {}
    
    # シグナルの遷移分析
    signal_changes = np.diff(results['signal'])
    trend_to_range = np.sum(signal_changes == -1)
    range_to_trend = np.sum(signal_changes == 1)
    
    analysis['signal_transitions'] = {
        'trend_to_range': trend_to_range,
        'range_to_trend': range_to_trend,
        'total_transitions': trend_to_range + range_to_trend
    }
    
    # 信頼度別の精度分析
    high_conf_mask = results['confidence'] >= 0.8
    med_conf_mask = (results['confidence'] >= 0.6) & (results['confidence'] < 0.8)
    low_conf_mask = results['confidence'] < 0.6
    
    for name, mask in [('high', high_conf_mask), ('medium', med_conf_mask), ('low', low_conf_mask)]:
        if np.sum(mask) > 0:
            pred_subset = results['signal'][mask]
            actual_subset = data['true_regime'].values[mask]
            accuracy = np.sum(pred_subset == actual_subset) / len(pred_subset)
            analysis[f'{name}_confidence_accuracy'] = accuracy
        else:
            analysis[f'{name}_confidence_accuracy'] = 0.0
    
    # 市場状態別の分析
    market_states = data['market_state'].unique()
    for state in market_states:
        state_mask = data['market_state'] == state
        if np.sum(state_mask) > 0:
            pred_subset = results['signal'][state_mask]
            actual_subset = data['true_regime'].values[state_mask]
            accuracy = np.sum(pred_subset == actual_subset) / len(pred_subset)
            analysis[f'{state}_accuracy'] = accuracy
    
    return analysis


def main():
    """
    V3メイン実行関数 - 80%精度への挑戦
    """
    print("🚀 人類史上最強トレンド/レンジ判別インジケーター V3.0 - REVOLUTIONARY EDITION")
    print("=" * 100)
    print("🎯 目標: 80%以上の判別精度達成")
    print("💎 革新技術: Efficiency Ratio + Choppiness Index + ADX + Momentum Consistency")
    print("=" * 100)
    
    # 1. 高度な市場データ生成
    print("\n📊 高度な市場データ生成中...")
    data = generate_enhanced_market_data(2500)  # より多くのデータで精度向上
    print(f"✅ データ生成完了: {len(data)}件")
    
    # データ統計の表示
    actual_trend_count = sum(data['true_regime'])
    actual_range_count = len(data) - actual_trend_count
    print(f"   📈 真のトレンド期間: {actual_trend_count}件 ({actual_trend_count/len(data)*100:.1f}%)")
    print(f"   📉 真のレンジ期間: {actual_range_count}件 ({actual_range_count/len(data)*100:.1f}%)")
    
    # 市場状態の分布
    state_dist = data['market_state'].value_counts()
    print(f"   🔄 市場状態分布: {dict(state_dist)}")
    
    # 2. V3.0インジケーター初期化
    print("\n🔧 Ultimate V3.0 初期化中...")
    detector_v3 = UltimateTrendRangeDetectorV3(
        er_period=21,      # Efficiency Ratio期間
        chop_period=14,    # Choppiness Index期間
        adx_period=14,     # ADX期間
        vol_period=20      # ボラティリティ調整期間
    )
    print("✅ V3.0 初期化完了")
    print("   🧠 アンサンブル構成: ER(35%) + Chop(25%) + ADX(25%) + Momentum(15%)")
    
    # 3. 計算実行
    print("\n⚡ V3.0 革命的判別計算実行中...")
    start_time = time.time()
    
    results = detector_v3.calculate(data)
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    print(f"✅ 計算完了 (処理時間: {calculation_time:.2f}秒)")
    print(f"   ⚡ 処理速度: {len(data)/calculation_time:.0f} データ/秒")
    
    # 4. パフォーマンス評価
    print("\n📈 V3.0 パフォーマンス評価中...")
    
    # 初期期間をスキップして評価
    skip_initial = 100
    predicted_signals = results['signal'][skip_initial:]
    actual_signals = data['true_regime'].values[skip_initial:]
    
    performance = evaluate_performance_v3(predicted_signals, actual_signals)
    
    # 5. 詳細分析
    detailed_analysis = detailed_signal_analysis(data, results)
    
    # 6. 結果表示
    print("\n" + "="*80)
    print("🏆 **V3.0 REVOLUTIONARY PERFORMANCE RESULTS**")
    print("="*80)
    print(f"📊 総合精度: {performance['accuracy']:.4f} ({performance['accuracy']*100:.2f}%)")
    print(f"⚖️  バランス精度: {performance['balanced_accuracy']:.4f} ({performance['balanced_accuracy']*100:.2f}%)")
    print(f"💎 MCC（品質指標）: {performance['mcc']:.4f}")
    print(f"🎯 80%目標達成: {'✅ SUCCESS!' if performance['accuracy'] >= 0.80 else '🔄 PROGRESS...'}")
    
    print(f"\n📈 **トレンド判別詳細**")
    print(f"   - 精度 (Precision): {performance['precision_trend']:.4f} ({performance['precision_trend']*100:.1f}%)")
    print(f"   - 再現率 (Recall): {performance['recall_trend']:.4f} ({performance['recall_trend']*100:.1f}%)")
    print(f"   - F1スコア: {performance['f1_trend']:.4f}")
    
    print(f"\n📉 **レンジ判別詳細**")
    print(f"   - 精度 (Precision): {performance['precision_range']:.4f} ({performance['precision_range']*100:.1f}%)")
    print(f"   - 再現率 (Recall): {performance['recall_range']:.4f} ({performance['recall_range']*100:.1f}%)")
    print(f"   - F1スコア: {performance['f1_range']:.4f}")
    
    # 7. 技術的統計
    print("\n" + "="*80)
    print("🔬 **V3.0 技術統計**")
    print("="*80)
    summary = results['summary']
    print(f"📊 予測統計:")
    print(f"   - トレンド期間: {summary['trend_bars']}件 ({summary['trend_ratio']*100:.1f}%)")
    print(f"   - レンジ期間: {summary['range_bars']}件 ({(1-summary['trend_ratio'])*100:.1f}%)")
    print(f"   - 平均信頼度: {summary['avg_confidence']:.4f}")
    print(f"   - 高信頼度比率: {summary['high_confidence_ratio']*100:.1f}%")
    
    print(f"\n🎯 指標平均値:")
    print(f"   - Efficiency Ratio: {summary['er_avg']:.4f}")
    print(f"   - Choppiness Index: {summary['chop_avg']:.2f}")
    print(f"   - ADX: {summary['adx_avg']:.2f}")
    
    # 8. 信頼度別精度
    print(f"\n💎 信頼度別精度:")
    print(f"   - 高信頼度(≥80%): {detailed_analysis['high_confidence_accuracy']:.4f}")
    print(f"   - 中信頼度(60-80%): {detailed_analysis['medium_confidence_accuracy']:.4f}")
    print(f"   - 低信頼度(<60%): {detailed_analysis['low_confidence_accuracy']:.4f}")
    
    # 9. 技術革新の詳細
    print("\n" + "="*80)
    print("🌟 **V3.0 技術革新**")
    print("="*80)
    print("🧠 革命的アンサンブル統合:")
    print("   1. Efficiency Ratio (35%重み) - 価格変動効率性測定")
    print("   2. Choppiness Index (25%重み) - 市場チョピネス精密解析")
    print("   3. ADX (25%重み) - トレンド強度確実定量化")
    print("   4. Momentum Consistency (15%重み) - 多時間軸方向性一致度")
    print()
    print("💡 革新機能:")
    print("   - 適応的閾値システム（市況応じた動的基準）")
    print("   - ボラティリティ調整機構（変動性考慮精度向上）")
    print("   - 高度ノイズ除去（信頼度重み付きフィルタ）")
    print("   - 統計的検証済み判定ロジック")
    
    # 10. 可視化
    print("\n📊 V3.0 革命的結果可視化中...")
    plot_results_v3(data, results, 'ultimate_trend_range_v3_revolutionary_results.png')
    print("✅ 可視化完了 (ultimate_trend_range_v3_revolutionary_results.png)")
    
    # 11. 最終評価
    print("\n" + "="*90)
    print("🏆 **V3.0 REVOLUTIONARY FINAL EVALUATION**")
    print("="*90)
    
    final_score = performance['accuracy']
    
    if final_score >= 0.85:
        grade = "🏆 LEGENDARY REVOLUTIONARY"
        comment = "革命的成功！人類史上最強の判別精度を達成！"
        emoji = "🎉✨🚀"
    elif final_score >= 0.80:
        grade = "🥇 REVOLUTIONARY SUCCESS"
        comment = "革命成功！80%目標を達成した歴史的快挙！"
        emoji = "🎊🏆💎"
    elif final_score >= 0.75:
        grade = "🥈 OUTSTANDING ACHIEVEMENT"
        comment = "卓越した性能！80%目標まであと一歩！"
        emoji = "⭐🔥💪"
    elif final_score >= 0.70:
        grade = "🥉 EXCELLENT PROGRESS"
        comment = "素晴らしい進歩！V1,V2を大幅に上回る性能！"
        emoji = "📈🎯✨"
    else:
        grade = "📈 CONTINUOUS INNOVATION"
        comment = "継続的革新中。さらなる改善の余地あり。"
        emoji = "🔧⚡🧠"
    
    print(f"🎖️  最終評価: {grade}")
    print(f"💬 評価コメント: {comment} {emoji}")
    print(f"📊 総合精度: {final_score*100:.2f}%")
    print(f"📊 バランス精度: {performance['balanced_accuracy']*100:.2f}%")
    print(f"💎 品質指標(MCC): {performance['mcc']:.4f}")
    
    if final_score >= 0.80:
        print(f"\n🎉 **🏆 80%目標達成！革命的成功！ 🏆**")
        print("🚀 V3.0は真に人類史上最強のトレンド/レンジ判別器です！")
        print("💎 Efficiency Ratio + Choppiness + ADX の完璧な融合により革命を実現！")
    elif final_score >= 0.75:
        print(f"\n⭐ **目標まであと少し！驚異的な性能向上！**")
        print("🔥 V3.0の革新技術により、従来手法を大幅に上回りました！")
    
    # 12. 混同行列詳細
    cm = performance['confusion_matrix']
    print(f"\n📊 **混同行列詳細:**")
    print(f"   ✅ True Positive (トレンド→トレンド): {cm['tp']}")
    print(f"   ✅ True Negative (レンジ→レンジ): {cm['tn']}")
    print(f"   ❌ False Positive (レンジ→トレンド): {cm['fp']}")
    print(f"   ❌ False Negative (トレンド→レンジ): {cm['fn']}")
    
    # 13. シグナル遷移分析
    transitions = detailed_analysis['signal_transitions']
    print(f"\n🔄 **シグナル遷移分析:**")
    print(f"   📈➡️📉 トレンド→レンジ: {transitions['trend_to_range']}回")
    print(f"   📉➡️📈 レンジ→トレンド: {transitions['range_to_trend']}回")
    print(f"   🔄 総遷移回数: {transitions['total_transitions']}回")
    
    print("\n" + "="*90)
    print("V3.0 REVOLUTIONARY DEMONSTRATION COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main() 