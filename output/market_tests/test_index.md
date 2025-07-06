# 4つのAdaptive UKF手法 - 相場データテスト結果

## テスト概要
このテストは4つのAdaptive UKF手法を様々な相場データで比較評価したものです。

## 比較手法
1. **標準UKF** - 基準手法
2. **私の実装版AUKF** - 統計的監視・適応制御
3. **論文版AUKF** - Ge et al. (2019) 相互相関理論
4. **Neural版AUKF** - Levy & Klein (2025) CNN ProcessNet

## テスト結果
### サンプルデータ（トレンド）
- チャート: [サンプルデータ（トレンド）_chart.png](サンプルデータ（トレンド）_chart.png)
- 性能データ: [サンプルデータ（トレンド）_performance.csv](サンプルデータ（トレンド）_performance.csv)

### サンプルデータ（ボラティリティ）
- チャート: [サンプルデータ（ボラティリティ）_chart.png](サンプルデータ（ボラティリティ）_chart.png)
- 性能データ: [サンプルデータ（ボラティリティ）_performance.csv](サンプルデータ（ボラティリティ）_performance.csv)

### サンプルデータ（サイクリック）
- チャート: [サンプルデータ（サイクリック）_chart.png](サンプルデータ（サイクリック）_chart.png)
- 性能データ: [サンプルデータ（サイクリック）_performance.csv](サンプルデータ（サイクリック）_performance.csv)

## 統合結果
- [統合レポート](comprehensive_test_summary.csv)
- [統計サマリー](performance_statistics.csv)
