#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PriceSourceインポートデバッグ
"""

import sys
import os
import warnings

# 警告を表示
warnings.filterwarnings('always')

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("価格ソースモジュールのインポートテスト...")
print(f"Python path: {sys.path[:3]}...")

# 段階的にインポート
try:
    print("\n1. indicatorsモジュールのインポート...")
    import indicators
    print("   成功")
except Exception as e:
    print(f"   失敗: {e}")

try:
    print("\n2. smootherモジュールの確認...")
    import indicators.smoother
    print("   成功")
except Exception as e:
    print(f"   失敗: {e}")

try:
    print("\n3. ultimate_smootherの直接インポート...")
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    print(f"   成功: {UltimateSmoother}")
except Exception as e:
    print(f"   失敗: {e}")

try:
    print("\n4. super_smootherの直接インポート...")
    from indicators.smoother.super_smoother import SuperSmoother
    print(f"   成功: {SuperSmoother}")
except Exception as e:
    print(f"   失敗: {e}")

# price_sourceをインポート
print("\n5. price_sourceのインポート...")
try:
    from indicators.price_source import PriceSource, UltimateSmoother as PS_US, SuperSmoother as PS_SS
    print(f"   PriceSource: 成功")
    print(f"   UltimateSmoother in price_source: {PS_US}")
    print(f"   SuperSmoother in price_source: {PS_SS}")
    
    # 利用可能なソースを確認
    sources = PriceSource.get_available_sources()
    print(f"\n   利用可能なソース数: {len(sources)}")
    
    # スムーザーソースを探す
    smoother_sources = [k for k in sources.keys() if k.startswith('us_') or k.startswith('ss_')]
    print(f"   スムーザーソース: {smoother_sources}")
    
except Exception as e:
    print(f"   失敗: {e}")
    import traceback
    traceback.print_exc()