#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from logging import getLogger, StreamHandler, Formatter


def get_logger(name: str = __name__) -> logging.Logger:
    """ロガーを取得する
    
    Args:
        name: ロガー名（デフォルトは呼び出し元のモジュール名）
    
    Returns:
        設定済みのロガーインスタンス
    """
    logger = getLogger(name)
    
    if not logger.handlers:
        # ハンドラーが未設定の場合のみ設定を行う
        handler = StreamHandler(sys.stdout)
        
        # フォーマッターを設定
        formatter = Formatter(
            fmt='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # ハンドラーをロガーに追加
        logger.addHandler(handler)
        
        # ログレベルを設定
        logger.setLevel(logging.INFO)
    
    return logger
