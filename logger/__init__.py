#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from logging import Logger
from pathlib import Path
from typing import Optional


class TTFLogger:
    """
    トレーディングシステムのロガー
    シングルトンパターンでロガーインスタンスを提供
    """
    
    _instance: Optional[Logger] = None
    
    @classmethod
    def get_logger(cls) -> Logger:
        """
        ロガーインスタンスを取得する
        
        Returns:
            設定済みのロガーインスタンス
        """
        if cls._instance is None:
            # ロガーを作成
            logger = logging.getLogger('ttf')
            logger.setLevel(logging.INFO)
            
            # フォーマッターを作成
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # コンソールハンドラーを追加
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # ログディレクトリを作成
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # ファイルハンドラーを追加
            file_handler = logging.FileHandler(
                log_dir / 'ttf.log',
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            cls._instance = logger
        
        return cls._instance


# 便利な関数としてエクスポート
def get_logger() -> Logger:
    """
    ロガーを取得する
    
    Returns:
        設定済みのロガーインスタンス
    """
    return TTFLogger.get_logger()
