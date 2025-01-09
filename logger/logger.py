#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from logging import Logger, Formatter, StreamHandler, FileHandler
from pathlib import Path
from typing import Optional, Dict, Any


class TTFLogger:
    """
    トレーディングシステムのロガークラス
    シングルトンパターンを使用して、システム全体で一貫したロギングを提供
    """
    
    _instance: Optional['TTFLogger'] = None
    _logger: Optional[Logger] = None
    
    def __new__(cls) -> 'TTFLogger':
        """シングルトンインスタンスを返す"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """コンストラクタ"""
        # 既に初期化済みの場合は何もしない
        if self._logger is not None:
            return
            
        self._logger = logging.getLogger('ttf')
        self._logger.setLevel(logging.INFO)
        self._handlers: Dict[str, logging.Handler] = {}
    
    def setup(self, config: Dict[str, Any]) -> None:
        """
        ロガーを設定する
        
        Args:
            config: ログ設定を含む辞書
                {
                    'level': ログレベル ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                    'file': ログファイル名,
                    'format': {
                        'file': ファイルログのフォーマット (オプション),
                        'console': コンソールログのフォーマット (オプション)
                    }
                }
        """
        # 既存のハンドラをクリア
        if self._logger.handlers:
            self._logger.handlers.clear()
        self._handlers.clear()
        
        # ログレベルの設定
        log_level = getattr(logging, config.get('level', 'INFO'))
        self._logger.setLevel(log_level)
        
        # フォーマットの設定
        formats = config.get('format', {})
        file_format = formats.get(
            'file',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_format = formats.get(
            'console',
            '%(levelname)s - %(message)s'
        )
        
        # ファイルハンドラーの設定
        log_file = config.get('file')
        if log_file:
            log_dir = Path(log_file).parent
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            
            file_handler = FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(Formatter(file_format))
            self._logger.addHandler(file_handler)
            self._handlers['file'] = file_handler
        
        # コンソールハンドラーの設定
        console_handler = StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(Formatter(console_format))
        self._logger.addHandler(console_handler)
        self._handlers['console'] = console_handler
    
    def get_logger(self) -> Logger:
        """設定されたロガーを取得する"""
        if self._logger is None:
            raise RuntimeError("ロガーが初期化されていません。setup()を呼び出してください。")
        return self._logger
    
    def set_level(self, level: str) -> None:
        """
        ログレベルを設定する
        
        Args:
            level: ログレベル ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        log_level = getattr(logging, level)
        self._logger.setLevel(log_level)
        for handler in self._handlers.values():
            handler.setLevel(log_level)
    
    def add_file_handler(
        self,
        file_path: str,
        level: str = 'INFO',
        format_str: Optional[str] = None
    ) -> None:
        """
        新しいファイルハンドラーを追加する
        
        Args:
            file_path: ログファイルのパス
            level: ログレベル
            format_str: ログフォーマット
        """
        log_dir = Path(file_path).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        
        handler = FileHandler(file_path, encoding='utf-8')
        handler.setLevel(getattr(logging, level))
        
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        handler.setFormatter(Formatter(format_str))
        
        self._logger.addHandler(handler)
        self._handlers[file_path] = handler
    
    def remove_handler(self, handler_key: str) -> None:
        """
        指定されたハンドラーを削除する
        
        Args:
            handler_key: ハンドラーのキー ('file', 'console', またはファイルパス)
        """
        if handler_key in self._handlers:
            handler = self._handlers[handler_key]
            self._logger.removeHandler(handler)
            handler.close()
            del self._handlers[handler_key]


# シングルトンインスタンスを作成
logger = TTFLogger()

# 便利なメソッドをモジュールレベルで提供
def setup_logger(config: Dict[str, Any]) -> None:
    """ロガーを設定する"""
    logger.setup(config)

def get_logger() -> Logger:
    """設定されたロガーを取得する"""
    return logger.get_logger()

def set_level(level: str) -> None:
    """ログレベルを設定する"""
    logger.set_level(level)

def add_file_handler(
    file_path: str,
    level: str = 'INFO',
    format_str: Optional[str] = None
) -> None:
    """新しいファイルハンドラーを追加する"""
    logger.add_file_handler(file_path, level, format_str)

def remove_handler(handler_key: str) -> None:
    """指定されたハンドラーを削除する"""
    logger.remove_handler(handler_key)
