"""
Database Utilities

This module provides database connection utilities.
"""

import os
import oracledb
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()

@contextmanager
def get_connection():
    """
    Get a database connection using environment variables.
    
    Returns:
        oracledb.Connection: Database connection
    """
    # 获取环境变量
    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")
    
    # 创建连接
    conn = oracledb.connect(
        user=user,
        password=password,
        dsn=dsn
    )
    
    try:
        yield conn
    finally:
        conn.close() 