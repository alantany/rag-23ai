import oracledb
from typing import Generator
import os

def get_oracle_connection() -> oracledb.Connection:
    return oracledb.connect(
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        dsn=os.getenv("ORACLE_DSN")
    ) 