import sqlite3

import pandas as pd

from gesture_classification.path_handling import get_sqlite_db_path


class SqliteDatabase:
    def __init__(self, db_uri: str = get_sqlite_db_path()):
        self._db_uri = f"{db_uri}"
        self._db_connection = None
        self.create_db()

    def create_db(self, location: str = None) -> None:
        if not location:
            location = self._db_uri

        self._db_connection = sqlite3.connect(location)
        self._db_uri = location

    def insert_data_frame(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "append"
    ) -> None:
        df.to_sql(table_name, self.db_connection, if_exists=if_exists)

    def read_table(self, table_name: str = "reference_data") -> pd.DataFrame:
        sql_query: str = f"SELECT * FROM {table_name}"

        return pd.read_sql(sql_query, self.db_connection)

    @property
    def db_uri(self) -> str:
        return self._db_uri

    @property
    def db_connection(self):
        return self._db_connection
