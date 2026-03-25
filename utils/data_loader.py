from typing import IO

import pandas as pd
import sqlalchemy


def load_csv_data(uploaded_file: IO) -> pd.DataFrame | None:
    """
    Загружает CSV-файл в DataFrame.

    Args:
        uploaded_file (IO): Загруженный CSV-файл.

    Returns:
        pd.DataFrame | None: DataFrame с данными или None при ошибке.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        return None


def load_db_data(
    host: str,
    port: str | int,
    user: str,
    password: str,
    database: str,
    table: str,
) -> pd.DataFrame | None:
    """
    Загружает данные из SQL БД (по имени таблицы или SQL-запросу).

    Args:
        host (str): Адрес сервера.
        port (str | int): Порт подключения.
        user (str): Имя пользователя.
        password (str): Пароль.
        database (str): Имя базы данных.
        table (str): Имя таблицы или SQL-запрос.

    Returns:
        pd.DataFrame | None: DataFrame с данными или None при ошибке.
    """
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    try:
        engine = sqlalchemy.create_engine(conn_str)
        if table.strip().lower().startswith("select"):
            query = table
        else:
            query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, engine)
        return df
    except Exception:
        return None
