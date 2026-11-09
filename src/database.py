
import os

import pandas as pd
import psycopg2


def load_prices():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="postgres",
        user="postgres",
        password=os.environ["PGPASSWORD"]
    )

    # Retrieve historical prices
    query = """
        SELECT ticker, date, open, high, low,
               close, adj_close, volume
        FROM prices
        ORDER BY ticker, date;
    """

    try:
        # Load the results into a pandas DataFrame
        with conn.cursor() as cursor:
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            sortedData = cursor.fetchall()
    finally:
        # Always close the database connection
        conn.close()

    return pd.DataFrame(sortedData, columns=columns)
