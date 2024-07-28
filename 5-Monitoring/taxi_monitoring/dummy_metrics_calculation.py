import datetime
import time
import random
import logging
import uuid
import pytz
import pandas as pd
import io
import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: [%(message)s]")

SEND_TIMEOUT = 10
rand = random.Random()
create_table_query = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics(
    timestamp TIMESTAMP,
    value1 INTEGER,
    value2 VARCHAR,
    value3 FLOAT
)
"""


def prep_db():
    # Connect to the postgres database that is defined in the prev. used docker-compose file
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:  # Create database if no database is available
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
            conn.execute(create_table_query)


def calculate_dummy_metrics_postgresql(curr):
    value1 = random.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = random.random()

    curr.execute(
		"INSERT INTO dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/Berlin')), value1, value2, value3)
	)


def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 100):
            with conn.cursor() as curr:
                calculate_dummy_metrics_postgresql(curr)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info(f"({i}) data sent ")

if __name__ == "__main__":
    main()