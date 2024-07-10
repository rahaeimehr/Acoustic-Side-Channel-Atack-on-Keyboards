import sqlite3


class DbHandler:
    def __init__(self, database_address='./key.db'):
        self.database_address = database_address
        self.db_address = self.build_file_address
        self.conn, self.cur = self.connection()
        self.create_db()

    @property
    def build_file_address(self):
        file_address = f"{self.database_address}"
        # print(f"File Path: {file_address}")
        return file_address

    def connection(self):
        try:
            conn = sqlite3.connect(self.db_address)
            cur = conn.cursor()
            return conn, cur
        except Exception as ex:
            print(ex)

    def create_db(self, ):
        try:
            with self.conn:
                self.cur.execute("""CREATE TABLE IF NOT EXISTS start_end(
                                            id INTEGER PRIMARY KEY,
                                            datetime_start timestamp,
                                            datetime_end timestamp);
                                            """)
                self.conn.commit()
                self.cur.execute("""CREATE TABLE IF NOT EXISTS keystrokes(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            datetime_down timestamp,
                            datetime_up timestamp,
                            key TEXT);
                            """)
                self.conn.commit()
                self.cur.execute("""CREATE TABLE IF NOT EXISTS keystrokes_timing(
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        first_key TEXT,
                                        second_key TEXT,
                                        delta_time TEXT,
                                        is_average TEXT);
                                        """)
                self.conn.commit()
                # print("** DB successfully created **")

        except Exception as ex:
            print(ex)

    def db_insert_start_end(self, data):
        try:
            with self.conn:
                self.cur.executemany(
                    "INSERT INTO start_end(datetime_start, datetime_end) VALUES(?, ?);", data)
                self.conn.commit()
        except Exception as ex:
            print(ex)

    def db_insert(self, data):
        try:
            with self.conn:
                self.cur.executemany(
                    "INSERT INTO keystrokes(datetime_down, datetime_up, key) VALUES(?, ?, ?);", data)
                self.conn.commit()
        except Exception as ex:
            print(ex)

    def db_insert_timing(self, data):
        try:
            with self.conn:
                self.cur.executemany(
                    "INSERT INTO keystrokes_timing(first_key, second_key, delta_time, is_average) VALUES(?,?,?,?);",
                    data)
                self.conn.commit()
        except Exception as ex:
            print(ex)

    def db_fetch_start_end(self,):
        with self.conn:
            self.cur.execute(f"SELECT id, datetime_start, datetime_end FROM start_end WHERE id = '{1}'")
            row = self.cur.fetchone()
        return row

    def db_fetch_all_specific_key(self, key_value='c'):
        with self.conn:
            self.cur.execute(f'''SELECT id, datetime, key FROM keystrokes WHERE key = "{key_value}"''')
            rows = self.cur.fetchall()
        return rows

    def db_fetch_all_specific_id(self, key_id):
        with self.conn:
            self.cur.execute(f"SELECT id, datetime, key FROM keystrokes WHERE id = '{key_id}'")
            rows = self.cur.fetchone()
        return rows

    def db_fetch_all_keys(self, ):
        with self.conn:
            self.cur.execute(f"SELECT id, datetime_down, datetime_up, key FROM keystrokes")
            row = self.cur.fetchall()
        return row

    def db_fetch_equal_delta_time(self, delta_time_min, delta_time_max):
        with self.conn:  # low_expression AND high_expression
            self.cur.execute(
                f"SELECT * FROM keystrokes_timing WHERE delta_time BETWEEN {delta_time_min} AND {delta_time_max}")
            rows = self.cur.fetchall()
        return rows

    def db_fetch_equal_two_keys(self, first_key, second_key):
        with self.conn:
            self.cur.execute(
                f"SELECT * FROM keystrokes_timing WHERE first_key = '{first_key}' AND second_key = '{second_key}'")
            row = self.cur.fetchone()
        return row

    def update_delta_time(self, id, delta_time):
        with self.conn:
            self.cur.execute(f"UPDATE keystrokes_timing SET delta_time='{delta_time}', is_average=1  WHERE id={id};")
            self.conn.commit()
