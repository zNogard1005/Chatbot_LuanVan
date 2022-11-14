import sqlite3
from sqlite3 import Error


def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection("sm_app.sqlite")

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        connection.commit()
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")
create_users_table = """

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question TEXT NOT NULL,
  tag TEXT NOT NULL
);
"""
execute_query(connection, create_users_table)
create_users = """INSERT INTO users (question,tag) VALUES (?,'diemchuan'),"""
def execute_query_insert(connection,question,tag):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute("INSERT INTO users (question,tag) VALUES (?,?)", (question,tag))
        connection.commit()
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")
def execute_query_delete(connection,id):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute("DELETE FROM users where id=?",(id,))
        connection.commit()
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

# question = "diem chuan cntt"
# tag = "diemchuan"
# execute_query_insert(connection, question, tag)
# # execute_query_delete(connection,2)
# select_users = "SELECT * from users"
# users = execute_read_query(connection, select_users)
# for user in users:
#     print(user)
