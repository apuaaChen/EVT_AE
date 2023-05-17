import sqlite3
connection = sqlite3.connect('./compiled_cache.db')
try:
    connection.execute("DROP TABLE compiled_operations")
except:
    pass
connection.execute("vacuum")
connection.close()