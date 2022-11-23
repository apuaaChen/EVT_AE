import sqlite3 as db

conn = db.connect('compiled_cache.db')
c = conn.cursor()
ids = c.execute('SELECT op_key FROM best_config').fetchall()
for id in ids:
    sql = 'DELETE FROM best_config WHERE op_key=?'
    c.execute(sql, (id[0],))
conn.commit()

    