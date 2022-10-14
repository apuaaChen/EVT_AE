import sqlite3 as db

conn = db.connect('my.db')
c = conn.cursor()
ids = c.execute('SELECT id FROM users').fetchall()

sql = 'DELETE FROM tasks WHERE id=?'
c.execute(sql, (ids[-1],))
conn.commit()

    