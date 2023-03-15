import sqlite3 as db

conn = db.connect('compiled_cache.db')
c = conn.cursor()
# ids = c.execute('SELECT op_key FROM best_config').fetchall()
ids = c.execute('SELECT op_key FROM compiled_operations').fetchall()

print(ids)

for id in ids:
    if "softmax_kernel" in id[0]:
        print(id)
        print("================================================")
        sql = 'DELETE FROM compiled_operations WHERE op_key=?'
        c.execute(sql, (id[0],))
conn.commit()
# # for id in ids:
#     sql = 'DELETE FROM best_config WHERE op_key=?'
#     c.execute(sql, (id[0],))
# conn.commit()

    