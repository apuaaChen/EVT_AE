################################################################################
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
# Helper functions to remove specific compiled kernels from DB
import sqlite3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--keyword", type=str, help="the keys containing the keyward will be deleted.")
args = parser.parse_args()

connection = sqlite3.connect("./compiled_cache.db")
cursor = connection.cursor()
cursor.execute("SELECT op_key FROM compiled_operations")
op_keys = cursor.fetchall()

op_keys_target = [op_key[0] for op_key in op_keys if args.keyword in op_key[0]]

for key in op_keys_target:
    cursor.execute(f"DELETE FROM compiled_operations WHERE op_key = ?", (key,))

print(f"Deleted {len(op_keys_target)} rows from db")
connection.commit()
connection.close()
