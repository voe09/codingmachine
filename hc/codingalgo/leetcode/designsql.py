# create tables, insert rows, delete rows, and select rows based on simple condition

class Table:

    def __init__(self, columns: list[str]):
        self.cols = columns
        self.vocab = {}
        for i in range(len(self.cols)):
            self.vocab[self.cols[i]] = i
        self.rows = []
    
    def insertRow(self, row: list[str]):
        self.rows.append(row)
    
    def deleteRows(self, whereCol: str, whereVal: str):
        self.rows = [row for row in self.rows if row[self.vocab[whereCol]] != whereVal]
    
    def selectRows(self, cols: list[str], whereCol: str, whereVal: str) -> list[list[str]]:
        rows = [r for r in self.rows if r[self.vocab[whereCol]] == whereVal]
        res = []
        for row in rows:
            col_vals = [row[self.vocab[c]] for c in cols]
            res.append(col_vals)
        return res

class SQL:

    def __init__(self):
        self.tables = {}
    
    def createTable(self, tableName: str, columns: list[str]):
        table = Table(columns)
        self.tables[tableName] = table

    def insertRow(self, tableName: str, row: list[str]):
        if tableName not in self.tables:
            return
        
        self.tables[tableName].insertRow(row)
    
    def deleteRows(self, tableName: str, whereCol: str, whereVal: str):
        if tableName not in self.tables:
            return

        self.tables[tableName].deleteRows(whereCol, whereVal)

    
    def selectRows(self, tableName: str, cols: list[str], whereCol: str, whereVal: str):
        if tableName not in self.tables:
            return

        return self.tables[tableName].selectRows(cols, whereCol, whereVal)

sql = SQL()
sql.createTable("people", ["age", "name", "sex"])
sql.insertRow("people", ["32", "bryan", "M"])
sql.insertRow("people", ["16", "Lucy", "F"])
sql.insertRow("people", ["93", "Chris", "M"])
print(sql.selectRows("people", ["name"], "sex", "F"))
sql.deleteRows("people", "sex", "M")
print(sql.selectRows("people", ["name"], "sex", "F"))



# Optimization
# create tables, insert rows, delete rows, and select rows based on simple condition

class Table:

    def __init__(self, columns: list[str]):
        self.cols = columns
        self.vocab = {col: i for i, col in enumerate(self.cols)}
        self.rows = []
        self.index = {col: {} for col in self.cols}
    
    def insertRow(self, row: list[str]):
        row_index = len(self.rows)
        self.rows.append(row)

        for col, val in zip(self.cols, row):
            if val not in self.index[col]:
                self.index[col][val] = set()
            self.index[col][val].add(row_index)

    
    def deleteRows(self, whereCol: str, whereVal: str):
        rows_to_delete = list(self.index.get(whereCol, {}).get(whereVal, set()))

        for row_idx in rows_to_delete:
            row = self.rows[row_idx]
            for col, val in zip(self.cols, row):
                self.index[col][val].discard(row_idx)
            self.rows[row_idx] = None
    
    def selectRows(self, cols: list[str], whereCol: str, whereVal: str) -> list[list[str]]:
        selected_row_index = self.index.get(whereCol, {}).get(whereVal, set())

        res = []
        for idx in selected_row_index:
            row = self.rows[idx]
            res.append([row[self.vocab[col]] for col in cols])
        return res

class SQL:

    def __init__(self):
        self.tables = {}
    
    def createTable(self, tableName: str, columns: list[str]):
        table = Table(columns)
        self.tables[tableName] = table

    def insertRow(self, tableName: str, row: list[str]):
        if tableName not in self.tables:
            return
        
        self.tables[tableName].insertRow(row)
    
    def deleteRows(self, tableName: str, whereCol: str, whereVal: str):
        if tableName not in self.tables:
            return

        self.tables[tableName].deleteRows(whereCol, whereVal)

    
    def selectRows(self, tableName: str, cols: list[str], whereCol: str, whereVal: str):
        if tableName not in self.tables:
            return

        return self.tables[tableName].selectRows(cols, whereCol, whereVal)

sql = SQL()
sql.createTable("people", ["age", "name", "sex"])
sql.insertRow("people", ["32", "bryan", "M"])
sql.insertRow("people", ["16", "Lucy", "F"])
sql.insertRow("people", ["93", "Chris", "M"])
print(sql.selectRows("people", ["name"], "sex", "F"))
sql.deleteRows("people", "sex", "M")
print(sql.selectRows("people", ["name"], "sex", "F"))



from collections import defaultdict
from typing import Optional


class Table:

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.col_index = {}
        for i, col in enumerate(columns):
            self.col_index[col] = i
        
        self.rows = []
        self.index = {} # dict[str, dict[str, list[int]]]
        for col in self.columns:
            self.index[col] = defaultdict(set)


    def insert(self, row: list[str]):
        row_index = len(self.rows)
        self.rows.append(row)

        for col, val in zip(self.columns, row):
            self.index[col][val].add(row_index)
        
    def delete(self, whereCols: list[str], whereVals: list[str]):

        row_indexs = self._select(whereCols, whereVals)
        
        for idx in row_indexs:
            row = self.rows[idx]
            self.rows[idx] = None
            for col, val in zip(self.columns, row):
                self.index[col][val].remove(idx)

    def _select(self, whereCols: list[str], whereVals: list[str]) -> set[int]:
        row_indexs = self.index[whereCols[0]][whereVals[0]]
        for col, val in zip(whereCols[1:], whereVals[1:]):
            row_indexs &= self.index[col][val]
        return row_indexs

    def select(self, whereCols: list[str], whereVals: list[str], orderBy: Optional[str] = None):

        row_indexes = self._select(whereCols, whereVals)

        rows = []
        for idx in row_indexes:
            rows.append(self.rows[idx])

        if orderBy:
            rows = sorted(rows, key = lambda x: x[self.col_index[orderBy]])

        return rows    


class SQL:

    def __init__(self):
        self.tables = {}

    def create(self, tableName: str, columns: list[str]):
        self.tables[tableName] = Table(columns)

    def insert(self, tableName: str, row: list[str]):
        self.tables[tableName].insert(row)

    def delete(self, tableName: str, whereCols: list[str], whereVals: list[str]):
        self.tables[tableName].delete(whereCols, whereVals)
    
    def select(self, tableName: str, whereCols: list[str], whereVals: list[str], orderBy: Optional[str] = None):
        return self.tables[tableName].select(whereCols, whereVals, orderBy)


sql = SQL()

sql.create("people", ["name", "age", "sex", "score"])
sql.insert("people", ["Allan", "30", "M", "95"])
sql.insert("people", ["Chris", "35", "M", "90"])
sql.insert("people", ["Hannah", "28", "F", "99"])
sql.insert("people", ["Allan", "30", "M", "70"])
print(sql.select("people", ["sex", "name"], ["F", "Chris"], "score"))



import unittest

from typing import Optional


class Table:

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.col_index = {}
        for i, col in enumerate(cols):
            self.col_index[col] = i
        self.rows = []

    def insertRow(self, row: list[str]):
        self.rows.append(row)
    
    def delete(
        self, 
        whereCols: list[str], 
        whereVals: list[str], 
        joiningOp: str,
    ):
        rows_to_delete = []
        for row in self.rows:
            if joiningOp == "AND":
                to_delete = all(row[self.col_index[whereCol]] == whereVal for whereCol, whereVal in zip(whereCols, whereVals))
            else:
                to_delete = any(row[self.col_index[whereCol]] == whereVal for whereCol, whereVal in zip(whereCols, whereVals))
            if to_delete:
                rows_to_delete.append(row)
        
        for row in rows_to_delete:
            self.rows.remove(row)
    
    def query(self, cols: list[str], whereCols: list[str], whereVals: list[str], joiningOp: str, orderBy: Optional[str] = None, asc: bool = True):
        rows_selected = []
        for row in self.rows:
            if joiningOp == "AND":
                selected = all(row[self.col_index[whereCol]] == whereVal for whereCol, whereVal in zip(whereCols, whereVals))
            else:
                selected = any(row[self.col_index[whereCol]] == whereVal for whereCol, whereVal in zip(whereCols, whereVals))
            if selected:
                rows_selected.append(row)
        
        if orderBy:
            rows_selected = sorted(rows_selected, key = lambda x: x[self.col_index[orderBy]], reverse=not asc)
        
        results = []
        for row in rows_selected:
            results.append([row[self.col_index[col]] for col in cols])
        return results



class TestTable(unittest.TestCase):

    def setUp(self):
        self.table = Table(["id", "name", "age"])
        self.table.insertRow(["1", "Alice", "30"])
        self.table.insertRow(["2", "Bob", "25"])
        self.table.insertRow(["3", "Charlie", "35"])
        self.table.insertRow(["4", "Alice", "22"])

    def test_insertRow(self):
        self.table.insertRow(["5", "David", "28"])
        self.assertIn(["5", "David", "28"], self.table.rows)
        self.assertEqual(len(self.table.rows), 5)

    def test_delete_with_AND(self):
        # Delete rows where name="Alice" AND age="30"
        self.table.delete(["name", "age"], ["Alice", "30"], "AND")
        remaining_names = [r[self.table.col_index["name"]] for r in self.table.rows]
        self.assertNotIn("30", [r[self.table.col_index["age"]] for r in self.table.rows if r[self.table.col_index["name"]] == "Alice"])
        self.assertEqual(len(self.table.rows), 3)

    def test_delete_with_OR(self):
        # Delete rows where name="Bob" OR age="22"
        self.table.delete(["name", "age"], ["Bob", "22"], "OR")
        names = [r[self.table.col_index["name"]] for r in self.table.rows]
        self.assertNotIn("Bob", names)
        self.assertNotIn("22", [r[self.table.col_index["age"]] for r in self.table.rows])
        self.assertEqual(len(self.table.rows), 2)

    def test_query_with_AND(self):
        results = self.table.query(
            cols=["id", "name"],
            whereCols=["name", "age"],
            whereVals=["Alice", "30"],
            joiningOp="AND",
        )
        self.assertEqual(results, [["1", "Alice"]])

    def test_query_with_OR(self):
        results = self.table.query(
            cols=["id", "name"],
            whereCols=["name", "age"],
            whereVals=["Bob", "35"],
            joiningOp="OR",
        )
        # Should match Bob (id=2) and Charlie (id=3)
        self.assertEqual(sorted(results), sorted([["2", "Bob"], ["3", "Charlie"]]))

    def test_query_with_orderBy_asc(self):
        results = self.table.query(
            cols=["id", "name"],
            whereCols=["name"],
            whereVals=["Alice"],
            joiningOp="AND",
            orderBy="age",
            asc=True
        )
        # Alice (22) before Alice (30)
        self.assertEqual(results, [["4", "Alice"], ["1", "Alice"]])

    def test_query_with_orderBy_desc(self):
        results = self.table.query(
            cols=["id", "name"],
            whereCols=["name"],
            whereVals=["Alice"],
            joiningOp="AND",
            orderBy="age",
            asc=False
        )
        # Alice (30) before Alice (22)
        self.assertEqual(results, [["1", "Alice"], ["4", "Alice"]])

    def test_query_no_match(self):
        results = self.table.query(
            cols=["id"],
            whereCols=["name"],
            whereVals=["Nonexistent"],
            joiningOp="AND"
        )
        self.assertEqual(results, [])


unittest.main()