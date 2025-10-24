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