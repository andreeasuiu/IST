import sqlite3
conn = sqlite3.connect("feedbackdata.db")
c = conn.cursor()

def create_feedbacktable():
    c.execute('CREATE TABLE IF NOT EXISTS feedbacktable(interaction TEXT, results TEXT, speed TEXT, suggestions TEXT)')

def add_feedbackdata(interaction, results, speed, suggestions):
    c.execute('INSERT INTO feedbacktable(interaction, results, speed, suggestions) VALUES (?,?,?,?)',(interaction, results, speed, suggestions))
    conn.commit()

