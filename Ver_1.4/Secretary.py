import xlsxwriter
from datetime import datetime

class Secretary():
    data_path = "./data/xlsx/"
    def __init__(self):
        self.date = str(datetime.date(datetime.now()))
        self.workbook = xlsxwriter.Workbook("{}{}.xlsx".format(self.data_path, self.date))
        self.worksheet = self.workbook.add_worksheet()
        self.worksheet.set_column('A:A', 50)
        self.worksheet.set_column('B:B', 15)
        self.worksheet.set_column('C:C', 10)
        self.worksheet.set_column('D:D', 20)
        self.worksheet.write('A1', 'Họ và Tên')
        self.worksheet.write('B1', 'ID')
        self.worksheet.write('C1', 'Nhiệt độ')
        self.worksheet.write('D1', 'Thời gian')
        self.count = 2


    def note(self, name, ID, temp=None):
        time = datetime.time(datetime.now())
        self.worksheet.write('A{}'.format(self.count), name)
        self.worksheet.write('B{}'.format(self.count), ID)
        self.worksheet.write('C{}'.format(self.count), temp)
        self.worksheet.write('D{}'.format(self.count), str(time)[:7])
        self.count = self.count+1
        self.workbook.close()