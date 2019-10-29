# coding=utf-8
import pandas as pd
import os
from openpyxl import load_workbook

def excel_file(sheet="mnist_fashion",iterations= 50 , knn_size = "small"):

    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__))+ "/evaluation.xlsx")
    book = load_workbook(path_to_file)
    writer = pd.ExcelWriter(path_to_file, engine = 'openpyxl')
    writer.book = book

    
    if os.path.isfile(path_to_file):
        xls = pd.ExcelFile(path_to_file)

        sheetX = xls.parse(sheet)
        print(sheetX)
        if sheet == "mnist_fashion":
            for i in ["aaa","bbb","ccc","ddd","eee","fff","ggg","hhh","jjj"]:
                var1 = sheetX.ix[:,i]
                if var1[0] == iterations:
                    if var1[1] == knn_size:
                        var1[2] = 0.2
                        var1[3] = 0.3
                sheetX.ix[:,i] = var1
        sheetX.to_excel(path_to_file, sheet_name=sheetX,index=False)
        writer.save()
        writer.close()
        print(var1)



if __name__ == "__main__":

    excel_file(sheet="mnist_fashion")