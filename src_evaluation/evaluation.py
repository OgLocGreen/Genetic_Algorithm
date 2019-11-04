# coding=utf-8
import pandas as pd
import os
import openpyxl
from openpyxl import load_workbook

def write_cell(path_to_file,dataset,iterations,knn_size,small_dataset,algorithmus,acc):
    book = load_workbook(path_to_file)
    #sheetX = book.active
    if small_dataset == True:
        dataset = str(dataset + "_small")

    if os.path.isfile(path_to_file):
        for sheet in book:   
            if sheet.title ==  dataset:
                if iterations == 50:
                    if knn_size == "small":
                        if algorithmus == "GA":
                            sheet["B3"] = acc
                        else:
                            sheet["B4"] = acc
                    elif knn_size == "medium":
                        if algorithmus == "GA":
                            sheet["C3"] = acc
                        else:
                            sheet["C4"] = acc
                    elif knn_size == "big":
                        if algorithmus == "GA":
                            sheet["D3"] = acc
                        else:
                            sheet["D4"] = acc
                elif iterations == 250:
                    if knn_size == "small":
                        if algorithmus == "GA":
                            sheet["E3"] = acc
                        else:
                            sheet["E4"] = acc
                    elif knn_size == "medium":
                        if algorithmus == "GA":
                            sheet["F3"] = acc
                        else:
                            sheet["F4"] = acc
                    elif knn_size == "big":
                        if algorithmus == "GA":
                            sheet["G3"] = acc
                        else:
                            sheet["G4"] = acc
                elif iterations == 500:
                    if knn_size == "small":
                        if algorithmus == "GA":
                            sheet["H3"] = acc
                        else:
                            sheet["H4"] = acc
                    elif knn_size == "medium":
                        if algorithmus == "GA":
                            sheet["I3"] = acc
                        else:
                            sheet["I4"] = acc
                    elif knn_size == "big":
                        if algorithmus == "GA":
                           sheet["J3"] = acc
                        else:
                            sheet["J4"] = acc

        book.save(filename=path_to_file)


if __name__ == "__main__":

    excel_file = "/evaluation.xlsx"
    dataset = "mnist_fashion"
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__))+ excel_file)

    #excel_file(path_to_file = path_to_file, sheet="mnist_fashion")
    write_cell(path_to_file= path_to_file,dataset= dataset, iterations = 250, knn_size = "small", algorithmus= "GA", acc = 0.9)