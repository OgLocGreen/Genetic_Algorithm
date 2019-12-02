# coding=utf-8
import pandas as pd
import os
import openpyxl
from openpyxl import load_workbook

def write_cell(path_to_file,dataset,iterations,knn_size,small_dataset,algorithmus,acc,recall_score_var,precision_score_var,f1_score_var):
    book = load_workbook(path_to_file)
    #sheetX = book.active
    if small_dataset:
        dataset = str(dataset + "_small")
    if os.path.isfile(path_to_file):
        for sheet in book:   
            if sheet.title ==  dataset:
                if iterations == 10:
                  if knn_size == "small":
                        if algorithmus == "GA":
                            sheet["J3"] = acc
                            sheet["J4"] = recall_score_var
                            sheet["K3"] = precision_score_var
                            sheet["K4"] = f1_score_var 
                        if algorithmus == "RS":
                            sheet["J5"] = acc
                            sheet["J5"] = recall_score_var
                            sheet["K6"] = precision_score_var
                            sheet["K6"] = f1_score_var 
                if iterations == 50:
                    if knn_size == "small":
                        if algorithmus == "GA":
                            sheet["B3"] = acc
                            sheet["B4"] = recall_score_var
                            sheet["C3"] = precision_score_var
                            sheet["C4"] = f1_score_var
                        else:
                            sheet["B5"] = acc
                            sheet["B6"] = recall_score_var
                            sheet["C5"] = precision_score_var
                            sheet["C6"] = f1_score_var
                    elif knn_size == "big":
                        if algorithmus == "GA":
                            sheet["D3"] = acc
                            sheet["D4"] = recall_score_var
                            sheet["E3"] = precision_score_var
                            sheet["E4"] = f1_score_var
                        else:
                            sheet["D5"] = acc
                            sheet["D6"] = recall_score_var
                            sheet["E5"] = precision_score_var
                            sheet["E6"] = f1_score_var
                elif iterations == 250:
                    if knn_size == "small":
                        if algorithmus == "GA":
                            sheet["F3"] = acc
                            sheet["F4"] = recall_score_var
                            sheet["G3"] = precision_score_var
                            sheet["G4"] = f1_score_var
                        else:
                            sheet["F5"] = acc
                            sheet["F6"] = recall_score_var
                            sheet["G5"] = precision_score_var
                            sheet["G6"] = f1_score_var
                    elif knn_size == "big":
                        if algorithmus == "GA":
                            sheet["H3"] = acc
                            sheet["H4"] = recall_score_var
                            sheet["I3"] = precision_score_var
                            sheet["I4"] = f1_score_var
                        else:
                            sheet["H5"] = acc
                            sheet["H6"] = recall_score_var
                            sheet["I5"] = precision_score_var
                            sheet["I6"] = f1_score_var

        book.save(filename=path_to_file)


if __name__ == "__main__":

    excel_file = "/../../data/evaluation.xlsx"
    dataset = "mnist_fashion"
    path_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__))+ excel_file)

    write_cell(path_to_file= path_to_file,dataset= dataset, iterations = 250,small_dataset =True, knn_size = "small", algorithmus= "GA", acc = 0.9 , recall_score_var = 0.8,precision_score_var= 0.8,f1_score_var =0.9)
    write_cell(path_to_file= path_to_file,dataset= dataset, iterations = 250,small_dataset =False, knn_size = "small", algorithmus= "GA", acc = 0.9 , recall_score_var = 0.8,precision_score_var= 0.8,f1_score_var =0.9)
    write_cell(path_to_file= path_to_file,dataset= dataset, iterations = 250,small_dataset =False, knn_size = "small", algorithmus= "RS", acc = 0.9 , recall_score_var = 0.8,precision_score_var= 0.8,f1_score_var =0.9)
    write_cell(path_to_file= path_to_file,dataset= dataset, iterations = 250,small_dataset =True, knn_size = "small", algorithmus= "RS", acc = 0.9 , recall_score_var = 0.8,precision_score_var= 0.8,f1_score_var =0.9)
