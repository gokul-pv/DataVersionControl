import pathlib as pl
import pandas as pd

def assetIsFilePresent(path):
    if path.is_file():
        raise AssertionError("File %s present in : %s" % (str(path),str(path.resolve())))


def test1():
    path = pl.Path("data.zip")
    assetIsFilePresent(path)

def test2():
    path = pl.Path("model.h5")
    assetIsFilePresent(path)  

def test3():
  df = pd.read_csv('metrics.csv')
  index = df.index
  number_of_rows = len(index)
  last_row = df.iloc[number_of_rows - 1]
  acc_check = last_row['Test accuracy'] < 0.7
  if acc_check: 
    raise AssertionError("Model accurarcy is less than 70%")

def test4():
  df = pd.read_csv('metrics.csv')
  index = df.index
  number_of_rows = len(index)
  last_row = df.iloc[number_of_rows - 1]
  class_acc_check = last_row['Accurarcy of Cat'] < 70 or last_row['Accurarcy of dog'] < 70
  if class_acc_check: 
    raise AssertionError("Class wise accurarcy is less than 70%")   