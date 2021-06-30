"""calculate the average emissivity and opacity of a cell"""
import numpy
import pandas
def average_emissivity(file_name):
    avg_file = file_name+".avg"
    df = pandas.read_csv(file_name, delimiter="\t")
    if df.shape[0]==1:
        df.to_csv(avg_file)
    elif df.shape[0]>1:
        dr = df["#depth"]
        pass

def average_opacity(file_name):
    df = pandas.read_csv(file_name, delimiter="\t")
    if df.shape[0]>=1:
        #Should check if 
        pass

if __name__ == 'main':
    average_emissivity(None)
    average_opacity(None)
    pass