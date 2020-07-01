import pandas as pd
import pdb


#constants
DIABETIC_ATTRIBUTES = 19
DIGITS_ATTRIBUTES = 16

def read_diabetic_data(filename):
    attributes_df = pd.read_csv(filename, header = None, usecols = range(DIABETIC_ATTRIBUTES))
    classes_df = pd.read_csv(filename, header = None, usecols = [DIABETIC_ATTRIBUTES])
    return attributes_df, classes_df

def read_digits_data(filename):
    attributes_df = pd.read_csv(filename, header = None, usecols = range(DIGITS_ATTRIBUTES))
    classes_df = pd.read_csv(filename, header = None, usecols = [DIGITS_ATTRIBUTES])
    return attributes_df, classes_df
