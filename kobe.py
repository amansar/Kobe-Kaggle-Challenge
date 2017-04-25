"""
K     K     O O O     B B B     E E E E E
K   K     O       O   B     B   E
K K       O       O   B B B     E E E E E
K K       O       O   B     B   E
K   K     O       O   B     B   E
K     K     O O O     B B B     E E E E E


For preprocessing, get rid of data points that don't have shot made field.
Create test/training splits from data that does have shot made field.
"""
import csv
from sklearn.ensenmble import *
import pandas as pd
import numpy as np
