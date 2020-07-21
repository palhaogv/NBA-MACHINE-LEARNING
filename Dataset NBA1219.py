import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import func
from func import data_set


DB_test, DB_test_a, DB_test_g = func.testing_1920()
DB_test = DB_test.groupby('MONTH').mean()

