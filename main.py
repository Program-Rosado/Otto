from sklearn.preprocessing import StandardScaler
from pandas import get_dummies, concat, read_json
from matplotlib.pyplot import hist, show, boxplot, subplots
from numpy import arange, argmax
from sklearn.ensemble import RandomForestClassifier
# import numpy as np
import seaborn as sns
from numpy import arange, argmax
import pandas as pd
from pathlib import Path
import gc
import datetime
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\rosadom\Documents\DSFoundations\MainProject_DSFoundations\otto_training_data.csv")
print(df.head(10))
sns.pairplot(df[['aid', 'ts', 'type']])
