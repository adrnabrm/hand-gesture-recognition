import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

# read csv
df = pd.read_csv('not_peace.csv')
print(df)
# converts the string literals into lists

shape = df.shape
print(shape)
for x in df['coords']:
    coords = np.array(ast.literal_eval(x))
    xs = np.array([x for x, y in coords])
    ys = np.array([y for x, y in coords])
    colors = np.random.rand(21)
    plt.scatter(xs,-ys, c=colors, cmap='viridis')
plt.show()

# CALC DISTANCES BETWEEN EACH NODE
plt.show()


