import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('BipedalWalkerHardcore-v3_sac_1.5.csv')
print(data.head(3))
data.cumsum()
data.plot()
plt.show()
