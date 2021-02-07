import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
import os
import sys
import re
import matplotlib.pyplot as plt

currDir = os.getcwd()

df = pd.read_csv(os.path.abspath("..") + "/Data/finance.csv", header=None)
with open(os.path.abspath("..") + "/Data/DataIndex.csv", 'r') as f:
	df_index = f.read()

df_index = df_index.replace("\xa0", "").replace(" = ", ",")
split = re.search("(?<=\\n)..+(?=column 0)", df_index).span()[0]
rows = df_index[:(split - 1)].replace("row ", "")
cols = df_index[split:].replace("column ", "")

rows = pd.DataFrame([row.split(",") for row in rows.split("\n")])
rows.columns = ['Date', 'Row']

cols = pd.DataFrame([col.split(",") for col in cols.split("\n")])
cols.columns = ['Ticker', 'Column']

df.columns = cols['Ticker'].values
df.index = rows['Date'].values

emp_cov = np.matmul(df.values.T, df.values)

cov = GraphicalLasso(max_iter=300, alpha=1.5, tol=0.01, enet_tol=0.01).fit(df)
covCV = GraphicalLassoCV().fit(df.values)

G = nx.from_numpy_matrix(cov.get_precision())

nodes = list(G.nodes(data=True))
names_dict = {val[0]: cols['Ticker'].values[i] for i, val in enumerate(nodes)}
G = nx.relabel_nodes(G, names_dict)

graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
graphs = [g for g in graphs if len(g) > 1]

amzn = G.subgraph([list(names_dict.values()).index('GOOG')])

nx.draw_networkx(graphs[0], pos=nx.spring_layout(graphs[0]),
	node_size=[len(v) * 200 for v in graphs[0].nodes()])
plt.show()

nx.draw_networkx(graphs[0], pos=nx.spring_layout(graphs[0]),
	node_size=[len(v) * 200 for v in graphs[0].nodes()])
plt.show()