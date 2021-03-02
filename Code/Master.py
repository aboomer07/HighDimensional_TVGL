import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
import os
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Force the correct directory
if os.getcwd().split("/")[-1] == "Code":
    os.chdir("..")
currDir = os.getcwd()

# If output directory does not already exist, create one
if not os.path.isdir("Output"):
    os.mkdir("Output")
outputDir = currDir + "/Output/"

currDir = os.getcwd()

df = pd.read_csv(os.path.abspath(".") + "/Data/finance.csv", header=None)
with open(os.path.abspath(".") + "/Data/DataIndex.csv", 'r') as f:
    df_index = f.read()

#### DATA PREP ###

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

df = df.sample(50, axis=1)

#### ESTIMATION ###

emp_cov = np.matmul(df.values.T, df.values)

cov = GraphicalLasso(max_iter=300, alpha=1.5, tol=0.01, enet_tol=0.01).fit(df)
covCV = GraphicalLassoCV().fit(df.values)

G = nx.from_numpy_matrix(cov.get_precision())

cov = GraphicalLasso(max_iter=300, alpha=0.01, tol=0.01, enet_tol=0.01, verbose=True).fit(df)
covCV = GraphicalLassoCV().fit(df.values)

#### VISUALIZATION JACOB ###

p = cov.precision_  # get precision matrix

sns.heatmap(p)  # get a quick look
plt.show()

# prepare the matrix for network illustration
p = pd.DataFrame(p, columns=df.columns, index=df.columns)
links = p.stack().reset_index()
links.columns = ['var1', 'var2', 'value']
links = links.loc[(abs(links['value']) > 0.17) & (links['var1'] != links['var2'])]

# build the graph using networkx lib
G = nx.from_pandas_edgelist(links, 'var1', 'var2', create_using=nx.Graph())
pos = nx.spring_layout(G, k=0.2 * 1 / np.sqrt(len(G.nodes())), iterations=20)
plt.figure(3, figsize=(15, 15))
nx.draw(G, pos=pos)
nx.draw_networkx_labels(G, pos=pos)
plt.savefig(outputDir + 'NetworkGraph.png')
plt.close()


#### ALPHA RANGE ###

def output_lasso(alpha, data=df):
    cov = GraphicalLasso(max_iter=300, alpha=alpha, tol=0.01, enet_tol=0.01, verbose=True).fit(data)

    precision = cov.precision_  # get precision matrix

    # prepare the matrix for network illustration
    precision = pd.DataFrame(precision, columns=df.columns, index=df.columns)
    links = precision.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    links = links.loc[(abs(links['value']) > 0.17) & (links['var1'] != links['var2'])]

    # build the graph using networkx lib
    G = nx.from_pandas_edgelist(links, 'var1', 'var2', create_using=nx.Graph())
    pos = nx.spring_layout(G, k=0.2 * 1 / np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(15, 15))
    nx.draw(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos)
    plt.savefig(outputDir + "NetworkGraph_Alpha" + str(alpha) + ".png")
    plt.close()

alphas = [0.005, 0.01, 0.05, 0.1]

for alpha in alphas:
    output_lasso(alpha)


#### VISUALIZATION ANDY ###

G = nx.from_numpy_matrix(cov.covariance_)

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
