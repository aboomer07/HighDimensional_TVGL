################################################################################
# High Dimensional Project Master Code
################################################################################

################################################################################
############################# TABLE OF CONTENTS ################################
################################################################################

################################################################################
# Section 1: Load Libraries and Directory Definition
################################################################################

################################################################################
# Section 2: Import Data and Data Prep
################################################################################

################################################################################
# Section 2: Import Data and Data Prep
################################################################################

################################################################################
# Section 3: Estimation and Plotting of Graphical LASSO
################################################################################

################################################################################
# Section 4: Time Varying Graphical LASSO
################################################################################

################################################################################
########################## END TABLE OF CONTENTS ###############################
################################################################################

################################################################################
# Section 1
################################################################################

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

################################################################################
# Section 2
################################################################################

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

dates = ['2010-01-13', '2010-01-19', '2010-01-22', '2010-01-27', '2010-02-01',
    '2010-02-04', '2010-02-09', '2010-02-12', '2010-02-18', '2010-02-23',
    '2010-02-26', '2010-03-03', '2010-03-08', '2010-03-11', '2010-03-16',
    '2010-03-19']

stock_cols = [2, 321, 30, 241, 48, 180]

stock_names = ['Apple', 'MSFT', 'AMAZON', 'Intel', 'Boeing', 'Fedex']

time_set = [[93, 95], [96, 98], [99, 101], [102, 104], [105, 107], [108, 110], [111, 113], [114, 116], [117, 119], [120, 122], [123, 125], [126, 128], [129, 131], [132, 134], [135, 137], [138, 140]]

df = df.iloc[time_set[0][0]:time_set[-1][1], stock_cols]

################################################################################
# Section 3
################################################################################

#### ESTIMATION ###

emp_cov = np.matmul(df.values.T, df.values)

cov = GraphicalLasso(max_iter=300, alpha=1.5, tol=0.01, enet_tol=0.01).fit(df)
covCV = GraphicalLassoCV().fit(df.values)

G = nx.from_numpy_matrix(cov.get_precision())

cov = GraphicalLasso(max_iter=300, alpha=0.01, tol=0.01, enet_tol=0.01, verbose=True).fit(df)
covCV = GraphicalLassoCV(alphas=list(np.linspace(0, 1.5, 50)), max_iter=300, tol=0.01, enet_tol=0.01, verbose=True).fit(df)

#### VISUALIZATION ###

p = covCV.precision_  # get precision matrix

sns.heatmap(p, annot=True, xticklabels=stock_names, yticklabels=stock_names)
plt.set_title('Delta Precision Matrix ' + dates[i])
plt.show()

#### ALPHA RANGE ###

def output_lasso(alpha, layout='circular', data=df):
    cov = GraphicalLasso(max_iter=400, alpha=alpha, tol=0.01, enet_tol=0.01, verbose=True).fit(data)

    precision = cov.precision_  # get precision matrix

    # prepare the matrix for network illustration
    precision = pd.DataFrame(precision, columns=df.columns, index=df.columns)
    links = precision.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    links = links.loc[(abs(links['value']) > 0.17) & (links['var1'] != links['var2'])]

    # build the graph using networkx lib
    G = nx.from_pandas_edgelist(links, 'var1', 'var2', create_using=nx.Graph())

    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.2 * 1 / np.sqrt(len(G.nodes())), iterations=20)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'kk':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)

    plt.figure(3, figsize=(15, 15))
    nx.draw(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos, font_size=15)
    plt.savefig(outputDir + "NetworkGraph_Alpha" + str(alpha) + "_Layout" + layout +  ".png")
    plt.close()

alphas = [0.005, 0.01, 0.05, 0.1]

for alpha in alphas:
    output_lasso(alpha, layout='random')

################################################################################
# Section 4
################################################################################

empcov = np.load(os.getcwd() + "/Data/empcov.npy")
cov = np.load(os.getcwd() + "/Data/cov.npy")
theta_diff = np.load(os.getcwd() + "/Data/theta_diff.npy")
theta_est = np.load(os.getcwd() + "/Data/theta_est.npy")

if not os.path.isdir(outputDir + "/empcov"):
    os.mkdir(outputDir + "/empcov")
empcovDir = outputDir + "/empcov/"

if not os.path.isdir(outputDir + "/cov"):
    os.mkdir(outputDir + "/cov")
covDir = outputDir + "/cov/"

if not os.path.isdir(outputDir + "/theta_diff"):
    os.mkdir(outputDir + "/theta_diff")
covDir = outputDir + "/theta_diff/"

if not os.path.isdir(outputDir + "/theta_est"):
    os.mkdir(outputDir + "/theta_est")
covDir = outputDir + "/theta_est/"

G = nx.from_numpy_matrix(np.matrix(theta_est[0]), create_using=nx.Graph)
layout = nx.circular_layout(G)
nx.draw(G, layout, ax=ax[1])
nx.draw_networkx_labels(G, pos=layout, labels=dict(zip(range(6), stock_names)))

for i in range(len(dates)):

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    sns.heatmap(theta_est[i], annot=True, xticklabels=stock_names, yticklabels=stock_names, ax=ax[0])
    ax[0].set_title('Precision Matrix: ' + dates[i])
    sns.heatmap(theta_diff[i], annot=True, xticklabels=stock_names, yticklabels=stock_names, ax=ax[1], vmin=min([theta_diff[i].min() for i in range(theta_diff.shape[0])]), vmax=max([theta_diff[i].max() for i in range(theta_diff.shape[0])]))
    ax[1].set_title('Delta Precision Matrix ' + dates[i])

    plt.savefig(outputDir + "/PrecMats" + str(i).zfill(3) + ".png")
    plt.close()

os.chdir("Output")
os.system(
    "ffmpeg -framerate 1 -i PrecMats%3d.png -c:v h264 -r 30 -s 1920x1080 ./Heatmaps.mp4")
os.chdir("..")
