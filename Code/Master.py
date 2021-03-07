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
# Section 5: Centrality Analysis
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

times_dict = {'OG':[], 'EXP':[]}
stocks_dict = times_dict.copy()

stocks_dict['OG'] = [2, 321, 30, 241, 48, 180]
stocks_dict['EXP'] = stocks_dict['OG'].copy() + list(range(50, 75))

def timing_set(center, samplesPerStep_left, count_left, samplesPerStep_right, count_right ):
    time_set = []
    count_left = min(count_left, center/samplesPerStep_left)
    # print 'left timesteps: = ', count_left
    start = max(center- samplesPerStep_left*(count_left), 0)
    for i in range(count_left):
        time_interval = [start, start + samplesPerStep_left -1]
        time_set.append(time_interval)
        start = start + samplesPerStep_left
    count_right = min(count_right, 245/samplesPerStep_left)
    # print 'right timesteps: = ', count_right
    for i in range(count_right):
        time_interval = [start, start + samplesPerStep_right -1]
        time_set.append(time_interval)
        start = start + samplesPerStep_right
    return time_set

samps = 3
times_dict['OG'] = timing_set(102, samps, 3, samps, 13)
times_dict['EXP'] = timing_set(102, samps, 25, samps, 45)

df = df.iloc[time_set[0][0]:time_set[-1][1], stock_cols]

################################################################################
# Section 3
################################################################################

#### ESTIMATION ###

emp_cov = np.matmul(df.values.T, df.values)

cov = GraphicalLasso(max_iter=300, alpha=1.5, tol=0.01, enet_tol=0.01).fit(df)
covCV = GraphicalLassoCV().fit(df.values)

G = nx.from_numpy_matrix(cov.get_precision())

alpha = 0.18

cov = GraphicalLasso(max_iter=300, alpha=alpha, tol=0.01, enet_tol=0.01, verbose=True).fit(df)
# covCV = GraphicalLassoCV(alphas=list(0.18), max_iter=300, tol=0.01, enet_tol=0.01, verbose=True).fit(df)

#### VISUALIZATION ###

# pCV = covCV.precision_  # get precision matrix
p = cov.precision_

fig, ax = plt.subplots(1, 1)

sns.heatmap(p, annot=True, xticklabels=stock_names, yticklabels=stock_names, ax=ax)
ax.set_title('Static Lasso Perturbed Node Alpha ' + str(alpha))
plt.savefig(outputDir + "/Static_Psi5_alpha" + str(alpha) + ".png")
plt.close()

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

def heats(penalty, alpha, beta, times_dict, stocks_dict, time_exp, stock_exp):

    with open(os.path.abspath(".") + "/Data/DataIndex.csv", 'r') as f:
        df_index = f.read()

    df_index = df_index.replace("\xa0", "").replace(" = ", ",")
    split = re.search("(?<=\\n)..+(?=column 0)", df_index).span()[0]
    rows = df_index[:(split - 1)].replace("row ", "")
    cols = df_index[split:].replace("column ", "")

    rows = pd.DataFrame([row.split(",") for row in rows.split("\n")])
    rows.columns = ['Date', 'Row']
    rows['Date'] = pd.to_datetime(rows['Date'].apply(lambda x: int(x[:-2])), format='%Y%m%d').astype(str)
    rows = dict(zip(rows['Row'], rows['Date']))

    cols = pd.DataFrame([col.split(",") for col in cols.split("\n")])
    cols.columns = ['Ticker', 'Column']
    cols = dict(zip(cols['Column'], cols['Ticker']))

    currSuff = "Psi%s"%penalty + "Alpha%s"%alpha + "Beta%s"%beta + ["", "TimeExpand"][time_exp] + ["", "StockExpand"][stock_exp]

    theta_est_file = "/Data/theta_est" + currSuff + ".npy"
    theta_diff_file = "/Data/theta_diff" + currSuff + ".npy"

    theta_est = np.load(os.getcwd() + theta_est_file)
    theta_diff = np.load(os.getcwd() + theta_diff_file)

    times = times_dict[['OG', 'EXP'][time_exp]]
    dates = [rows[str(i[0])] for i in times]

    stocks = stocks_dict[['OG', 'EXP'][stock_exp]]
    stock_names = [cols[str(i)] for i in stocks]

    pngfile = "PrecMats" + currSuff

    for i in range(len(dates)):

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

        sns.heatmap(theta_est[i], annot=True, xticklabels=stock_names, yticklabels=stock_names, ax=ax[0])
        ax[0].set_title('Precision Matrix: ' + dates[i])
        sns.heatmap(theta_diff[i], annot=True, xticklabels=stock_names, yticklabels=stock_names, ax=ax[1], vmin=min([theta_diff[i].min() for i in range(theta_diff.shape[0])]), vmax=max([theta_diff[i].max() for i in range(theta_diff.shape[0])]))
        ax[1].set_title('Delta Precision Matrix ' + dates[i])

        plt.savefig(outputDir + "/" + pngfile + str(i) + ".png")
        plt.close()

heats(1, 0.18, 13, times_dict, stocks_dict, False, False)
heats(2, 0.18, 13, times_dict, stocks_dict, False, False)
heats(3, 0.18, 13, times_dict, stocks_dict, False, False)
heats(5, 0.18, 13, times_dict, stocks_dict, False, False)
heats(5, 0.18, 13, times_dict, stocks_dict, True, False)

# G = nx.from_numpy_matrix(np.matrix(theta_est[0]), create_using=nx.Graph)
# layout = nx.circular_layout(G)
# nx.draw(G, layout, ax=ax[1])
# nx.draw_networkx_labels(G, pos=layout, labels=dict(zip(range(6), stock_names)))

################################################################################
# Section 5
################################################################################

def central(penalty, alpha, beta, times_dict, stocks_dict, time_exp, stock_exp, centrality):

    with open(os.path.abspath(".") + "/Data/DataIndex.csv", 'r') as f:
        df_index = f.read()

    df_index = df_index.replace("\xa0", "").replace(" = ", ",")
    split = re.search("(?<=\\n)..+(?=column 0)", df_index).span()[0]
    rows = df_index[:(split - 1)].replace("row ", "")
    cols = df_index[split:].replace("column ", "")

    rows = pd.DataFrame([row.split(",") for row in rows.split("\n")])
    rows.columns = ['Date', 'Row']
    rows['Date'] = pd.to_datetime(rows['Date'].apply(lambda x: int(x[:-2])), format='%Y%m%d')
    rows = dict(zip(rows['Row'], rows['Date']))

    cols = pd.DataFrame([col.split(",") for col in cols.split("\n")])
    cols.columns = ['Ticker', 'Column']
    cols = dict(zip(cols['Column'], cols['Ticker']))

    currSuff = "Psi%s"%penalty + "Alpha%s"%alpha + "Beta%s"%beta + ["", "TimeExpand"][time_exp] + ["", "StockExpand"][stock_exp]

    theta_est_file = "/Data/theta_est" + currSuff + ".npy"

    theta_est = np.load(os.getcwd() + theta_est_file)

    times = times_dict[['OG', 'EXP'][time_exp]]
    dates = {i:rows[str(times[i][0])] for i in range(len(times))}

    stocks = stocks_dict[['OG', 'EXP'][stock_exp]]
    stock_names = {i:cols[str(stocks[i])] for i in range(len(stocks))}

    pngfile = centrality + "Centrality" + currSuff

    central_list = []

    for i in range(len(theta_est)):
        G = nx.Graph(nx.from_numpy_matrix(np.matrix(theta_est[i]), create_using=nx.DiGraph))

        if centrality == 'EigenVector':
            central = nx.eigenvector_centrality(G, weight='weight')
        elif centrality == 'Betweenness':
            central = nx.betweenness_centrality(G, weight='weight')
        elif centrality == 'Katz':
            central = nx.katz_centrality(G, weight='weight')
        elif centrality == 'Information':
            central = nx.information_centrality(G, weight='weight')

        central_list.append(central)

    vals = pd.DataFrame.from_dict(central_list).unstack().reset_index().rename({'level_0':'StockIndex', 'level_1': 'Times', 0 : centrality}, axis=1)

    vals['StockNames'] = vals['StockIndex'].map(stock_names)
    vals['Date'] = vals['Times'].map(dates)

    return(vals)

centrality = 'Katz'
vals = central(3, 0.18, 13, times_dict, stocks_dict, False, False, centrality)
sns.lineplot(x='Date', y=centrality, hue='StockNames', data=vals)
plt.show()

