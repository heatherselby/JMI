import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
import numpy as np

def barplot_err(x, y, xerr=None, yerr=None, data=None, **kwargs):

    _data = []
    for _i in data.index:

        _data_i = pd.concat([data.loc[_i:_i]]*3, ignore_index=True, sort=False)
        _row = data.loc[_i]
        if xerr is not None:
            _data_i[x] = [_row[x]-_row[xerr], _row[x], _row[x]+_row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y]-_row[yerr], _row[y], _row[y]+_row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    _ax = sns.barplot(x=x,y=y,data=_data,errorbar='sd',**kwargs)

    return _ax

if __name__ == '__main__':
    # df = pd.read_excel("Radiology attending timings.xlsx", sheet_name="Sheet2")
    # percentiles = [0.5]
    # quantile_funcs = [(p, lambda x: x.quantile(p)) for p in percentiles]
    # df1 = df.groupby(['Segmentation', 'Diagnosis'])["Time"].agg(quantile_funcs).reset_index()
    # df1["Segmentation"] = df1["Segmentation"].str.replace("FULL", "FM")
    # cat_segmentation = CategoricalDtype(
    #     ["FM", "BB", "RB"],
    #     ordered=True
    # )
    # df1["Segmentation"]=df1["Segmentation"].astype(cat_segmentation)
    # df1.sort_values(["Segmentation"], inplace=True)
    # df1.columns = ['Diagnosis', 'Segmentation', "Time"]
    # print(df1)
    print("Heather")
    df = pd.read_excel("Radiology attending timings.xlsx", sheet_name="Sheet2")
    # # df1 = df.groupby(['Diagnosis', 'Segmentation']).median(numeric_only=True).reset_index()
    # df1 = df.groupby(['Diagnosis', 'Segmentation'], as_index=False).agg({'Time': ['mean', 'sem']})
    # df1.columns = ['Diagnosis', 'Segmentation', 'Mean', 'SEM']
    df["Segmentation"] = df["Segmentation"].str.replace("FULL", "FM")
    #
    # df = df[["Segmentation", "Time"]]
    # df2 = df.groupby(['Segmentation'], as_index=False).agg({'Time': ['mean', 'sem']})
    # df2.columns = ['Segmentation', 'Mean', 'SEM']
    # df2["Segmentation"] = df2["Segmentation"].str.replace("FULL", "FM")
    # df2.to_csv("Timings1.csv")
    #
    # df1.sort_values(["Segmentation"], inplace=True)
    # df1.to_csv("Timings.csv")
    # duplicates = 1000
    # # duplicate observations to get good std bars
    # dfCopy = df1.loc[df1.index.repeat(duplicates)].copy()
    # dfCopy['Y'] = np.random.normal(dfCopy['Mean'].values, dfCopy['SEM'].values)
    cat_segmentation = CategoricalDtype(
        ["FM", "BB", "RB"],
        ordered=True
    )
    df["Segmentation"] = df["Segmentation"].astype(cat_segmentation)
    #
    params = {'legend.fontsize': 10,
              'legend.title_fontsize': 11,
              'axes.titlesize': 11,
              'axes.labelsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'lines.linewidth': 1,
              'lines.markersize': 3,
              'xtick.bottom': True,
              'ytick.left': True,
              }
    sns.set(style="white", context="paper", font="Arial", rc=params)
    #
    # # fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,
    # #                                sharex=True, gridspec_kw={'height_ratios': [1, 1.8]})
    #
    # # fig, ax1 = plt.subplots(1,1)
    # #
    # # fig.tight_layout()

    colors = ["#c03266", "#e0b227", "#3b85c6"]
    colors = ["#c03266", "#e0b227", "#3b85c6", "#004d41"]
    fig, ax = plt.subplots(1,1)
    ax = sns.boxplot(x='Diagnosis', y='Time', hue='Segmentation', palette=colors, order=["Benign", "ADC", "SCLC"],
                     medianprops={"color": "black"}, data=df)
    # ax = barplot_err(x='Diagnosis', y='Mean', hue='Segmentation', order=["Benign", "ADC", "SCLC"], yerr="SEM", capsize=.05, errwidth=1, palette=colors, data=df1)
    # plt.savefig('timing.png', dpi=300, bbox_inches='tight')

    # g = sns.catplot(x="Diagnosis", y="Time", data=df1, kind="bar", errorbar="sd",
    #                   hue="Segmentation", palette=colors, order=["Benign", "ADC", "SCLC"])
    # g.set_axis_labels("Nodules", "Segmentation Times (seconds)")
    # g.legend.set_title("Segmentation Methods")
    # g.set(yscale="log")
    # plt.savefig('timing.png', dpi=300, bbox_inches='tight')

    # ax2 = sns.barplot(x="Diagnosis", y="Time",
    #                   hue="Segmentation", order=["Benign", "ADC", "SCLC"], capsize=.1, errorbar=('ci', 68), width=0.6,
    #                   data=df1, palette=colors, ax=ax2)
    # for container in ax2.containers:
    #     ax2.bar_label(container)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt = '%.1f')
    # ax1.set_ylim(500, 750)
    # ax2.set_ylim(0, 125)
    ax.set_ylabel("Segmentation Time (seconds)")
    # ax2.set_ylabel(None)
    # ax1.get_xaxis().set_visible(True)
    ax.set_xlabel("Nodule")
    # fig.text(0.05, 0.55, "Segmentation Time (seconds)", va="center", rotation="vertical", fontsize=10)
    # ax2.get_legend().remove()
    ax.legend(title="Segmentation Method")._legend_box.align = "left"
    # ax1.xaxis.tick_bottom()
    # fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
    # d = .01
    # kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    # ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    plt.yscale("log")
    plt.savefig('timings4.png', dpi=300, bbox_inches='tight')
