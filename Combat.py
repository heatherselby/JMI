import pandas as pd
from neuroCombat import neuroCombat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':

    FM= pd.read_csv("VA_SF_FM_Features.csv")
    FM.rename(columns={'Unnamed: 0': 'Patient'}, inplace=True)

    # samples = pd.read_csv("VA_PA_RAJ_FM_Features.csv")
    # samples.rename(columns={'Unnamed: 0': 'Patient'}, inplace=True)
    # samples = samples["Patient"].tolist()

    # features = pd.read_csv("VA_PA_FM_ICC_MS.csv")

    diagnosis = pd.read_excel("Diagnosis_Key_2.xlsx")
    meta = pd.read_csv("Meta.csv")

    # FM = FM[~FM["Patient"].isin(samples)]
    dat = FM.copy()

    annot = meta.merge(diagnosis, on="Patient", how="inner")

    dat = dat[dat.Patient.isin(annot.Patient)]
    annot = annot[annot.Patient.isin(dat.Patient)]

    annot.sort_values(by="Patient", inplace=True)
    dat.sort_values(by="Patient", inplace=True)

    dat.set_index("Patient", inplace=True)
    annot.set_index("Patient", inplace=True)

    annot["Diagnosis"] = annot["Diagnosis"].str.strip()
    annot.Diagnosis.replace(to_replace=["Adenocarcinoma", "Small cell lung cancer", "Squamous cell carcinoma",
                                        "Non-small cell lung cancer"],
                            value="Malignant", inplace=True)
    print(dat.shape)

    scaler = StandardScaler()
    scaler.fit(dat)
    scaled_data = scaler.transform(dat)

    dat_scaled = pd.DataFrame(scaled_data, columns=dat.columns)

    # covars = pd.DataFrame({'Batch': annot["Contrast"].map({'N': 1, 'Y': 2}),
    #                        'Diagnosis': annot["Diagnosis"].map({'Benign': 1, 'Malignant': 2})})
    # categorical_cols = ['Diagnosis']
    # batch_col = 'Batch'
    #
    # dat1 = dat_scaled.transpose()
    # data_combat = neuroCombat(dat=dat1,
    #                           covars=covars,
    #                           batch_col=batch_col,
    #                           categorical_cols=categorical_cols)["data"]
    # dat_scaled = pd.DataFrame(data_combat.transpose(), columns=dat.columns, index=dat.index)

    X = dat_scaled.copy()
    pca = PCA(n_components=4)
    pca.fit(X)
    x_pca = pca.transform(X)

    colors = ["#5f3d97", "#e46225"]

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

    fig, ax = plt.subplots()
    fig = sns.jointplot(data=x_pca, x=x_pca[:,0], y=x_pca[:,1], hue=annot["Contrast"].tolist(), s=50, palette=colors,
                        edgecolor='black', linewidth=2)
    fig.set_axis_labels(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f} %)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f} %)")
    plt.legend(title="Contrast", markerscale=1)._legend_box.align = "left"
    texts = fig.ax_joint.legend_.texts
    for t, label in zip(texts, ["NCE", "CE"]):
        t.set_text(label)
    plt.suptitle("Dataset 2\nBefore ComBat", x=0, y=.95, horizontalalignment='left', verticalalignment='top')
    plt.savefig('VA_SF_Before.png', dpi=300, bbox_inches='tight')

    covars = pd.DataFrame({'Batch': annot["Contrast"].map({'N': 1, 'Y': 2}),
                           'Diagnosis': annot["Diagnosis"].map({'Benign': 1, 'Malignant': 2})})
    categorical_cols = ['Diagnosis']
    batch_col = 'Batch'

    dat1 = dat_scaled.transpose()
    data_combat = neuroCombat(dat=dat1,
                              covars=covars,
                              batch_col=batch_col,
                              categorical_cols=categorical_cols)["data"]
    dat_scaled = pd.DataFrame(data_combat.transpose(), columns=dat.columns, index=dat.index)

    X = dat_scaled.copy()
    pca = PCA(n_components=4)
    pca.fit(X)
    x_pca = pca.transform(X)

    colors = ["#5f3d97", "#e46225"]

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

    fig, ax = plt.subplots()
    fig = sns.jointplot(data=x_pca, x=x_pca[:, 0], y=x_pca[:, 1], hue=annot["Contrast"].tolist(), s=50, palette=colors,
                        edgecolor='black', linewidth=2)
    fig.set_axis_labels(xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f} %)",
                        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f} %)")
    plt.legend(title="Contrast", markerscale=1)._legend_box.align = "left"
    texts = fig.ax_joint.legend_.texts
    for t, label in zip(texts, ["NCE", "CE"]):
        t.set_text(label)
    plt.suptitle("Dataset 2\nAfter ComBat", x=0, y=.95, horizontalalignment='left', verticalalignment='top')
    plt.savefig('VA_SF_After.png', dpi=300, bbox_inches='tight')

    # fig, ax = plt.subplots()
    # fig = sns.jointplot(data=x_pca, x=x_pca[:,0], y=x_pca[:,2], hue=annot["Contrast"].tolist(), s=50, palette=colors,
    #                     edgecolor='black', linewidth=1)
    # fig.set_axis_labels(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f} %)", ylabel=f"PC3 ({pca.explained_variance_ratio_[2]*100:.2f} %)")
    # plt.legend(title="Contrast", markerscale=2)._legend_box.align = "left"
    # texts = fig.ax_joint.legend_.texts
    # for t, label in zip(texts, ["NCE", "CE"]):
    #     t.set_text(label)
    # plt.savefig('PC1_PC3_After.png', dpi=300, bbox_inches='tight')