import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline
import itertools
from neuroCombat import neuroCombat


def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    params:
    - df : pd.DataFrame
    - thresh : correlation threshold, will remove one of pairs of features with
               a correlation greater than this value
    """

    corrMatrix = df.corr()
    corrMatrix.loc[:, :] = np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def combat(annot, dat):
    annot["Batch"] = annot["Contrast"]
    covars = pd.DataFrame({'Batch': annot["Batch"].map({'N': 1, 'Y': 2}),
                           'Diagnosis': annot["Diagnosis"].map({'Benign': 1, 'Malignant': 2})})
    categorical_cols = ['Diagnosis']
    batch_col = 'Batch'

    dat1 = dat.transpose()
    data_combat = neuroCombat(dat=dat1,
                              covars=covars,
                              batch_col=batch_col,
                              categorical_cols=categorical_cols)["data"]
    dat = pd.DataFrame(data_combat.transpose(), columns=dat.columns, index=dat.index)
    return dat

def bootstrap():
    scores = list()
    for _ in range(100):
        # bootstrap sample
        indices = randint(0, 1000, 1000)
        sample = dataset[indices]
        # calculate and store statistic
        statistic = mean(sample)
        scores.append(statistic)


if __name__ == '__main__':
    meta = pd.read_csv("Meta.csv")
    diagnosis = pd.read_excel("Diagnosis_Key_2.xlsx")
    segs = ["FULL", "RB", "BB", "nnUNet"]
    samples = pd.read_csv("VA_SF_nnUNet_Features.csv")
    samples.rename(columns={'Unnamed: 0': 'Patient'}, inplace=True)
    samples = samples["Patient"].tolist()

    fig = go.Figure()
    for seg in segs:
        dat = pd.read_csv("VA_SF_" + seg + "_Features.csv")
        dat.rename(columns={'Unnamed: 0': 'Patient'}, inplace=True)
        # if seg != "nnUNet":
        #     # excluded = pd.read_csv("VA_PA_" + seg + "_Raj.csv", index_col=0)
        #     # excluded = excluded.index
        #     icc_features = pd.read_csv("VA_PA_" + seg + "_ICC.csv")
        #     dat1 = dat[["Patient"] + icc_features["Features"].tolist()]
        #     # dat = dat[~dat['Patient'].isin(excluded)]
        dat = dat[dat["Patient"].isin(samples)]

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

        dat = combat(annot, dat)

        shape_features = dat.loc[:,dat.columns.str.contains("shape")]
        statistic_features = dat.loc[:, dat.columns.str.contains("firstorder")]
        texture_features = dat.loc[:, ~dat.columns.str.contains("shape|firstorder")]

        features = [shape_features, statistic_features, texture_features]
        deleted = []
        for feats in features:
            deleted.append(find_correlation(feats))
        deleted_features = list(itertools.chain(*deleted))

        dat = dat.drop(columns = deleted_features)

        deleted_features = find_correlation(dat)
        dat = dat.drop(columns=deleted_features)

        X = dat.copy()
        Y = np.where(annot["Diagnosis"] == "Malignant", 1, 0)
        print(annot["Diagnosis"].value_counts())

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=101)
        folds = [(train, test) for train, test in cv.split(X, Y)]

        metrics = ['auc', 'fpr', 'tpr', 'thresholds', 'y_true', 'y_pred']
        results = {
            'train': {m:[] for m in metrics},
            'val': {m:[] for m in metrics},
        }

        for train, test in tqdm(folds, total=len(folds)):
            x_train = X.iloc[train, :]
            y_train = Y[train]
            x_test = X.iloc[test, :]
            y_test = Y[test]
            pipe = Pipeline([('rf', RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=101))])
            pipe.fit(x_train, y_train)
            sets = [x_train, x_test]
            label = [y_train, y_test]
            for i, ds in enumerate(results.keys()):
                y_pred = pipe.predict_proba(sets[i])[:, 1]
                labels = label[i]
                fpr, tpr, thresholds = roc_curve(labels, y_pred)
                results[ds]['fpr'].append(fpr)
                results[ds]['tpr'].append(tpr)
                results[ds]['thresholds'].append(thresholds)
                results[ds]['auc'].append(roc_auc_score(labels, y_pred))
                results[ds]['y_pred'].append(y_pred)
                results[ds]['y_true'].append(label[i])
        kind = 'val'

        c_fill = 'rgba(52, 152, 219, 0.2)'
        c_line = 'rgba(52, 152, 219, 0.5)'
        c_line_main = 'rgba(41, 128, 185, 1.0)'
        c_grid = 'rgba(189, 195, 199, 0.5)'
        c_annot = 'rgba(149, 165, 166, 0.5)'
        c_highlight = 'rgba(192, 57, 43, 1.0)'
        fpr_mean = np.linspace(0, 1, 100)
        stats = list()
        interp_tprs = []
        for i in range(100):
            fpr = results[kind]['fpr'][i]
            tpr = results[kind]['tpr'][i]
            stats.append(tpr)
            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_std = np.std(interp_tprs, axis=0)
        n_samples = len(results[kind]['tpr'])
        tpr_lower = tpr_mean - 1.96 * tpr_std / np.sqrt(n_samples)
        tpr_upper = tpr_mean + 1.96 * tpr_std / np.sqrt(n_samples)
        tpr_mean[-1] = 1.0
        # tpr_std = 2 * np.std(interp_tprs, axis=0)
        # tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
        # tpr_lower = tpr_mean - tpr_std
        auc = np.mean(results[kind]['auc'])
        print("The AUC is" + str(auc))
        fig.add_trace(
            go.Scatter(
                x=fpr_mean,
                y=tpr_mean,
                line=dict(color=c_line_main, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=seg + ' (AUROC): ' + f'{auc:.3f}'))
        fig.add_trace(
            go.Scatter(
                x=fpr_mean,
                y=tpr_lower,
                fill='tonexty',
                fillcolor=c_fill,
                line=dict(color=c_line, width=1),
                hoverinfo="skip",
                showlegend=False,
                name='lower')
        )
        fig.add_trace(
            go.Scatter(
            x=fpr_mean,
            y=tpr_upper,
            fill='tonexty',
            fillcolor=c_fill,
            line=dict(color=c_line, width=1),
            hoverinfo="skip",
            showlegend=False,
            name='upper')
        )
        fig.add_shape(
            type='line',
            line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        xaxis_title="1 - Specificity",
        yaxis_title="Sensitivity",
        width=800,
        height=800,
        legend=dict(
            yanchor="bottom",
            xanchor="right",
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range=[0, 1],
        gridcolor=c_grid,
        scaleanchor="x",
        scaleratio=1,
        linecolor='black')
    fig.update_xaxes(
        range=[0, 1],
        gridcolor=c_grid,
        constrain='domain',
        linecolor='black')
    fig.write_image("VA_SF_ROC.pdf")