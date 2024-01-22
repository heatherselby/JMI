import pandas as pd
import pingouin as pg
import os

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    if (a_set & b_set):
        c = list(a_set & b_set)
        return c
    else:
        print("No common elements")

def icc(files, icc_value=0.8):
    num=0
    df = []
    for file in files:
        num += 1
        file = file.filter(regex=r'^original.*', axis=1)
        file.insert(0, "Radiologist", num, True)
        df.append(file)
    df = pd.concat(df)
    df.index.name="Samples"
    df.reset_index(inplace=True)
    icc=[]
    for column in df.columns[2::]:
        icc.append(
            {
                'Features': column,
                'ICC': pg.intraclass_corr(data=df, targets="Samples", raters="Radiologist", ratings=column)["ICC"].iloc[5],
                'CI': pg.intraclass_corr(data=df, targets="Samples", raters="Radiologist", ratings=column)["CI95%"].iloc[5],
                'Lower CI': pg.intraclass_corr(data=df, targets="Samples", raters="Radiologist", ratings=column)["CI95%"].iloc[
                        5][0],
                'Upper CI': pg.intraclass_corr(data=df, targets="Samples", raters="Radiologist", ratings=column)["CI95%"].iloc[
                        5][1]
            }
        )
    icc_df = pd.DataFrame(icc)
    return icc_df[icc_df["ICC"] >= icc_value]
#
# def combat(annot1, dat):
#     annot1["Batch"] = annot1["Contrast"]
#     covars = pd.DataFrame({'Batch': annot1["Batch"].map({'N': 1, 'Y': 2}),
#                            'Diagnosis': annot1["Diagnosis"].map({'Benign': 1, 'Malignant': 2})})
#     categorical_cols = ['Diagnosis']
#     batch_col = 'Batch'
#
#     dat1 = dat.transpose()
#     data_combat = neuroCombat(dat=dat1,
#                               covars=covars,
#                               batch_col=batch_col,
#                               categorical_cols=categorical_cols)["data"]
#     dat = pd.DataFrame(data_combat.transpose(), columns=dat.columns, index=dat.index)
#     return dat
#
if __name__ == '__main__':
    path = os.getcwd()
    # meta = pd.read_csv("Meta.csv")
#     # meta = meta[meta["Manufacturer's Model Name"].str.contains("Discovery CT750 HD")]
#     # meta = meta[meta["Manufacturer's Model Name"].str.contains("Discovery CT750 HD|LightSpeed VCT")]
#     diagnosis = pd.read_excel("Diagnosis_Key_2.xlsx")
#
#     annot = meta.merge(diagnosis, on="Patient", how="inner")
#
#     FULL = pd.read_csv("SF_FM_Features.csv")
#     RB = pd.read_csv("SF_RB_Features.csv")
#     BB = pd.read_csv("SF_BB_Features.csv")
#     HPUNET = pd.read_csv("SF_HPUNET_Features.csv")
    segmentations = ["FM", "RB", "BB"]
    for seg in segmentations:
        raj = pd.read_csv("VA_PA_RAJ_" + seg + "_Features.csv", index_col=0)
        sachin = pd.read_csv("VA_PA_SACHIN_" + seg + "_Features.csv", index_col=0)
        df1 = [raj, sachin]
        icc_ms = icc(df1)
        icc_ms.sort_values(by=["ICC"], ascending=False, inplace=True)

        icc_ms_features = icc_ms["Features"].tolist()
        icc_ms.to_csv("VA_PA_" + seg + "_ICC_MS.csv", index=False)

        eroded = pd.read_csv("VA_PA_ERODED_" + seg + "_Features.csv",index_col=0)
        dilated = pd.read_csv("VA_PA_DILATED_" + seg + "_Features.csv", index_col=0)
        df2 = [eroded, raj, dilated]
        icc_ed = icc(df2)
        icc_ed.sort_values(by=["ICC"], ascending=False, inplace=True)
        icc_ed_features = icc_ed["Features"].tolist()
        icc_ed.to_csv("VA_PA_" + seg + "_ICC_ED.csv", index=False)

        common = common_member(icc_ed_features, icc_ms_features)
        print(len(common), icc_ed.shape[0], icc_ms.shape[0])




#
#     dat = FULL.copy()
#     dat.rename(columns={'Unnamed: 0': 'Patient'}, inplace=True)
#     annot1 = annot.copy()
#     dat = dat[dat.Patient.isin(annot1.Patient)]
#     annot1 = annot1[annot1.Patient.isin(dat.Patient)]
#     annot1.sort_values(by="Patient", inplace=True)
#     dat.sort_values(by="Patient", inplace=True)
#
#     dat.set_index("Patient", inplace=True)
#     annot1.set_index("Patient", inplace=True)
#     dat1 = dat[icc_full["Features"].tolist()]
    # dat = dat.filter(regex="^original*", axis=1)

    # annot1["Diagnosis"] = annot1["Diagnosis"].str.strip()
    # annot1.Diagnosis.replace(to_replace=["Adenocarcinoma", "Small cell lung cancer", "Squamous cell carcinoma",
    #                                      "Non-small cell lung cancer"],
    #                          value="Malignant", inplace=True)
    # dat = combat(annot1, dat1)
    # print(dat.head())

#     X = dat.copy()
#     Y = np.where(annot1["Diagnosis"] == "Malignant", 1, 0)
#
#     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=101)
#     folds = [(train, test) for train, test in cv.split(X, Y)]
#     metrics = ['auc', 'fpr', 'tpr', 'thresholds', 'y', 'y_pred']
#     results = {
#         'train': {m: [] for m in metrics},
#         'val': {m: [] for m in metrics},
#     }
#     # total_df = []
#     for train, test in tqdm(folds, total=len(folds)):
#         X_train = X.iloc[train, :]
#         y_train = Y[train]
#         X_test = X.iloc[test, :]
#         y_test = Y[test]
#         pipe = Pipeline([('rf', RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=101))])
#         # pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(penalty="l2", class_weight="balanced", random_state=101))])
#         # pipe = Pipeline([('sc', StandardScaler()), ('svc', SVC(class_weight="balanced", probability=True, random_state=101))])
#         pipe.fit(X_train, y_train)
#         # df = pd.DataFrame({'features': X_train.columns, 'importance': pipe.named_steps['rf'].feature_importances_})
#         # df.sort_values(by="importance", ascending=False, inplace=True)
#         # total_df.append(df)
#         sets = [X_train, X_test]
#         label = [y_train, y_test]
#         for i, ds in enumerate(results.keys()):
#             y_pred = pipe.predict_proba(sets[i])[:, 1]
#             labels = label[i]
#             fpr, tpr, thresholds = roc_curve(labels, y_pred)
#             results[ds]['fpr'].append(fpr)
#             results[ds]['tpr'].append(tpr)
#             results[ds]['thresholds'].append(thresholds)
#             results[ds]['auc'].append(roc_auc_score(labels, y_pred))
#             results[ds]['y_pred'].append(y_pred)
#             results[ds]['y'].append(label[i])
#     # total_df = pd.concat(total_df, axis=1)
#     # total_df.to_csv("SF_Features_" + name_segs[k] + ".csv")
#     kind = 'val'
#     # y_preds = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:,1]
#     # y_preds = np.array([item for sublist in results[kind]['y_pred'] for item in sublist])
#     # y = np.array([item for sublist in results[kind]['y'] for item in sublist])
#     # auc_delong, var = compare_auc_delong_xu.delong_roc_variance(y, y_preds)
#     # alpha = .95
#     # auc_std = np.sqrt(var)
#     # lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
#     # ci = stats.norm.ppf(
#     #     lower_upper_q,
#     #     loc=auc_delong,
#     #     scale=auc_std)
#     # ci[ci > 1] = 1
#
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []
#     for i in range(100):
#         fpr = results[kind]['fpr'][i]
#         tpr = results[kind]['tpr'][i]
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     std_tpr = np.std(tprs, axis=0)
#     n_samples = len(results[kind]['tpr'])
#     tprs_lower = mean_tpr - 1.96 * std_tpr / np.sqrt(n_samples)
#     tprs_upper = mean_tpr + 1.96 * std_tpr / np.sqrt(n_samples)
#     # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     mean_auc = auc(mean_fpr, mean_tpr)
#     print(mean_auc)
#     # if k == 0:
#     #     preds = pd.DataFrame({'y': y, 'y_probs': y_preds})
#     # else:
#     #     pvalue = delong_roc_test(preds["y"].to_numpy(), preds["y_probs"].to_numpy(), y_preds)
#     #     pvalue = 10 ** pvalue.ravel()[0]
#     #     print('{:0.3e}'.format(pvalue))
# #     fig = go.Figure()
# #     fig.add_trace(
# #         go.Scatter(
# #             x=mean_fpr,
# #             y=tprs_upper,
# #             line=dict(color=c_line[k], width=1),
# #             hoverinfo="skip",
# #             showlegend=False,
# #             name='upper'), )
# #     fig.add_trace(
# #         go.Scatter(
# #             x=mean_fpr,
# #             y=tprs_lower,
# #             fill='tonexty',
# #             fillcolor=c_fill[k],
# #             line=dict(color=c_line[k], width=1),
# #             hoverinfo="skip",
# #             showlegend=False,
# #             name="lower"), )
# #     fig.add_trace(
# #         go.Scatter(
# #             x=mean_fpr,
# #             y=mean_tpr,
# #             line=dict(color=c_line_main[k], width=2),
# #             hoverinfo="skip",
# #             showlegend=True,
# #             name=name_segs[k] + f": {auc_delong:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"))
# # fig.add_shape(
# #     type='line',
# #     line=dict(dash='dash'),
# #     x0=0, x1=1, y0=0, y1=1
# # )
# # fig.update_layout(
# #     template='plotly_white',
# #     title_x=0.5,
# #     xaxis_title="1 - Specificity",
# #     yaxis_title="Sensitivity",
# #     width=800,
# #     height=800,
# #     legend=dict(
# #         yanchor="bottom",
# #         xanchor="right",
# #         x=0.95,
# #         y=0.01,
# #     ),
# #     font=dict(
# #         family="Times New Roman",
# #         size=25,
# #         color="Black"
# #     ),
# #     legend_title_text="Mean AUROC (95% CI)"
# # )
# # fig.update_yaxes(
# #     range=[0, 1],
# #     gridcolor=c_grid,
# #     scaleanchor="x",
# #     scaleratio=1,
# #     linecolor='black')
# # fig.update_xaxes(
# #     range=[0, 1],
# #     gridcolor=c_grid,
# #     constrain='domain',
# #     linecolor='black')
# # fig.show()
# # fig.write_image(os.path.join(path, "SF_RF.pdf"))
#
#
#     # bb1 = pd.read_csv("PA_BB_Features_1.csv", index_col=0)
#     # bb2 = pd.read_csv("PA_BB_Features_2.csv", index_col=0)
#     # icc_bb = icc(bb1, bb2)
#     # icc_bb.sort_values(by=["ICC"], ascending=False, inplace=True)
#     # icc_bb.to_csv("icc_bb.csv")










