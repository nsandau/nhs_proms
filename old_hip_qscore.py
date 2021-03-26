# %% [markdown]

# todo:

# implementere tabnet!
# skal lave wrapper som finder cat indexes - eg. missing indicators etc
# nnet:
# Embeddings til sidst? for at se om det forbedrer de fundne modeller. eventuelt også ved at fjerne de features som er fundet ved GA.
# andre lr schedulers??

# Error analysis
# cv_preds = cross_val_predict(bcv_elnet.best_estimator_, X=X_train_ros, y=y_train_ros, cv=5, n_jobs=10)


# %%
# cv_preds_df = pd.DataFrame(y_train_ros.copy())

# cv_preds_df['hip_replacement_post_op_q_score_bin'] = cv_preds
# cv_preds_df['true'] = y_train_ros

# true_preds = X_train_ros.loc[[indx for indx in X_train_ros.index if indx not in idx]]
# false_preds = X_train_ros.loc[cv_preds_df[cv_preds_df['hip_replacement_post_op_q_score_bin'] != cv_preds_df['true']].index]


# %%
import category_encoders as ce
import missingno as msno
from nnet_pytorch import TabularModel, TabularModelWrapper
from pyearth import Earth
# jeg har edited check_X_y call til at tillade nan
from genetic_selection import GeneticSelectionCV
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
    OrdinalEncoder,
)
from imblearn.over_sampling import RandomOverSampler
from skopt import BayesSearchCV, load
from skopt.callbacks import DeltaYStopper, CheckpointSaver
from skopt.plots import plot_objective
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump, load
%load_ext autoreload
%autoreload 2


# %%
pd.set_option('max_columns', None)
np.random.seed(123)
prototype = True
n_jobs = 8  # for bcv n_jobs = bcv_npoints * cv_folds
scoring = "roc_auc"
error_score = "raise"
n_folds = 2

# Bayes
bcv_npoints = 5
iter_bayes = 2
bayes_n_best = 5

# Genetic algo
n_popu = 10
n_gener = 10
n_gen_no_chg = 3


# %%
df = pd.read_csv("data_preprocessed/qscore_bin.csv")


# %%
# msno.heatmap(df)
# msno.matrix(df)


# %%
# same prop of outcomes for each year
for year in df["year"].unique():
    counts = df[df["year"] == year].value_counts("hip_replacement_post_op_q_score_bin")
    print("Year:", year, "1:", counts[1], "0:",
          counts[0], "prop:", round(counts[0]/counts[1], 2))


# small difference in outcome proportions
for age in df["age_band"].unique()[1:]:
    counts = df[df["age_band"] == age].value_counts(
        "hip_replacement_post_op_q_score_bin")
    print("age:", age, "1:", counts[1], "0:", counts[0],
          "prop:", round(counts[0]/counts[1], 2))

# %%
# rows with missing have same prop of outcomes
null_data = df[df.isnull().any(axis=1)].value_counts(
    'hip_replacement_post_op_q_score_bin')
non_null_data = df[~df.isnull().any(axis=1)].value_counts(
    'hip_replacement_post_op_q_score_bin')
print('prop missing:', round(null_data[0]/null_data[1], 2),
      'prop_non_miss:', round(non_null_data[0]/null_data[1], 2))

# %%
# rescale ordinal features starting from 1 to start from zero

#ord_scales = df.min() == 1
#ord_scales = ord_scales.to_dict()
#ord_cols = [key for key, val in ord_scales.items() if val == True]
#df[ord_cols] -= 1

# Scaler eq5_index med 10 så jeg kan identificere det som cont

#df["pre_op_q_eq5d_index"] += 10

# %%
# recode age_band to ordinal # sklearn implementation - supporter ikke nan før 0.24
age_ordering = [["20 to 29", "30 to 39", "40 to 49", "50 to 59",
                 "60 to 69", "70 to 79", "80 to 89", "90 to 120"]]
oe = OrdinalEncoder(categories=age_ordering)

# ce implementering
ce_age_ordering = [
    {'col': 'age_band', 'mapping': {"20 to 29": 0, "30 to 39": 1, "40 to 49": 2,  "50 to 59": 3, "60 to 69": 4,
                                    "70 to 79": 5, "80 to 89": 6, "90 to 120": 7}}
]

ce_oe = ce.OrdinalEncoder(mapping=ce_age_ordering, cols=["age_band"],
                          handle_unknown='return_nan', handle_missing='return_nan', drop_invariant=False, return_df=True)

df = ce_oe.fit_transform(df)

# %%
train = df[df["year"] != "2018/19"].drop(["year"], axis=1)
test = df[df["year"] == "2018/19"].drop(["year"], axis=1)
# %%

X_train = train.drop(["hip_replacement_post_op_q_score_bin"], axis=1)
y_train = train["hip_replacement_post_op_q_score_bin"]

X_test = test.drop(["hip_replacement_post_op_q_score_bin"], axis=1)
y_test = test["hip_replacement_post_op_q_score_bin"]

kfolds = StratifiedKFold(n_splits=n_folds)

# %%
# random oversampling of training set
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X_train, y_train)


# %%
# create prototype sample and shuffle the dataset
if prototype:
    index_proto = np.random.choice(len(X_ros), 80000, replace=False)
    X_train_ros = X_ros.iloc[index_proto]
    y_train_ros = y_ros.iloc[index_proto]
else:
    shuffle_idx = np.random.choice(len(X_ros), len(X_ros), replace=False)
    X_train_ros = X_ros.iloc[shuffle_idx]
    y_train_ros = y_ros.iloc[shuffle_idx]


# %%
cont_vars = ["hip_replacement_pre_op_q_score", "pre_op_q_eq_vas", "pre_op_q_eq5d_index"]
cat_vars = [
    "age_band",
    "gender",
    "pre_op_q_symptom_period",
    "pre_op_q_previous_surgery",
    "pre_op_q_living_arrangements",
    "pre_op_q_disability",
    "heart_disease",
    "high_bp",
    "stroke",
    "circulation",
    "lung_disease",
    "diabetes",
    "kidney_disease",
    "nervous_system",
    "liver_disease",
    "cancer",
    "depression",
    "arthritis",
    "pre_op_q_mobility",
    "pre_op_q_self_care",
    "pre_op_q_activity",
    "pre_op_q_discomfort",
    "pre_op_q_anxiety",
    "hip_replacement_pre_op_q_pain",
    "hip_replacement_pre_op_q_sudden_pain",
    "hip_replacement_pre_op_q_night_pain",
    "hip_replacement_pre_op_q_washing",
    "hip_replacement_pre_op_q_transport",
    "hip_replacement_pre_op_q_dressing",
    "hip_replacement_pre_op_q_shopping",
    "hip_replacement_pre_op_q_walking",
    "hip_replacement_pre_op_q_limping",
    "hip_replacement_pre_op_q_stairs",
    "hip_replacement_pre_op_q_standing",
    "hip_replacement_pre_op_q_work",
]

# %%


def fit_bcv(estimator,
            pgrid,
            X=X_train_ros,
            y=y_train_ros,
            chkpt_path=None,
            fit=True,
            nnet=False):

    if nnet:
        n_cores = 1
    else:
        n_cores = n_jobs

    bcv = BayesSearchCV(
        estimator=estimator,
        search_spaces=pgrid,
        scoring=scoring,
        cv=kfolds,
        n_iter=iter_bayes,
        n_points=bcv_npoints,
        error_score=error_score,
        optimizer_kwargs={"initial_point_generator": "lhs"},
        verbose=2,
        n_jobs=n_cores)

    # callbacks
    callbacks = []
    callbacks.append(DeltaYStopper(delta=0.01, n_best=bayes_n_best))
    if chkpt_path != None:
        callbacks.append(CheckpointSaver(checkpoint_path=chkpt_path))

    # fit
    if fit:
        bcv.fit(X, y, callback=callbacks)
    return bcv


def fit_ga(estimator,
           X=X_train_ros,
           y=y_train_ros,
           fit=True,
           nnet=False,
           n_jobs=n_jobs):

    if nnet:
        n_cores = 1
    else:
        n_cores = n_jobs

    sel = GeneticSelectionCV(estimator,
                             cv=kfolds,
                             verbose=1,
                             scoring=scoring,
                             n_population=n_popu,
                             n_generations=n_gener,
                             n_gen_no_change=n_gen_no_chg,
                             caching=True,
                             n_jobs=n_cores)

    if fit:
        sel.fit(X, y)
    return sel


# %%
# TRANSFORMERS

def select_cont(x): return list(*np.where(pd.DataFrame(x).nunique() > 10))
def select_cat(x): return list(*np.where(pd.DataFrame(x).nunique() < 10))
def select_all(x): return list(np.arange(x.shape[1]))


# %%
cat_trans = Pipeline([
    ("imp_cat", SimpleImputer(strategy="most_frequent", add_indicator=False)),
    ("oh", OneHotEncoder(sparse=False))
])

cont_trans = Pipeline([
    ("imp_cont", SimpleImputer(strategy="mean", add_indicator=False)),
    ("ptrans", PowerTransformer(standardize=False)),
    ("std_scale", StandardScaler())]
)

miss_trans = Pipeline([
    ("miss", MissingIndicator(sparse=False)),
    ("oh_miss", OneHotEncoder(sparse=False))
])

prep = ColumnTransformer([
    ("cont", cont_trans, select_cont),
    ("miss", miss_trans, select_all),
    ("cat", cat_trans, select_cat)
], remainder="passthrough"
)

# %% [markdown]
#  NNET

# %%

cont_trans_nn = Pipeline([
    ("imp", SimpleImputer(strategy="mean", add_indicator=False)),
    ("std_scale", StandardScaler())]
)

# Behøver måske ikke missing indicators da modellen allerede overfitter?? og det er jo egentlig cat værdier?

prep_nn = ColumnTransformer(
    [
        ("cont", cont_trans_nn, select_cont),
        ("miss", miss_trans, select_all),
        ("imp_cat", SimpleImputer(strategy="most_frequent", add_indicator=False), select_cat),
    ],
    remainder="passthrough"
)
# %%

pipe_nnet = Pipeline([
    ("prep", prep_nn),
    ("nnet", TabularModelWrapper(
        cont_cols=cont_vars,
        estimator_class=TabularModel,
        epochs=1,
        lr=0.01,
        bsz=2048,
        l2reg=0.1,
        device='cuda'))
])

pgrid_nnet = {
    "nnet__bsz": Categorical([1024, 1664]),  # gier det mening?
    "nnet__n_embeddings": Integer(1, 100),
    "nnet__l2reg": Real(0.00001, 1, prior="log-uniform"),
    "nnet__epochs": Integer(1, 10),
    "nnet__n_hidden": Integer(1, 10),
    "nnet__units_dense_in": Integer(128, 2048),
    "nnet__units_hidden": Integer(128, 2048),
    "nnet__dense_p": Real(0, 1),
    "nnet__embed_p": Real(0, 1),
    "nnet__embed_bn": Categorical([True, False]),
    "nnet__dense_bn": Categorical([True, False]),
    "nnet__lr": Real(0.00001, 0.5, prior='log-uniform')
}


# %%
bcv_nnet = fit_bcv(pipe_nnet, pgrid_nnet, nnet=True)


# %%
ga_nnet = fit_ga(pipe_nnet, nnet=True)
# %%

# EXTRACT EMBEDDINGS ## skal laves om når jeg bruger missing indicator!
emb_layers = bcv_nnet.best_estimator_.steps[1][1].model.embeds

emb_mats = [emb_layers.state_dict()[name].cpu().numpy()
            for name in emb_layers.state_dict()]

emb_vals = dict()
for idx, cat in enumerate(cat_vars):
    orig_vals = X_train_ros[cat].values
    emb_vals[cat] = np.concatenate(
        [emb_mats[idx][int(val)].reshape(1, -1) for val in orig_vals])

embedding_df = pd.DataFrame(np.concatenate(
    [array for array in emb_vals.values()], axis=1))

X_embed = pd.concat(
    [X_train_ros[cont_vars].reset_index(drop=True), embedding_df], axis=1)

# %%
# ELASTIC NET ####
# Noter:
# bedste AUC = 0.76. # 0.84 med SimpleImputer + indicators


# %%

pipe_elnet = Pipeline([
    ("prep", prep),
    ("elnet", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5))
])

pgrid_elnet = {
    "prep__cont__ptrans": Categorical([PowerTransformer(standardize=False), None]),
    "elnet__l1_ratio": Real(0, 1),
    "elnet__C": Real(0.0000001, 1, prior="log-uniform")}


# %%
bcv_elnet = fit_bcv(pipe_elnet, pgrid_elnet)


# %%
# dump(bcv_elnet.cv_results_, "./bcv/bcv_elnet")
# bcv_elnet = load("./bcv/bcv_elnet")


# %%
ga_elnet = fit_ga(bcv_elnet.best_estimator_)

#  %%
# %%
# Finetuning

# %% [markdown]
# LinearSVM

# %% SVM initial tuning
pipe_svm = Pipeline([
    ("prep", prep),
    ("svm", LinearSVC())
])


pgrid_svm = {
    "prep__cont__ptrans": Categorical([PowerTransformer(standardize=False), None]),
    "svm__C": Real(0.00001, 1, prior="log-uniform")}


# %%
bcv_svm = fit_bcv(pipe_svm, pgrid_svm)

# %%
ga_svm = fit_ga(bcv_svm.best_estimator_)

# %%
# MARS
pipe_mars = Pipeline([
    ("prep", prep),
    ("mars", Earth())
]
)

pgrid_mars = {
    "prep__cont__ptrans": Categorical([PowerTransformer(standardize=False), None]),
    "mars__max_degree": Integer(1, 3)}

# %%
bcv_mars = fit_bcv(pipe_mars, pgrid_mars)


# %%
# NB Gaussion

pipe_nb = Pipeline([
    ("prep", prep),
    ("mars", GaussianNB())
]
)

cross_val_score(pipe_nb, X_train_ros, y_train_ros,
                n_jobs=n_jobs, cv=kfolds, scoring=scoring)


# %% [markdown]
# ###  RANDOM FOREST

# %%
# RF INITIAL TUNING

prep_rf = ColumnTransformer([
    ("miss", miss_trans, select_all)
])

pipe_rf = Pipeline([("missing", prep_rf), ("clf", RandomForestClassifier())])

pgrid_rf = {"clf__n_estimators": Integer(1, 4000),
            "clf__min_samples_leaf": Integer(1, 500)}

bcv_rf = fit_bcv(pipe_rf, pgrid_rf)


# %%
#print(*pipe_rf.get_params(), sep = "\n")

# %%
# Feature selection Random Forest
ga_rf = fit_ga(bcv_rf.best_estimator_)

# %%
# Fine-tuning

# %% [markdown]
# ## XGBOOST

# %%
pgrid_xgb = {
    "n_estimators": Integer(1, 4000),
    "min_samples_leaf": Integer(1, 500)}

bcv_xgb = fit_bcv(XGBClassifier(), pgrid_xgb)


# %%
ga_xgb = fit_ga(bcv_xgb.best_estimator_)

# %% [markdown]
# ## LightGBM

# %%


pgrid_lgbm = {"n_estimators": Integer(1, 100),
              "min_child_samples": Integer(20, 200)}

bcv_lgbm = fit_bcv(LGBMClassifier(), pgrid_lgbm)


# %%
ga_lgbm = fit_ga(estimator=bcv_lgbm.best_estimator_)


# %%

selector = GeneticSelectionCV(LGBMClassifier(),
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=5,
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=40,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)
selector = selector.fit(X_train_ros, y_train_ros)

# %% [markdown]
# ## ENSAMBLES

models = [bcv_elnet, bcv_lgbm]

base_learners = [(str(i), model.best_estimator_) for i, model in enumerate(models)]

# doesnt need hp tuning
models.append(pipe_nb)

# %%

# skal finde ud af hvordan jeg kan køre bcv på weights? Skal inputtes som list. wrapper?
# SVM har ikke predict_proba - drop?, nnet laver en fejl fordi den bliver passet en numpy array som y??
vclf = VotingClassifier(base_learners, voting="soft", weights=[1, 2])

cross_val_score(vclf, X_train_ros, y_train_ros, scoring=scoring,
                cv=kfolds, n_jobs=1, error_score="raise")


# %% [markdown]
# ## STACKING
#
# Kan lave 'passthrough' True/False. Kræver dog også pipeline med preprocessing endnu engang?

# %%
base_learners = [(k, v.best_estimator_) for k, v in cv_results.items()]
models_stack = list()
cv_res_stack = dict()

# %% [markdown]
# ### Final Estimator: Log Reg

# %%
pgrid_logr_st = {"final_estimator__l1_ratio": Real(0, 1)}

models_stack.append(("logr_st", LogisticRegression(), pgrid_logr_st))

# %% [markdown]
# ### Run model stacking
#
# Den fejler med  array must not contain infs or NaNs

# %%
for model, f_clf, pgrid in models_stack:

    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=f_clf,
        cv=5,  # used for model training not eval
        n_jobs=1,
    )

    stack_cv = BayesSearchCV(
        estimator=stack,
        search_spaces=pgrid,
        scoring=scoring,
        cv=kfolds,
        n_iter=iter_bayes,
        n_jobs=n_jobs,  # maximum of n_points * cv per iteration
        n_points=6,
        error_score=error_score,
        optimizer_kwargs={"initial_point_generator": "lhs"},
    )

    cv_res_stack[model] = stack_cv.fit(
        X_proto, y_proto, callback=DeltaYStopper(delta=0.01, n_best=bayes_n_best)
    )


# %%
for model, result in cv_res_stack.items():
    print("Model: ", model, " Score: ", result.best_score_)
    print("Model: ", model, " Params: ", result.best_params_)


# %% [markdown]
# ## OUTLIER / ANOMALY DETECTION
# %% [markdown]
# Prepare dataset without upsampling

# %%
index_anom = np.random.choice(X_train.shape[0] + 1, 50000, replace=False)

X_anom = X_train.iloc[index_iso]
y_anom = y_train.iloc[index_iso]

y_anom[y_anom == 0] = -1

# %% [markdown]
# ### ISOLATION FOREST

# %%
X_train_anom = train[train['hip_replacement_post_op_q_score_bin']
                     != 0].drop(['hip_replacement_post_op_q_score_bin'], axis=1)
X_test_anom = train[train['hip_replacement_post_op_q_score_bin']
                    == 0].drop(['hip_replacement_post_op_q_score_bin'], axis=1)

# %%
anom_irf = IsolationForest(n_estimators=1000)
anom_irf.fit(X_train_anom)
# %%
preds = anom_irf.predict(X_test_anom)
pd.DataFrame(preds).value_counts()
# %%
anom_irf.pre
