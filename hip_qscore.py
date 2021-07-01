
# TODO:
# tune en lgbm på dataset med missing values - har jeg gjort men forskel på CVscore og test set er massiv!
# den finder næsten ikke nogle pos cases ?

# bruge tuned lgbm til at impute vals for andre models
# tune log-reg og RF for at teste om impute gør det bedre

# bruge tunede models til at teste permuation feature imp

# xgb og lgbm params https://sites.google.com/view/lauraepp/parameters


# %%
from numpy.testing._private.nosetester import NoseTester
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import optuna
from utils import run_study, split_train_test


# %%
# CONFIG
kfolds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
# %%
df = pd.read_csv('data/data_preprocessed/qscore.csv')

# %%

# WITH MISSING DATA

train = df[df["year"] != "2018/19"].drop(["year"], axis=1)
X_train = train.drop(["hip_replacement_post_op_q_score_bin"], axis=1)
y_train = train["hip_replacement_post_op_q_score_bin"]

test = df[df["year"] == "2018/19"].drop(["year"], axis=1)
X_test = test.drop(["hip_replacement_post_op_q_score_bin"], axis=1)
y_test = test["hip_replacement_post_op_q_score_bin"]

# %%
# WITHOUT MISSING
train_nm = train.dropna()
X_train_nm = train_nm.drop(["hip_replacement_post_op_q_score_bin"], axis=1)
y_train_nm = train_nm["hip_replacement_post_op_q_score_bin"]

test_nm = test.dropna()
X_test_nm = test_nm.drop(["hip_replacement_post_op_q_score_bin"], axis=1)
y_test_nm = test_nm["hip_replacement_post_op_q_score_bin"]

# %%

# skal have kørt pipelines ind i obj_func
# den smider nogle fejl
# har ikke weightet, one_hottet eller noget andet ...


def obj_logr(trial):
    df = train.dropna()
    X, y = split_train_test(df, y_var="hip_replacement_post_op_q_score_bin")
    logreg = LogisticRegression(
        solver="saga",
        l1_ratio=trial.suggest_float("l1_ratio", 0, 1),
        C=trial.suggest_loguniform("C", 1e-6, 1))
    cv_score = cross_val_score(logreg, X, y,
                               scoring="roc_auc", cv=kfolds, n_jobs=-1)
    return np.mean(cv_score)


# %%
study_logr = run_study(obj_logr, n_trials=10, study_name="logr")

# %%

logr = logreg = LogisticRegression(
    solver="saga",
    l1_ratio=0.83,
    C=0.05)

logr.fit(X_train_nm, y_train_nm)

# %%
probas = logr.predict_proba(X_test_nm)
roc_auc_score(y_test_nm, probas[:, 1])
# %%

# SKAL ÆNDRE NAMES TIL AT VÆRE SAMME SOM ARG SÅ JEG KAN BRUGE DICT FRA STUDY


def obj_lgbm(trial):
    lgbm = LGBMClassifier(
        learning_rate=0.1,
        max_depth=trial.suggest_int('max_depth', 1, 15),  # usually 3-12 -> tune!
        num_iteration=trial.suggest_int("num_iter", 1, 2000),  # 1 - inf -> tune
        # tune typical 255 (can be > 4000) -> tune!
        num_leaves=trial.suggest_int("num_leaves", 1, 2000),
        min_data_in_leaf=trial.suggest_int("data_in_leaf", 1, 50),  # 1- 50 -> Tune!
        # prop of rows to sample # typical 0.7 (reducing may improve generalize better?)
        sub_row=trial.suggest_int("sub_row", 70, 100) / 100,
        # same as sub_row but for cols
        sub_feature=trial.suggest_int("sub_cols", 70, 100) / 100,
        use_missing=True,
        is_unbalanced=True
    )
    cv_score = cross_val_score(lgbm, X_train, y_train,
                               scoring="roc_auc", cv=kfolds, n_jobs=-1)
    return np.mean(cv_score)


# %%
lgbm = LGBMClassifier(
    learning_rate=0.1,
    max_depth=14,
    num_iteration=388,
    num_leaves=1559,
    min_data_in_leaf=33,
    sub_row=0.99,
    sub_feature=0.93,
    use_missing=True,
    is_unbalanced=True
)

#lgbm.fit(X_train, y_train)
repeat_kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
cv_score = cross_val_score(lgbm, X_train, y_train,
                           scoring="roc_auc", cv=repeat_kfold, n_jobs=-1)


# %%
probas = lgbm.predict_proba(X_test)
roc_auc_score(y_test, probas[:, 1]),

# %%
preds = lgbm.predict(X_test)
f1_score(y_test, preds)


# %%
