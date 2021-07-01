
import optuna


def run_study(obj, n_trials, study_name, run=True):

    pruner = optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(multivariate=True)
    storage = optuna.storages.RDBStorage(
        url="mysql+pymysql://admin:Testuser1234@database-1.c17p2riuxscm.us-east-2.rds.amazonaws.com/optuna", heartbeat_interval=120, grace_period=360)
    study = optuna.create_study(storage=storage,
                                study_name=study_name,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner,
                                direction="maximize")
    if run:
        study.optimize(obj, n_trials=n_trials)
    return study


def plot_study(study):

    plots = [
        optuna.visualization.plot_contour(study),
        optuna.visualization.plot_param_importances(study),
        optuna.visualization.plot_slice(study),
        optuna.visualization.plot_edf(study),
        optuna.visualization.plot_parallel_coordinate(study), ]
    for plt in plots:
        plt.show()


def split_train_test(df, y_var):
    X = df.drop([y_var], axis=1)
    y = df[y_var]
    return X, y
