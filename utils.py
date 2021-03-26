
import optuna


def run_study(obj, n_trials, study_name):

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

    study.optimize(obj, n_trials=n_trials)
    return study
