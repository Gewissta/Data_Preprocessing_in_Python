# импортируем необходимые библиотеки, классы и функции
import random
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import typer

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.loggers import WandbLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import Sign
from etna.models import CatBoostModelMultiSegment
from etna.pipeline import Pipeline
from etna.transforms import LagTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import StandardScalerTransform

# задаем переменную пути к файлу данных
FILE_PATH = Path(__file__)

# пишем функцию, задающую стартовое значение генератора 
# псевдослучайных чисел для воспроизводимости
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    
# пишем функцию инициализации логгера
def init_logger(config: dict, project: str = 'wandb-sweeps', 
                tags: Optional[list] = ['test', 'sweeps']):
    tslogger.loggers = []
    wblogger = WandbLogger(project=project, 
                           tags=tags, 
                           config=config)
    tslogger.add(wblogger)

    
# пишем функцию, загружающую данные
def dataloader(file_path: Path, freq: str = 'D') -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


# пишем целевую функцию Optuna
def objective(trial: optuna.Trial, 
              metric_name: str, 
              ts: TSDataset, 
              horizon: int, 
              lags: int, 
              seed: int):
    """
    Целевая функция Optuna.
    """

    # задаем стартовое значение генератора псевдослучайных
    # чисел для воспроизводимости
    set_seed(seed)

    # задаем модель и признаки
    pipeline = Pipeline(
        model=CatBoostModelMultiSegment(
            iterations=trial.suggest_int('iterations', 10, 100),
            depth=trial.suggest_int('depth', 1, 12),
        ),
        transforms=[
            StandardScalerTransform('target'),
            SegmentEncoderTransform(),
            LagTransform(
                in_column='target', 
                lags=list(range(horizon, 
                                horizon + 
                                trial.suggest_int('lags', 1, lags)))),
        ],
        horizon=horizon,
    )

    # инициализируем логгер WandB
    init_logger(pipeline.to_dict())

    # запускаем перекрестную проверку
    metrics, _, _ = pipeline.backtest(
        ts=ts, metrics=[MAE(), SMAPE(), Sign(), MSE()])
    return metrics[metric_name].mean()


# создаем typer.Typer приложение
app = typer.Typer()

# используем декоратор, он сообщает Typer, что 
# нижеприведенная функция является «командой»
@app.command()
# пишем функцию запуска оптимизации
def run_optuna(
    horizon: int = 14,
    metric_name: str = 'MAE',
    storage: str = 'sqlite:///optuna.db',
    study_name: Optional[str] = None,
    n_trials: int = 200,
    file_path: Path = 'Data/example_dataset.csv',
    direction: str = 'minimize',
    freq: str = 'D',
    lags: int = 24,
    seed: int = 11,
):
    """
    Запускает оптимизацию Optuna 
    для CatBoostModelMultiSegment.
    """
    # загружаем данные
    ts = dataloader(file_path, freq=freq)

    # создаем сессию оптимизации Optuna
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(multivariate=True, 
                                           group=True),
        load_if_exists=True,
        direction=direction,
    )

    # запускаем оптимизацию Optuna
    study.optimize(
        partial(objective, metric_name=metric_name, ts=ts, 
                horizon=horizon, lags=lags, seed=seed), 
        n_trials=n_trials
    )

# запускаем
if __name__ == '__main__':
    typer.run(run_optuna)
    
    
