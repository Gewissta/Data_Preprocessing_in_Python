import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import (BaseEstimator, 
                          TransformerMixin)
from collections import Counter


# создаем собственный класс, заменяющий отрицательные
# и нулевые значения на небольшие положительные
class Replacer(BaseEstimator, TransformerMixin):
    """
    Заменяет отрицательные и нулевые значения на
    небольшое положительное значение.
    
    Параметры
    ----------
    repl: float, по умолчанию 0.1
        Значение для замены.
    """
    def __init__(self, repl_value=0.1):
        self.repl_value = repl_value
    
    # fit здесь бездельничает
    def fit(self, X, y=None):
        return self
    
    # transform выполняет всю работу: применяет преобразование 
    # с помощью заданного значения параметра repl_value
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X[X <= 0] = self.repl_value
        else:
            X = np.where(X <= 0, self.repl_value, X)
        return X
    
    
# пишем функцию, проверяющую порядок переменных в наборах
# (порядок генерации признаков может повлиять 
# на результат CatBoost)
def check_vars_order(hist_data, new_data):
    if hist_data == new_data:
        print('Одни и те же списки переменных, ' + 
              'один и тот же порядок')
    elif sorted(hist_data) == sorted(new_data):
         print('Одни и те же списки переменных, ' + 
               'разный порядок')
    else:
        print('Совершенно разные списки переменных')
        
        
# пишем функцию конструирования календарных признаков
def create_calendar_vars(df):
    # выделяем из индекса ПОРЯДКОВЫЕ НОМЕРА КВАРТАЛОВ
    df['quarter'] = df.index.quarter
    # выделяем из индекса ПОРЯДКОВЫЕ НОМЕРА МЕСЯЦЕВ, 
    # где 1 - январь, 12 - декабрь
    df['month'] = df.index.month
    # пишем функцию для создания переменной - сезон
    def get_season(month):
        if (month > 11 or month < 3):
            return 0
        elif (month == 3 or month <= 5):
            return 1
        elif (month >= 6 and month < 9):
            return 2
        else:
            return 3
    # создаем переменную - сезон
    df['season'] = df['month'].apply(
        lambda x: get_season(x))
    
    return df


# функция для создания скользящих статистик
def moving_stats(series, alpha=1, seasonality=1, 
                 periods=1, min_periods=1, window=4, 
                 aggfunc='mean', fillna=None): 
    """
    Создает скользящие статистики.
    
    Параметры
    ----------
    alpha: int, по умолчанию 1
        Коэффициент авторегрессии.
    seasonality: int, по умолчанию 1
        Коэффициент сезонности.
    periods: int, по умолчанию 1
        Порядок лага, с которым вычисляем скользящие 
        статистики.    
    min_periods: int, по умолчанию 1
        Минимальное количество наблюдений в окне для
        вычисления скользящих статистик.
    window: int, по умолчанию 4
        Ширина окна. Не должна быть меньше
        горизонта прогнозирования.
    aggfunc: string, по умолчанию 'mean'
        Агрегирующая функция.  
    fillna, int, по умолчанию 0
        Стратегия импутации пропусков.
    """
    # задаем размер окна для определения 
    # диапазона коэффициентов 
    size = window if window != -1 else len(series) - 1
    # задаем диапазон значений коэффициентов в виде списка
    alpha_range = [alpha ** i for i in range(0, size)]
    # задаем минимально требуемое количество наблюдений 
    # в окне для вычисления скользящего среднего с 
    # поправкой на сезонность 
    min_required_len = max(
        min_periods - 1, 0) * seasonality + 1
    
    # функция для получения лагов в соответствии 
    # с заданной сезонностью 
    def get_required_lags(series):
        # возвращает вычисленные лаги в соответствии 
        # с заданной сезонностью 
        return pd.Series(series.values[::-1]
                         [:: seasonality])
        
    # функция для вычисления скользящей статистики
    def aggregate_window(series):
        tmp_series = get_required_lags(series)
        size = len(tmp_series)
        tmp = tmp_series * alpha_range[-size:]

        if aggfunc == 'mdad':
            return tmp.to_frame().agg(
                lambda x: np.nanmedian(
                    np.abs(x - np.nanmedian(x))))
        else:
            return tmp.agg(aggfunc)
    
    # собственно вычисление скользящих статистик
    features = series.shift(periods=periods).rolling(
        window=seasonality * window 
        if window != -1 else len(series) - 1, 
        min_periods=min_required_len).aggregate(
        aggregate_window)
    
    # импутируем пропуски
    if fillna is not None:
        features.fillna(fillna, inplace=True)
        
    return features


# создаем класс, выполняющий дамми-кодирование
class DFOneHotEncoder(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # выделяем список категориальных переменных
        cat_cols = X.select_dtypes(
            include='object').columns.tolist()
        # определяем дамми-переменные для 
        # категориальных переменных
        self.ohe = OneHotEncoder(sparse=False, 
                                 handle_unknown='ignore')
        self.ohe.fit(X[cat_cols])
        return self

    def transform(self, X):
        # выделяем списки количественных 
        # и категориальных переменных
        num_cols = X.select_dtypes(
            exclude='object').columns.tolist()
        cat_cols = X.select_dtypes(
            include='object').columns.tolist()
        # выполняем дамми-кодирование 
        # категориальных переменных
        Xt = self.ohe.transform(X[cat_cols])
        # извлекаем имена дамми-переменных
        cols = self.ohe.get_feature_names_out()
        # превращаем массив NumPy с дамми-переменными
        # в объект DataFrame
        X_dum = pd.DataFrame(Xt, index=X[cat_cols].index, 
                             columns=cols)
        # конкатерируем датафрейм, содержащий 
        # количественные переменные, с датафреймом, 
        # содержащим дамми-переменные
        X_res = pd.concat([X[num_cols], X_dum], axis=1)
        return X_res
    
    
# пишем собственный класс CountEncoder, заменяющий категории 
# относительными или абсолютными частотами
class CountEncoder(BaseEstimator, TransformerMixin):
    """   
    Классический CountEncoder. Может кодировать категории 
    переменных абсолютными или относительными частотами. 
    Умеет обрабатывать категории с одинаковыми частотами, 
    новые категории, пропуски.
    
    Параметры
    ----------
    min_count: float, значение по умолчанию 0.04
        Пороговое значение для частоты категории, меньше которого 
        возвращаем nan_value - специальное значение для пропусков 
        и редких категорий, для способа кодировки 'frequency' 
        задаем число с плавающей точкой, для способа 
        кодировки 'count' задаем целое число.
    encoding_method: str, значение по умолчанию 'frequency'
        Способ кодирования частотами - абсолютными ('count') 
        или относительными ('frequency').
    correct_equal_freq: bool, значение по умолчанию False
        Включает корректировку одинаковых частот для категорий.
    nan_value: int, значение по умолчанию -1
        Специальное значение для пропусков и редких категорий.
    copy: bool, значение по умолчанию True
        Возвращает копию.
    """
    
    def __init__(self, min_count=0.04, encoding_method='frequency', 
                 correct_equal_freq=False, nan_value=-1, copy=True):
        
        # если задано значение для encoding_method, 
        # не входящее в список, выдать ошибку
        if encoding_method not in ['count', 'frequency']:
            raise ValueError("encoding_method takes only " 
                             "values 'count' and 'frequency'")
            
        # если для параметра min_count задано значение, которое 
        # меньше/равно 0 или значение больше 1 и при этом для 
        # параметра encoding_method задано значение 'frequency',
        # выдать ошибку
        if ((min_count <= 0) | (min_count > 1)) & (
            encoding_method == 'frequency'):
            raise ValueError(f"min_count for 'frequency' should be " 
                             f"between 0 and 1, was {min_count}")
        
        # если для параметра encoding_method задано значение 'count'
        # и значение для параметра min_count не является
        # целочисленным, выдать ошибку
        if encoding_method == 'count' and not isinstance(min_count, int):
            raise ValueError("encoding_method 'count' requires " 
                             "integer values for min_count")

        # пороговое значение для частоты категории, меньше которого 
        # возвращаем nan_value - специальное значение для пропусков 
        # и редких категорий, для способа кодировки 'frequency' 
        # задаем число с плавающей точкой, для способа кодировки 
        # 'count' задаем целое число
        self.min_count = min_count
        # способ кодирования частотами - абсолютными ('count') 
        # или относительными ('frequency')
        self.encoding_method = encoding_method
        # корректировка одинаковых частот для категорий
        self.correct_equal_freq = correct_equal_freq
        # специальное значение для пропусков и редких категорий
        self.nan_value = nan_value
        # выполнение копирования
        self.copy = copy
                
    def __is_numpy(self, X):
        """
        Метод проверяет, является ли наш объект массивом NumPy.
        """
        return isinstance(X, np.ndarray)


    def fit(self, X, y=None):
        # создаем пустой словарь counts
        self.counts = {}
        
        # записываем результат метода __is_numpy
        is_np = self.__is_numpy(X)
        
        # записываем общее количество наблюдений
        n_obs = len(X)
        
        # если 1D-массив, то переводим в 2D
        if len(X.shape) == 1:
            if is_np:
                X = X.reshape(-1, 1)
            else:
                X = X.to_frame()
            
        # записываем количество столбцов
        ncols = X.shape[1]
    
        for i in range(ncols):
            # если выбрано значение 'frequency' для encoding_method 
            if self.encoding_method == 'frequency':
                # если объект - массив NumPy:
                if is_np:
                    # создаем временный словарь cnt, ключи - категории
                    # переменной, значения - абсолютные частоты категорий
                    cnt = dict(Counter(X[:, i]))
                    # абсолютные частоты заменяем на относительные
                    cnt = {key: value / n_obs for key, value in cnt.items()}
                # если объект - Dataframe pandas:
                else:
                    # создаем временный словарь cnt, 
                    # ключи - категории переменной,
                    # значения - относительные частоты категорий
                    cnt = (X.iloc[:, i].value_counts() / n_obs).to_dict()
                    
                # если значение min_count больше 0
                if self.min_count > 0:
                    # если относительная частота категории меньше min_count,
                    # возвращаем nan_value
                    cnt = dict((k, self.nan_value if v < self.min_count else v) 
                               for k, v in cnt.items())
                    
            # если выбрано значение 'count' для encoding_method        
            if self.encoding_method == 'count':
                # если объект - массив NumPy:
                if is_np:
                    # создаем временный словарь cnt, 
                    # ключи - категории переменной,
                    # значения - абсолютные частоты категорий
                    cnt = dict(Counter(X[:, i]))
                # если объект - Dataframe pandas:
                else:
                    # создаем временный словарь cnt, 
                    # ключи - категории переменной,
                    # значения - абсолютные частоты категорий
                    cnt = (X.iloc[:, i].value_counts()).to_dict()
                    
                # если значение min_count больше 0
                if self.min_count > 0:
                    # если абсолютная частота категории меньше min_count,
                    # возвращаем nan_value
                    cnt = dict((k, self.nan_value if v < self.min_count else v) 
                               for k, v in cnt.items())
            
            # обновляем словарь counts, ключом словаря counts будет 
            # индекс переменной, значением словаря counts будет 
            # словарь cnt, ключами будут категории переменной, 
            # значениями - частоты категорий переменной
            self.counts.update({i: cnt})
        
        # если для параметра correct_equal_freq задано значение True,
        # добавляем случайный шум к частотам категорий
        if self.correct_equal_freq:
            noise_param = 0.01
            np.random.seed(0)
            for v in self.counts.values():
                for key, value in v.items():
                    noise_value = value * noise_param
                    noise = np.random.uniform(-noise_value, noise_value)
                    v[key] = value + noise
 
        return self

    def transform(self, X):        
        # выполняем копирование массива
        if self.copy:
            X = X.copy()
            
        # записываем результат метода __is_numpy
        is_np = self.__is_numpy(X)
        
        # если 1D-массив, то переводим в 2D
        if len(X.shape) == 1:
            if is_np:
                X = X.reshape(-1, 1)
            else:
                X = X.to_frame()
                
        # записываем количество столбцов
        ncols = X.shape[1]

        for i in range(ncols):
            cnt = self.counts[i]
            
            # дополняем словарь неизвестными категориями, 
            # которых не было в методе fit
            
            # если объект - массив NumPy
            if is_np:
                unknown_categories = set(X[:, i]) - set(cnt.keys())
            
            # если объект - датафрейм pandas
            else:
                unknown_categories = set(X.iloc[:, i].values) - set(cnt.keys())
                
            for category in unknown_categories:
                cnt[category] = 1.0
            
            # если объект - массив NumPy:
            if is_np:
                # получаем из словаря по каждой переменной кортеж 
                # с категориями и список с частотами
                k, v = list(zip(*sorted(cnt.items())))
                
                # кортеж преобразовываем в массив
                v = np.array(v)
                
                # возвращаем индексы
                ix = np.searchsorted(k, X[:, i], side='left')
                # заменяем категории частотами с помощью индексов
                X[:, i] = v[ix]
            # если объект - Dataframe pandas:
            else:
                # заменяем категории частотами
                X.iloc[:, i].replace(cnt, inplace=True)
                
        return X
    
    def fit_transform(self, X):
        """
        Метод .fit_transform() для совместимости
        """
        self.fit(X)
        
        return self.transform(X)
    
    
# пишем класс SmoothingTargetEncoder, выполняющий кодирование 
# средним значением зависимой переменной, сглаженным
# через сигмоиду
class SmoothingTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Автор: Dmitry Larko
    https://www.kaggle.com/dmitrylarko
    
    Параметры
    ---------
    columns_names: list
        Cписок признаков.
    k: int, значение по умолчанию 2
        Половина минимально допустимого размера категории, 
        при которой мы полностью «доверяем» апостериорной 
        вероятности.
    f: float, значение по умолчанию 0.25
        Угол наклона сигмоиды (определяет соотношение между 
        апостериорной и априорной вероятностями).
    """
    
    def __init__(self, columns_names, k=2, f=0.25):
        
        # инициализируем публичные атрибуты
        
        # список признаков, которые будем кодировать
        self.columns_names = columns_names
        
        # половина минимально допустимого размера категории, 
        # при которой мы полностью «доверяем» 
        # апостериорной вероятности
        self.k = k
        
        # угол наклона сигмоиды (определяет соотношение между
        # апостериорной и априорной вероятностями)
        self.f = f
        
        # создаем пустой словарь, куда будем сохранять обычные 
        # средние значения зависимой переменной в каждой 
        # категории признака:
        # ключами в словаре будут названия признаков, а значениями 
        # - таблицы, в которых напротив каждой категории признака 
        # будет указано среднее значение зависимой переменной 
        # в данной категории признака
        self.simple_values = {}
        
        # создаем пустой словарь, куда будем сохранять сглаженные 
        # средние значения зависимой переменной в каждой 
        # категории признака:
        # ключами в словаре будут названия признаков, а значениями 
        # - таблицы, в которых напротив каждой категории признака
        # будет указано сглаженное среднее значение зависимой 
        # переменной в данной категории признака
        self.smoothing_values = {}
        
        # создадим переменную, в которой будем хранить глобальное среднее 
        # (среднее значение зависимой переменной по обучающему набору)
        self.dataset_mean = np.nan
    
    def smoothing_func(self, N):
        # метод задает весовой коэффициент - сигмоиду для сглаживания
        return 1 / (1 + np.exp(-(N-self.k) / self.f))
    
    # fit должен принимать в качестве аргументов только X и y
    def fit(self, X, y=None):
        
        # выполняем копирование массива во избежание предупреждения 
        # SettingWithCopyWarning "A value is trying to be set on a 
        # copy of a slice from a DataFrame (Происходит попытка изменить 
        # значение в копии среза данных датафрейма)"
        X_ = X.copy()
        
        # создадим переменную, в которой будем хранить глобальное среднее 
        # (среднее значение зависимой переменной по обучающему набору)
        self.dataset_mean = np.mean(y)
        
        # добавляем в новый массив признаков зависимую переменную 
        # и называем ее __target__, именно эту переменную __target__ 
        # будем использовать в дальнейшем для вычисления среднего значения 
        # зависимой переменной для каждой категории признака
        X_['__target__'] = y
          
        # в цикле для каждого признака, который участвует в кодировании 
        # (присутствует в списке self.columns_names)
        for c in [x for x in X_.columns if x in self.columns_names]:
            # формируем набор, состоящий из значений данного признака 
            # и значений зависимой переменной
            # группируем данные по категориям признака, считаем среднее 
            # значение зависимой переменной для каждой категории 
            # признака и обновляем индекс
            stats = (X_[[c, '__target__']]
                     .groupby(c)['__target__']
                     .agg(['mean', 'size'])
                     .reset_index())       
            
            # вычислим весовой коэфициент alpha, который "перемешает" 
            # среднее значение зависимой переменной в категории признака 
            # и глобальное среднее, вычисленное по обучающему набору
            stats['alpha'] = self.smoothing_func(stats['size'])
            
            # вычисляем сглаженное среднее значение зависимой переменной 
            # для категории признака согласно формуле 
            stats['__smooth_target__'] = (stats['alpha'] * stats['mean'] 
                                          + (1 - stats['alpha']) * 
                                          self.dataset_mean)
            
            # вычисляем обычное среднее значение зависимой переменной 
            # для категории признака 
            stats['__simple_target__'] = stats['mean']
            
            # формируем набор, состоящий из признака, сглаженных средних 
            # значений зависимой переменной для категорий признака и обычных 
            # средних значений зависимой переменной для категорий признака
            stats = (stats.drop([x for x in stats.columns 
                                 if x not in ['__smooth_target__', 
                                              '__simple_target__',  c]], 
                                axis=1).reset_index())
            
            # сохраним сглаженные средние значения зависимой переменной 
            # для каждой категории признака в словарь
            self.smoothing_values[c] = stats[[c, '__smooth_target__']]
            # сохраним обычные средние значения зависимой переменной
            # для каждой категории признака в словарь
            self.simple_values[c] = stats[[c, '__simple_target__']]
           
        return self
    
    # transform принимает в качестве аргумента только X
    def transform(self, X, smoothing=True):        
        # скопируем массив данных с признаками, значения которых 
        # будем кодировать, этот массив будем изменять при вызове 
        # метода .transform(), поэтому важно его скопировать, 
        # чтобы не изменить исходный массив признаков
        transformed_X = X[self.columns_names].copy()
        
        if smoothing:     
            # в цикле для каждого признака, который 
            # участвует в кодировании
            for c in transformed_X.columns:
                # формируем датафрейм, состоящий из значений данного признака, 
                # и выполняем слияние с датафреймом, содержащим сглаженные 
                # средние для каждой категории признака, нам нужны только 
                # сглаженные средние для каждой категории признака, поэтому 
                # в датафрейме оставляем только столбец '__smooth_target__'
                transformed_X[c] = (transformed_X[[c]]
                                   .merge(self.smoothing_values[c], 
                                          on=c, how='left')
                                   )['__smooth_target__']   
        
        else:   
            # в цикле для каждого признака, который 
            # участвует в кодировании
            for c in transformed_X.columns:
                # формируем датафрейм, состоящий из значений данного
                # признака, и выполняем слияние с датафреймом, содержащим 
                # обычные средние значения зависимой переменной 
                # для каждой категории признака
                # нам нужны только обычные средние значения зависимой 
                # переменной для каждой категории признака, поэтому в
                # датафрейме оставляем только столбец '__simple_target__'
                transformed_X[c] = (transformed_X[[c]]
                                   .merge(self.simple_values[c], 
                                          on=c, how='left')
                                   )['__simple_target__']
            
        # пропуски или новые категории признаков заменяем средним 
        # значением зависимой переменной по обучающему набору 
        transformed_X = transformed_X.fillna(self.dataset_mean)
        
        # возвращаем закодированный массив признаков
        return transformed_X