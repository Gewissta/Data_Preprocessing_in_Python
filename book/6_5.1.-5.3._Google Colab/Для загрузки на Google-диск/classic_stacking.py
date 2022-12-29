import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator

class ClassicStacking(BaseEstimator):
    def __init__(self, 
                 estimators, 
                 final_estimator, 
                 cv=5, 
                 stack_method='predict_proba', 
                 passthrough=True):
        """
        Параметры
        ----------
        estimators: list
            Список базовых алгоритмов.
        final_estimator: экземпляр класса sklearn, строящего модель
            Мета-алгоритм.
        cv: int, по умолчанию 5
            Стратегия перекрестной проверки.
        stack_method: str, по умолчанию 'predict_proba'
            Метод получения прогнозов (predict, predict_proba).
        passthrough: bool, по умолчанию True
            Использование исходных признаков в ходе обучения.
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.passthrough = passthrough
        self.predicts_base = None
        self.predicts_base_test = None

    def fit(self, X_train, y_train, use_err=False, err=0.001, random_state=123):
        """
        X_train: массив формы (n_samples, n_features)
            Обучающий массив признаков.
        y_train: массив формы (n_samples,)
            Обучающий массив меток.
        use_err: bool, по умолчанию False
            Флаг использования случайного шума (True, False), 
            только для predict_proba
        err: float, по умолчанию 0.001
            Величина случайного шума
        random_state: int, по умолчанию 123
            Стартовое значение генератора псевдослучайных чисел.
        """
        np.random.seed(random_state)
        self.predicts_base = [None] * len(self.estimators)
        n = X_train.shape[0]
        m = len(set(y_train))
        for i, clf in enumerate(self.estimators):
            # получаем ответы (вероятности) для проверочных 
            # блоков перекрестной проверки
            pred = cross_val_predict(
                clf, X_train, y_train, cv=self.cv, n_jobs=-1, 
                method=self.stack_method)
            if use_err and self.stack_method == 'predict_proba':
                pred += err * np.random.randn(n, m)
            self.predicts_base[i] = pred
            # обучаем базовые алгоритмы на всей обучающей выборке
            clf.fit(X_train, y_train)
        return self

    def predict(self, X_train, y_train, X_test):
        """
        X_train: массив формы (n_samples, n_features)
            Обучающий массив признаков.
        y_train: массив формы (n_samples,)
            Обучающий массив меток.
        X_test: массив формы (n_samples, n_features)
            Тестовый массив признаков.
        """
        self.predicts_base_test = [None] * len(self.estimators)
        # получаем ответы (вероятности) базовых алгоритмов, обученных
        # на всей обучающей выборке, для тестовой выборки
        for i, clf in enumerate(self.estimators):
            if self.stack_method == 'predict_proba':
                self.predicts_base_test[i] = clf.predict_proba(X_test)
            elif self.stack_method == 'predict':
                self.predicts_base_test[i] = clf.predict(X_test)
        # если задано использование исходных признаков
        if self.passthrough:
            # конкатенируем по оси столбцов всю обучающую выборку с ответами
            # базовых алгоритмов для проверочных блоков
            train_stack = np.column_stack((X_train, 
                                           np.column_stack(self.predicts_base)))

            # конкатенируем по оси столбцов тестовую выборку с ответами базовых 
            # алгоритмов для тестовой выборки, обученными на всей обучающей выборке
            test_stack = np.column_stack((X_test, 
                                          np.column_stack(self.predicts_base_test)))
        else:
            train_stack = np.column_stack(self.predicts_base)
            test_stack = np.column_stack(self.predicts_base_test)

        # обучаем метамодель
        final_model = self.final_estimator
        final_model.fit(train_stack, y_train)

        # вычисляем вероятности классов с помощью метамодели
        final_proba = final_model.predict_proba(test_stack)
        return final_proba