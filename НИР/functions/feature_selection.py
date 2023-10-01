import pandas as pd
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor
# from boruta import BorutaPy
# from BorutaShap import BorutaShap
from sklearn.feature_selection import RFE


class FeatureSelection:
    """Класс для проведения Feature selection.

    Главным методом является filter_X.
    Алгоритм, который будет использоваться для выбора фич, фиксируется при объявлении объекта класса путем варьирования параметром method и rfe

    Параметры
    ---------
    method : string (default='pearson')
            'pearson'
            'rf_lasso'
            'catboost'
        Метод для проведения первого этапа фильтрации.

    rfe : bool (default=False)
        Использовать ли RFE.
    """

    def __init__(self, method='pearson', rfe=False, **kwargs) -> None:
        self.rfe = rfe
        self.method = method

    # Функция для отбора признаков на основе корреляции Пирсона:
    def _pearson_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:

        # Инициализация селектора признаков:
        selector = SelectKBest(score_func=f_regression, k=n_features)

        # Применение селектора, оценка результата:
        # X_selected = selector.fit_transform(X, y)
        selector.fit(X, y)

        # Собираем обратно датасет в уже отфильтрованном виде:
        # filtered_df = pd.DataFrame(columns=selector.get_feature_names_out(), index = X.index, data=X_selected)
        filtered_df = X.iloc[:, selector.get_support()]

        return filtered_df

    def _rf_lasso_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Функция для фильтрации с помощью RF и Lasso с выбором только совпадающих признаков"""
        # Инициализируем и обучаем случайный лес:
        rf_selector = RandomForestRegressor(n_estimators=500, random_state=1)
        rf_selector.fit(X, y)

        # Вычисляем значения важности признаков:
        feature_importances = rf_selector.feature_importances_

        # Собираем вспомогательный датафрейм с исходными колонками и значениями важности:
        temp_df_rf = pd.DataFrame(
            columns=X.columns, data=[feature_importances])

        # Обучаем линейную регрессию с L1-регуляризацией:
        l1_selector = Lasso(alpha=0.2, random_state=42,
                            max_iter=10000).fit(X, y)

        # Собираем вспомогательный датафрейм с исходными колонками и значениями важности:
        temp_df_l1 = pd.DataFrame(columns=X.columns, data=[l1_selector.coef_])

        # Найдём признаки, которые оставили и RF, и Lasso:
        temp_df = pd.concat([temp_df_rf, temp_df_l1])

        # Оставляем из исходного датафрейма только важные колонки (признаки):
        filtered_df = X.loc[:, (temp_df != 0).all(axis=0)]

        # filtered_df = X.iloc[:n_features]

        return filtered_df

    def _catboost_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:
        """Функция для фильтрации с помощью Feature Importance из Catboost"""

        # Инициализируем и обучаем регрессор:
        selector = CatBoostRegressor()
        selector.fit(X=X, y=y, verbose=False)

        # Собираем вспомогательный датафрейм с исходными колонками и значениями важности:
        feature_importances = pd.Series(
            selector.get_feature_importance(), X.columns)

        # Отбираем 100 признаков с самым большим значением важности:
        selected_features = feature_importances.sort_values(ascending=False)[
                            :n_features]

        # Оставляем из исходного датафрейма только важные колонки (признаки):
        filtered_df = X.loc[:, selected_features.index]

        return filtered_df

    # def _boruta_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    #     """Функция для фильтрации с помощью Boruta"""
    #     # Инициализация случайного леса в качестве эстиматора:
    #     rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

    #     # Инициализация Boruta для отбора признаков:
    #     selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

    #     # Поиск всех релевантных признаков:
    #     selector.fit(X.values, y.values)

    #     # Фильтруем датасет, чтобы получить итоговый вариант:
    #     filtered_df = X.loc[:, selector.support_]

    #     return filtered_df

    # def _borutashap_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    #     """Функция для фильтрации с помощью Boruta"""
    #     # # Инициализация Boruta для отбора признаков:
    #     selector = BorutaShap(importance_measure='shap', classification=False)

    #     # Поиск всех релевантных признаков:
    #     selector.fit(X=X, y=y, n_trials=100, sample=False,
    #                  verbose=False, random_state=0)

    #     # Получаем подвыборку в качестве итогового варианта:
    #     filtered_df = selector.Subset()

    #     return filtered_df

    def _rfe_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features=100) -> pd.DataFrame:
        # Инициализация и обучение RFE в качестве селектора признаков:
        selector = RFE(RandomForestRegressor(n_estimators=500, random_state=1),
                       n_features_to_select=n_features,
                       verbose=0)
        features = selector.fit(X, y)

        # Фильтруем датасет, чтобы получить итоговый вариант:
        filtered_df = X.loc[:, features.support_]

        return filtered_df

    def filter_X(self, X, y, n_features: int = 100, rfe_n_features: int = 55):
        """Применение Feature Selection.

        Args:
            X: pd.DataFrame
            y: pd.Series
            n_features (int, optional): Количество фичей, которые остаются после первой итерации. Defaults to 100.
            rfe_n_features (int, optional): Количество фичей, которое должно остаться после rfe. Используется только в случае, если при инициализации rfe=True. Defaults to 55.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.rfe and rfe_n_features is None:
            rfe_n_features = n_features
            n_features *= 2
        # print(self.method)
        t = time.time()

        n_features = min([n_features, X.shape[1]])
        if self.method == 'pearson':
            new_X = self._pearson_feature_selection(
                X, y, n_features=n_features)
        elif self.method == 'catboost':
            new_X = self._catboost_feature_selection(
                X, y, n_features=n_features)
        elif self.method == 'rf_lasso':
            new_X = self._rf_lasso_feature_selection(
                X, y,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        print(
            f"[Feature Selection] {self.method} completed. Out shape = {new_X.shape}. Total time = {round(time.time() - t, 2)}s")
        if not self.rfe:
            return new_X

        t = time.time()
        print(f"[Feature Selection] RFE. In shape = {new_X.shape}")
        new_X = self._rfe_feature_selection(new_X, y, int(rfe_n_features))
        print(
            f"[Feature Selection] RFE Done. Out shape = {new_X.shape}. Total time = {round(time.time() - t, 2)}s")
        # new_X = self._rfe_feature_selection(new_X, y, rfe_n_features)
        return new_X


def apply_feature_selection(X, y, fs_kwargs_params):
    rfe = fs_kwargs_params['rfe']
    method = fs_kwargs_params['method']
    n_features = int(fs_kwargs_params['n_features'])
    rfe_n_features = fs_kwargs_params.get('rfe_n_features')
    try:
        rfe_n_features = int(rfe_n_features)
    except:
        rfe_n_features = None

    print(f"[Feature Selection] method={method}, rfe={rfe}.")

    fs = FeatureSelection(method, rfe)
    X = fs.filter_X(
        X, y, n_features=n_features, rfe_n_features=rfe_n_features)

    columns_after_fs = X.columns.tolist()
    return columns_after_fs
