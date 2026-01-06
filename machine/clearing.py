import pandas as pd
import numpy as np
from typing import Literal, Optional, Tuple, Union

class DatasetCleaner:
    def __init__(self, df: pd.DataFrame):
        self.history = [df.copy()]
        self.current_index = 0
        self.current_df = self.history[self.current_index]

    def _update_history(self, new_df: pd.DataFrame):
        self.history = self.history[:self.current_index + 1]
        self.history.append(new_df)
        self.current_index += 1
        self.current_df = self.history[self.current_index]

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.current_df = self.history[self.current_index]

    def redo(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.current_df = self.history[self.current_index]

    def list_versions(self) -> int:
        return len(self.history)

    def get_version(self, index: int) -> pd.DataFrame:
        if 0 <= index < len(self.history):
            return self.history[index].copy()
        raise IndexError("Version index out of range")

    def count_missing_and_zeros(self, column: Optional[str] = None) -> Tuple[int, int, int]:
        if column is None:
            num_missing = self.current_df.isna().sum().sum()
            num_zeros = (self.current_df == 0).sum().sum()
            total = self.current_df.size
        else:
            num_missing = self.current_df[column].isna().sum()
            num_zeros = (self.current_df[column] == 0).sum()
            total = len(self.current_df)
        return num_missing, num_zeros, total

    def remove_row_or_column(self, row: Optional[Union[int, list]] = None, column: Optional[Union[str, list]] = None) -> None:
        new_df = self.current_df.copy()
        if row is not None:
            new_df = new_df.drop(index=row)
        if column is not None:
            new_df = new_df.drop(columns=column)
        self._update_history(new_df)

    def fill_missing(self, column: str, method: Literal['mean', 'median', 'mode', 'interpolate'] = 'median') -> None:
        new_df = self.current_df.copy()
        if method == 'mean':
            new_df[column] = new_df[column].fillna(new_df[column].mean())
        elif method == 'median':
            new_df[column] = new_df[column].fillna(new_df[column].median())
        elif method == 'mode':
            mode_val = new_df[column].mode()
            if not mode_val.empty:
                new_df[column] = new_df[column].fillna(mode_val[0])
        elif method == 'interpolate':
            new_df[column] = new_df[column].interpolate()
        self._update_history(new_df)

    def find_duplicates(self) -> pd.DataFrame:
        return self.current_df[self.current_df.duplicated(keep=False)]

    def remove_duplicates(self, keep: Literal['first', 'last', False] = 'first') -> None:
        new_df = self.current_df.drop_duplicates(keep=keep)
        self._update_history(new_df)

    def normalize(self, column: str) -> None:
        new_df = self.current_df.copy()
        min_val = new_df[column].min()
        max_val = new_df[column].max()
        new_df[column] = (new_df[column] - min_val) / (max_val - min_val)
        self._update_history(new_df)

    def standardize(self, column: str) -> None:
        new_df = self.current_df.copy()
        mean_val = new_df[column].mean()
        std_val = new_df[column].std()
        new_df[column] = (new_df[column] - mean_val) / std_val
        self._update_history(new_df)

    def encode_categories(self, column: str, method: Literal['onehot', 'label', 'binary', 'target'], target: Optional[str] = None) -> None:
        new_df = self.current_df.copy()
        if method == 'onehot':
            new_df = pd.get_dummies(new_df, columns=[column], drop_first=True)
        elif method == 'label':
            new_df[column] = new_df[column].astype('category').cat.codes
        elif method == 'binary':
            categories = new_df[column].unique()
            n_bits = int(np.ceil(np.log2(len(categories))))
            cat_map = {cat: format(i, f'0{n_bits}b') for i, cat in enumerate(categories)}
            for bit in range(n_bits):
                new_df[f'{column}_bin{bit}'] = new_df[column].map(lambda x: int(cat_map.get(x, '0'*n_bits)[bit]))
            new_df = new_df.drop(columns=[column])
        elif method == 'target':
            if target is None:
                raise ValueError("Target column required for target encoding")
            means = new_df.groupby(column)[target].mean()
            new_df[column] = new_df[column].map(means)
        self._update_history(new_df)

    def detect_outliers(self, column: str, method: Literal['zscore', 'iqr'] = 'iqr') -> pd.Series:
        if method == 'zscore':
            z_scores = np.abs((self.current_df[column] - self.current_df[column].mean()) / self.current_df[column].std())
            return z_scores > 3
        elif method == 'iqr':
            q1 = self.current_df[column].quantile(0.25)
            q3 = self.current_df[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (self.current_df[column] < lower) | (self.current_df[column] > upper)

    def correct_outliers(self, column: str, method: Literal['remove', 'cap'], detect_method: Literal['zscore', 'iqr'] = 'iqr') -> None:
        new_df = self.current_df.copy()
        outliers = self.detect_outliers(column, detect_method)
        if method == 'remove':
            new_df = new_df[~outliers]
        elif method == 'cap':
            q1 = new_df[column].quantile(0.25)
            q3 = new_df[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            new_df[column] = np.clip(new_df[column], lower, upper)
        self._update_history(new_df)

    def reduce_dimensionality(self, n_components: int = 2, columns: Optional[list] = None) -> None:
        if columns is None:
            data = self.current_df.select_dtypes(include=[np.number])
        else:
            data = self.current_df[columns]
        data = data.fillna(0)
        data_mean = np.mean(data, axis=0)
        data_centered = data - data_mean
        cov_matrix = np.cov(data_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
        reduced_data = np.dot(data_centered, top_eigenvectors)
        new_columns = [f'PC{i+1}' for i in range(n_components)]
        reduced_df = pd.DataFrame(reduced_data, columns=new_columns, index=data.index)
        new_df = pd.concat([self.current_df.drop(columns=data.columns), reduced_df], axis=1)
        self._update_history(new_df)

    def balance_classes(self, target_column: str, method: Literal['oversample', 'undersample'] = 'oversample') -> None:
        new_df = self.current_df.copy()
        class_counts = new_df[target_column].value_counts()
        if method == 'oversample':
            max_size = class_counts.max()
            dfs = []
            for class_label, count in class_counts.items():
                class_df = new_df[new_df[target_column] == class_label]
                oversampled = class_df.sample(max_size, replace=True)
                dfs.append(oversampled)
            new_df = pd.concat(dfs)
        elif method == 'undersample':
            min_size = class_counts.min()
            dfs = [new_df[new_df[target_column] == class_label].sample(min_size) for class_label in class_counts.index]
            new_df = pd.concat(dfs)
        self._update_history(new_df)

    def split_data(self, test_size: float = 0.2, target_column: Optional[str] = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        np.random.seed(random_state)
        if target_column:
            X = self.current_df.drop(columns=[target_column])
            y = self.current_df[target_column]
        else:
            X = self.current_df
            y = None
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        test_len = int(len(X) * test_size)
        test_indices = indices[:test_len]
        train_indices = indices[test_len:]
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        if y is not None:
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            return X_train, X_test, y_train, y_test
        return X_train, X_test, None, None

    def save_to_csv(self, filename: str) -> None:
        self.current_df.to_csv(filename, index=False)