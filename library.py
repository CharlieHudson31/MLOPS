from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
from sklearn import set_config
set_config(transform_output="pandas")  #forces built-in transformers to output df

class CustomOHETransformer(BaseEstimator, TransformerMixin):
  """
  A transformer that maps values into one hot encoding.

  This transformer follows the scikit-learn transformer interface and can be used in
  a scikit-learn pipeline.

  Parameters
  ----------
  mapping_dict : dict
      A dictionary defining the mapping from existing values to new values.
      Keys should be values present in the mapping_column, and values should
      be their desired replacements.

  Attributes
  ----------
  mapping_dict : dict
      The dictionary used for mapping values.

  """

  def __init__(self, target_column) -> None:
      """
      Initialize the CustomMappingTransformer.

      Parameters
      ----------
      target_column : string
         The name of the column to apply one hot encoding for.

      """
      self.target_column: str = target_column


  def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      """
      Fit method - performs no actual fitting operation.

      This method is implemented to adhere to the scikit-learn transformer interface
      but doesn't perform any computation.

      Parameters
      ----------
      X : pandas.DataFrame
          The input data to fit.
      y : array-like, default=None
          Ignored. Present for compatibility with scikit-learn interface.

      Returns
      -------
      self : CustomMappingTransformer
          Returns self to allow method chaining.
      """
      print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
      return self  #always the return value of fit

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      """
      Apply the mapping to the specified column in the input DataFrame.

      Parameters
      ----------
      X : pandas.DataFrame
          The DataFrame containing the column to transform.

      Returns
      -------
      pandas.DataFrame
          A copy of the input DataFrame with mapping applied to the specified column.

      Raises
      ------
      AssertionError
          If X is not a pandas DataFrame or if target_column is not in X.

      Notes
      -----
      This method provides warnings if:
      1. Keys in mapping_dict are not found in the column values
      2. Values in the column don't have corresponding keys in mapping_dict
      """
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
      missing_cols = list()
      assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__} unkown column {self.target_column}'
      warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

      X_: pd.DataFrame = X.copy()
      X_= pd.get_dummies(X_,
                             prefix=self.target_column,    #your choice
                             prefix_sep='_',     #your choice
                             columns=[self.target_column],
                             dummy_na=False,    #will try to impute later so leave NaNs in place
                             drop_first=False,   #really should be True but could screw us up later
                             dtype=int
                             )
      #X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
      return X_

  def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
      """
      Fit to data, then transform it.

      Combines fit() and transform() methods for convenience.

      Parameters
      ----------
      X : pandas.DataFrame
          The DataFrame containing the column to transform.
      y : array-like, default=None
          Ignored. Present for compatibility with scikit-learn interface.

      Returns
      -------
      pandas.DataFrame
          A copy of the input DataFrame with mapping applied to the specified column.
      """
      #self.fit(X,y)  #commented out to avoid warning message in fit
      result: pd.DataFrame = self.transform(X)
      return result
      
class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
  """
  A transformer that either drops or keeps specified columns in a DataFrame.

  This transformer follows the scikit-learn transformer interface and can be used in
  a scikit-learn pipeline. It allows for selectively keeping or dropping columns
  from a DataFrame based on a provided list.

  Parameters
  ----------
  column_list : List[str]
      List of column names to either drop or keep, depending on the action parameter.
  action : str, default='drop'
      The action to perform on the specified columns. Must be one of:
      - 'drop': Remove the specified columns from the DataFrame
      - 'keep': Keep only the specified columns in the DataFrame

  Attributes
  ----------
  column_list : List[str]
      The list of column names to operate on.
  action : str
      The action to perform ('drop' or 'keep').

  Examples
  --------
  >>> import pandas as pd
  >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
  >>>
  >>> # Drop columns example
  >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
  >>> dropped_df = dropper.fit_transform(df)
  >>> dropped_df.columns.tolist()
  ['C']
  >>>
  >>> # Keep columns example
  >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
  >>> kept_df = keeper.fit_transform(df)
  >>> kept_df.columns.tolist()
  ['A', 'C']
  """

  def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
      """
      Initialize the CustomDropColumnsTransformer.

      Parameters
      ----------
      column_list : List[str]
          List of column names to either drop or keep.
      action : str, default='drop'
          The action to perform on the specified columns.
          Must be either 'drop' or 'keep'.

      Raises
      ------
      AssertionError
          If action is not 'drop' or 'keep', or if column_list is not a list.
      """
      assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
      assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
      self.column_list: List[str] = column_list
      self.action: Literal['drop', 'keep'] = action

  def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      """
      Fit method - performs no actual fitting operation.

      This method is implemented to adhere to the scikit-learn transformer interface
      but doesn't perform any computation.

      Parameters
      ----------
      X : pandas.DataFrame
          The input data to fit.
      y : array-like, default=None
          Ignored. Present for compatibility with scikit-learn interface.

      Returns
      -------
      self : CustomDroppingColumnsTransformer
          Returns self to allow method chaining.
      """
      print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
      return self  #always the return value of fit

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      """
      Apply the mapping to the specified column in the input DataFrame.

      Parameters
      ----------
      X : pandas.DataFrame
          The DataFrame containing the column to transform.

      Returns
      -------
      pandas.DataFrame
          A copy of the input DataFrame with mapping applied to the specified column.

      Raises
      ------
      AssertionError
          If self.action is keep and there are columns in self.column_list that are not in X.
        Warning
          If self.action is drop and there are columns in self.column_list that are not in X.
      """
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
      if self.action == 'keep' and set(self.column_list)-set(X.columns.to_list()) != set():
        assert set(self.column_list)-set(X.columns.to_list()) == set(), f'{self.__class__.__name__}.transform unknown columns to keep: {set(self.column_list)-set(X.columns.to_list())}'

      unknown_cols = set(self.column_list)-set(X.columns.to_list())
      if self.action == 'drop' and unknown_cols != set():
        warnings.warn(
        "%s.transform unknown columns to drop: %s"
        % (self.__class__.__name__, unknown_cols))


      warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

      X_: pd.DataFrame = X.copy()
      if self.action == 'drop':
        X_.drop(columns=self.column_list, inplace=True, errors='ignore')
      else:
        X_ = X_[self.column_list]
      return X_

  def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
      """
      Fit to data, then transform it.

      Combines fit() and transform() methods for convenience.

      Parameters
      ----------
      X : pandas.DataFrame
          The DataFrame containing the column to transform.
      y : array-like, default=None
          Ignored. Present for compatibility with scikit-learn interface.

      Returns
      -------
      pandas.DataFrame
          A copy of the input DataFrame with mapping applied to the specified column.
      """
      #self.fit(X,y)  #commented out to avoid warning message in fit
      result: pd.DataFrame = self.transform(X)
      return result
  #your code below

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
  """
  A transformer that maps values in a specified column according to a provided dictionary.

  This transformer follows the scikit-learn transformer interface and can be used in
  a scikit-learn pipeline. It applies value substitution to a specified column using
  a mapping dictionary, which can be useful for encoding categorical variables or
  transforming numeric values.

  Parameters
  ----------
  mapping_column : str or int
      The name (str) or position (int) of the column to which the mapping will be applied.
  mapping_dict : dict
      A dictionary defining the mapping from existing values to new values.
      Keys should be values present in the mapping_column, and values should
      be their desired replacements.

  Attributes
  ----------
  mapping_dict : dict
      The dictionary used for mapping values.
  mapping_column : str or int
      The column (by name or position) that will be transformed.

  Examples
  --------
  >>> import pandas as pd
  >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
  >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
  >>> transformed_df = mapper.fit_transform(df)
  >>> transformed_df
     category
  0        1
  1        2
  2        3
  3        1
  """

  def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
      """
      Initialize the CustomMappingTransformer.

      Parameters
      ----------
      mapping_column : str or int
          The name (str) or position (int) of the column to apply the mapping to.
      mapping_dict : Dict[Hashable, Any]
          A dictionary defining the mapping from existing values to new values.

      Raises
      ------
      AssertionError
          If mapping_dict is not a dictionary.
      """
      assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
      self.mapping_dict: Dict[Hashable, Any] = mapping_dict
      self.mapping_column: Union[str, int] = mapping_column  #column to focus on

  def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      """
      Fit method - performs no actual fitting operation.

      This method is implemented to adhere to the scikit-learn transformer interface
      but doesn't perform any computation.

      Parameters
      ----------
      X : pandas.DataFrame
          The input data to fit.
      y : array-like, default=None
          Ignored. Present for compatibility with scikit-learn interface.

      Returns
      -------
      self : instance of CustomMappingTransformer
          Returns self to allow method chaining.
      """
      print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
      return self  #always the return value of fit

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      """
      Apply the mapping to the specified column in the input DataFrame.

      Parameters
      ----------
      X : pandas.DataFrame
          The DataFrame containing the column to transform.

      Returns
      -------
      pandas.DataFrame
          A copy of the input DataFrame with mapping applied to the specified column.

      Raises
      ------
      AssertionError
          If X is not a pandas DataFrame or if mapping_column is not in X.

      Notes
      -----
      This method provides warnings if:
      1. Keys in mapping_dict are not found in the column values
      2. Values in the column don't have corresponding keys in mapping_dict
      """
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
      assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
      warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

      column_set = set(X[self.mapping_column].unique())
      #now check to see if some keys are absent
      keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
      if keys_absent:
          print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

      X_: pd.DataFrame = X.copy()
      X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
      return X_

  def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
      """
      Fit to data, then transform it.

      Combines fit() and transform() methods for convenience.

      Parameters
      ----------
      X : pandas.DataFrame
          The DataFrame containing the column to transform.
      y : array-like, default=None
          Ignored. Present for compatibility with scikit-learn interface.

      Returns
      -------
      pandas.DataFrame
          A copy of the input DataFrame with mapping applied to the specified column.
      """
      #self.fit(X,y)  #commented out to avoid warning message in fit
      result: pd.DataFrame = self.transform(X)
      return result
  
class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that removes highly correlated features
    based on Pearson correlation.

    Parameters
    ----------
    threshold : float
        The correlation threshold above which features are considered too highly correlated
        and will be removed.

    Attributes
    ----------
    correlated_columns : Optional[List[Hashable]]
        A list of column names (which can be strings, integers, or other hashable types)
        that are identified as highly correlated and will be removed.
    """
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.correlated_columns = None
        return
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        df_corr = X.corr(method='pearson')
        masked_df = (df_corr.abs() > self.threshold)
        upper_mask = np.triu(masked_df, k=1).astype(bool)
        correlated_columns = [col for idx, col in enumerate(masked_df.columns) if upper_mask[:, idx].any()]
        self.correlated_columns = correlated_columns
        return
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.correlated_columns is not None, "PearsonTransformer.transform called before fit."
        X_: pd.DataFrame = X.copy()
        X_  =CustomDropColumnsTransformer(self.correlated_columns, 'drop').fit_transform(X_)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) ->pd.DataFrame:
        self.fit(X, y)
        result: pd.DataFrame = self.transform(X)
        return result
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, target_column) -> None:
      self.target_column = target_column
      self.high_wall = None
      self.low_wall = None
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
      assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
      mean = X[self.target_column].mean()
      std = X[self.target_column].std()
      self.low_wall = mean - 3 * std
      self.high_wall = mean + 3 * std
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      assert self.high_wall is not None and self.low_wall is not None, 'Sigma3Transformer.fit has not been called.'
      X_: pd.DataFrame = X.copy()
      X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
      return X_.reset_index(drop=True)
    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) ->pd.DataFrame:
      self.fit(X, y)
      result: pd.DataFrame = self.transform(X)
      return result