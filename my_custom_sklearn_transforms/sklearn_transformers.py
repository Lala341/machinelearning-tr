from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class ChangeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target

    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = pd.concat([X.copy(), y.copy()], axis=1, sort=False)
        #Los valores faltantes se llenan con la mediana del tipo de perfil
        for col in data.columns:
            if(col!=self.target):
                data[col] = data.groupby(self.target)[col].apply(lambda x: x.fillna(x.median()))

        #Se elimina las filas con un NaN en la columna profile.
        data.dropna( inplace=True)
        data.drop(labels=[self.target], axis='columns')

        return data