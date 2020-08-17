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

class DropChangeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, targetfinal):
        self.targetfinal = targetfinal
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        if self.targetfinal in data.columns:
            data.drop(labels=self.columns, axis='columns')
        return data

class ChangeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, targetfinal):
        self.targetfinal = targetfinal
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        #Los valores faltantes se llenan con la mediana del tipo de perfil
        if self.targetfinal in data.columns:
            for col in data.columns:
                if(col!=self.targetfinal):
                    data[col] = data.groupby(self.targetfinal)[col].apply(lambda x: x.fillna(x.median()))

            #Se elimina las filas con un NaN en la columna profile.
            #data.dropna(inplace=True)
            #data.drop(labels=[self.targetfinal], axis='columns')

        return data