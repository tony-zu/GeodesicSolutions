import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from dataclasses import dataclass, field
from collections import namedtuple

from sklearn.linear_model import LinearRegression
from gplearn.genetic import SymbolicRegressor
from sklearn.svm import SVR
from sklearn import svm,pipeline
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, max_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, median_absolute_error

def load_data(datapath: str) -> (pd.DataFrame, pd.DataFrame):
    '''
    :param datapath:
    :return:
        (maindf, factordf)

        TODO:
        describe what the columns in maindf contain?
        factordf seems to contain the independent variables - i.e. the X used in the fitting and later in forecast
    '''
    allcsv = glob.glob('*.{}'.format('csv'))

    maindf = pd.read_pickle('mainportdf.pkl')
    maindf = maindf['2000-01-01':]

    modpre = 'r8'
    factordf = pd.read_pickle(modpre + '_q5.pkl')
    factordf = pd.DataFrame(factordf)
    # factordf['date'] = pd.to_datetime(factordf['date'],format='%Y%m')
    # factordf = factordf.set_index('date')
    finfles = glob.glob(modpre + "*.csv")

    return (maindf, factordf)


def Looper(maindf: pd.DataFrame, factordf: pd.DataFrame) -> pd.DataFrame:
    '''

    :param maindf:
        data frame containing dependent variables per model data... need more explanation here
    :param factordf:
        data frame containing independent variables.
        Time series that has entire history
    :return:
        pd.DataFrame containing results data frame
    '''

    resarr = []
    for i,col in enumerate(maindf.columns[:2]):
        flename = modpre + '_' + col + '.csv'
        if flename not in finfles:
            print(flename)
            print("Model Num: " + str(i))
            if flename not in allcsv:
                mrgdf = pd.concat([maindf[col], factordf], axis=1).dropna()
            # mrgdf is the combined panel but then why redirect this way?
            y = mrgdf[col]
            X = mrgdf[factordf.columns]

            # scaler = preprocessing.StandardScaler().fit(X)

            # TODO: make this a parameter separately. Conceptually have a panel of Y, X_k where k=1,N for N-dim independent variables
            # Abstraction... data to form panel and then have the concept of ticking forward
            winsz = 120  # ten year win size

            respred = []
            resac = []

            print('Port: ' + col)


Method = namedtuple('Method',['name', 'preprocessor', 'regressor'])

@dataclass
class OracleLab:
    '''Class for wrapping experiments using different techniques applied to panel data'''
    the_panel: pd.DataFrame
    y_colname:str = ''
    x_colnames: list = field(default_factory=list)

    window_size: int = 0   # must be less than the timeseries lenght in the input panel

    methods: list = field(default_factory=list)  # list of tuples defining sklearn pipelines


    def add_method(self, name, preprocessor, regressor):
        '''
        :param name:   Name provided by user
        :param preprocessor:  Must be a valid preprocessor from sklearn
        :param regressor:  Must be a valid regressor from sklearn
        :return:
        '''
        self.methods.append(Method(name, preprocessor, regressor))
        return

    def run_experiments(self):
        if len(self.methods) == 0:
            raise ValueError('Must register methods to use the lab')
        if self.y_colname == '':
            raise ValueError('Must register names for the y and X variables')
        if self.window_size == 0:
            raise ValueError('Window size cannot be zero')

        y = self.the_panel[self.y_colname]
        X = self.the_panel[self.x_colnames]

        resname = []
        respred = []
        resac = []

        for i in range(len(y) - self.window_size):
            print (i, ', ', end='', flush=True)
            tmpname = []
            tmppred = []
            tmpac = []

            # fitting data
            tX = X[i:i+self.window_size].values
            ty = y[i:i+self.window_size].values
            # forecast data
            TX = X[i+self.window_size:i+self.window_size+1].values
            Ty = y[i+self.window_size:i+self.window_size+1].values

            for method in self.methods:
                tmpname.append(method.name)
                pipe = make_pipeline(method.preprocessor, method.regressor)
                pipe.fit(tX, ty)
                # forecast
                tp = float(pipe.predict(TX))
                tmppred.append(float(tp))  #  double casting?
                tmpac.append(float(Ty))

                resname.append(tmpname)
                respred.append(tmppred)
                resac.append(tmpac)
        df_pred = pd.DataFrame(respred)
        df_act = pd.DataFrame(resac)
        print (len(resname))
        print (len(respred))
        print (len(resac))

        return df_pred, df_act

def compute_errors(predicted: pd.DataFrame, actuals: pd.DataFrame)-> pd.DataFrame:
    '''
    :param predicted:  forecasted results
    :param actuals:  actuals
    :return:  dataframe containing error measures
    '''
    mres = []
    for j in range(len(predicted.columns)):
        pred = predicted.iloc[:,j]
        true = actuals.iloc[:,j]

        m1 = r2_score(true, pred)
        m2 = max_error(true, pred)
        m3 = mean_absolute_percentage_error(true, pred)
        m4 = mean_absolute_error(true, pred)
        m5 = mean_squared_error(true, pred)
        m6 = median_absolute_error(true, pred)

        sr = np.mean(pred) / np.std(pred) * np.sqrt(12)

        res = [j, m1, m2, m3, m4, m5, m6, sr]
        mres.append(res)

    return pd.DataFrame(mres,columns=['mod_id','r2','max_err','mape','mae','mse','mdae','sr'])


if __name__ == '__main__':
    maindf, factordf = load_data('./')
    mrgdf = pd.concat([maindf[maindf.columns[1]], factordf], axis=1).dropna()

    ml = OracleLab(the_panel = mrgdf,
               y_colname=maindf.columns[1], x_colnames=list(factordf.columns),
               window_size = 120)

    ml.add_method('Linear Regression', StandardScaler(), LinearRegression(fit_intercept=False))
    ml.add_method('SVR', StandardScaler(), SVR())
    ml.add_method('ElasticNetCV', StandardScaler(), ElasticNetCV(random_state=0))
    ml.add_method('Ridge(alpha=1.0)', StandardScaler(), Ridge(alpha=1.0))
    ml.add_method('GradientBoostingRegressor(random_state=0)', StandardScaler(), GradientBoostingRegressor(random_state=0))

    df_pred, df_act = ml.run_experiments()
    
    fdf = compute_errors(df_pred, df_act)

    method_name_list = [k[0] for k in ml.methods]










