import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model


from arch.__future__ import reindexing
reindexing = True

def rolling_window(month_return, end_date: str, window):
    df = month_return.copy()
    index = df.index
    end_loc = np.where(index >= end_date)[0].min()

    test_length = len(df[(df.index >= end_date)])
    forecasts = {}
    am = arch_model(df, vol='Garch', p=1,o=0,q=1, dist='Normal',rescale=False)


    for i in range(test_length):
        res = am.fit(first_obs=i, last_obs=i + end_loc, disp="off")
        temp = res.forecast(horizon=window).variance
        fcast = temp.iloc[0]
        forecasts[fcast.name] = fcast

    forecasts = pd.DataFrame(forecasts).T

    #인덱스 수정 및 컬럼 명 변경
    forecasts = forecasts.rename(columns={'h.1': 'forecast'})
    tmp = forecasts[1:-1].index
    forecasts = forecasts[0:-2]
    forecasts.index = list(tmp)

    return forecasts


