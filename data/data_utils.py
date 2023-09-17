import pandas as pd

def create_date_df(period, values):
    dates = pd.date_range(start='20230101', periods=100)
    data = pd.DataFrame({'Date':dates, 'Value':values})
    return data