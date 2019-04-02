import pandas as pd

def data_cleanup(df):

    #Converts columns with dates as strings to datetime objects
    df['last_trip_datetime'] = pd.to_datetime(df["last_trip_date"])
    df["signup_date_datetime"] = pd.to_datetime(df["last_trip_date"])
    df = df.drop(columns=["last_trip_date", "signup_date"])

    #If no trips were seen in June or July, Churn is set to True
    df["churn"] = df['last_trip_datetime'].dt.month < 6

    #Takes the mode of Avg rating by driver and of driver
    df.loc[:,'avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mode()[0], inplace=True)
    df.loc[:,'avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mode()[0], inplace = True)

    #Drops all phone rows that are NA
    df.dropna(inplace=True,axis = 0, how='any')

    return df





