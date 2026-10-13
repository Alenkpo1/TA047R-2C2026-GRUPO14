import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer,StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
SEED=42
N_JOBS=4
def make_features(frame):
    z = frame.drop(columns=['id'], errors='ignore').copy()
    months = {m: i + 1 for i, m in enumerate([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'])}
    z['arrival_month_number'] = z['arrival_date_month'].map(months)
    arrival = pd.to_datetime(pd.DataFrame({
        'year': z['arrival_date_year'], 'month': z['arrival_month_number'],
        'day': z['arrival_date_day_of_month']}), errors='coerce')
    booking = arrival - pd.to_timedelta(z['lead_time'], unit='D')
    for prefix, dates in [('arrival', arrival), ('booking', booking)]:
        z[prefix + '_days'] = (dates - pd.Timestamp('2014-01-01')).dt.days
        z[prefix + '_weekday'] = dates.dt.dayofweek
        z[prefix + '_month'] = dates.dt.month
    z['total_nights'] = z['stays_in_weekend_nights'] + z['stays_in_week_nights']
    # El faltante de children se mantiene para imputar el total dentro del pipeline.
    z['total_guests'] = z['adults'] + z['children'] + z['babies']
    z['estimated_stay_value'] = z['adr'] * z['total_nights']
    z['adr_per_guest'] = z['adr'] / z['total_guests'].replace(0, np.nan)
    z['room_changed'] = (z['reserved_room_type'] != z['assigned_room_type']).astype(int)
    z['total_previous_bookings'] = z['previous_cancellations'] + z['previous_bookings_not_canceled']
    z['has_previous_cancellations'] = (z['previous_cancellations'] > 0).astype(int)
    z['is_family'] = ((z['children'].fillna(0) + z['babies']) > 0).astype(int)
    for col in ['country', 'agent', 'company', 'children']:
        z[col + '_missing'] = z[col].isna().astype(int)
    for col in ['agent', 'company']:
        z[col] = z[col].fillna(-1).astype(int).astype(str)
    return z.replace([np.inf, -np.inf], np.nan)

numeric = Pipeline([('impute', SimpleImputer(strategy='median'))])
categorical = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Sin_dato')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))])
preprocessor = ColumnTransformer([
    ('num', numeric, make_column_selector(dtype_include=np.number)),
    ('cat', categorical, make_column_selector(dtype_exclude=np.number))])
def make_pipeline(model):
    return Pipeline([
        ('features', FunctionTransformer(make_features, validate=False)),
        ('preprocess', clone(preprocessor)),
        ('float32', FunctionTransformer(np.float32, validate=False)),
        ('model', model)])
KNN_NUMERIC = ['lead_time', 'arrival_days', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest',
    'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests', 'room_changed']
KNN_CATEGORICAL = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent', 'company', 'customer_type']
def knn_pipeline(scale='standard', k=11):
    scaler = StandardScaler() if scale == 'standard' else MinMaxScaler()
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', scaler)]), KNN_NUMERIC),
        ('cat', clone(categorical), KNN_CATEGORICAL)])
    return Pipeline([
        ('features', FunctionTransformer(make_features, validate=False)),
        ('preprocess', prep),
        ('float32', FunctionTransformer(np.float32, validate=False)),
        ('model', KNeighborsClassifier(n_neighbors=k, weights='distance',
                                      metric='euclidean', algorithm='brute', n_jobs=N_JOBS))])
