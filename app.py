import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Các thư viện sklearn cần thiết
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# Thư viện geopy dùng trong DistanceCalculator
try:
    from geopy.distance import geodesic
except ImportError:
    print("Warning: geopy not installed. DistanceCalculator will return dummy distances.")
    def geodesic(a, b): return 0

CITY_CENTER = (34.0522, -118.2437)  # Los Angeles center lat-lon

# === CUSTOM TRANSFORMERS ===

class DatetimeFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, date_occurred_col='Date_Occurred', date_reported_col='Date_Reported'):
        self.date_occurred_col = date_occurred_col
        self.date_reported_col = date_reported_col
        self.median_occurred_ = None
        self.median_reported_ = None

    def fit(self, X, y=None):
        X_ = X.copy()
        X_[self.date_occurred_col] = pd.to_datetime(X_[self.date_occurred_col], errors='coerce')
        X_[self.date_reported_col] = pd.to_datetime(X_[self.date_reported_col], errors='coerce')
        self.median_occurred_ = X_[self.date_occurred_col].median() or pd.Timestamp('1970-01-01')
        self.median_reported_ = X_[self.date_reported_col].median() or pd.Timestamp('1970-01-01')
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.date_occurred_col] = pd.to_datetime(X_[self.date_occurred_col], errors='coerce').fillna(self.median_occurred_)
        X_[self.date_reported_col] = pd.to_datetime(X_[self.date_reported_col], errors='coerce').fillna(self.median_reported_)
        X_['Day_O_Wday'] = X_[self.date_occurred_col].dt.weekday + 2
        X_['Day_R_Wday'] = X_[self.date_reported_col].dt.weekday + 2
        X_['Date_O_Month'] = ((X_[self.date_occurred_col].dt.day - 1) // 7) + 1
        X_['Date_R_Month'] = ((X_[self.date_reported_col].dt.day - 1) // 7) + 1
        X_['ReportingDelay'] = (X_[self.date_reported_col] - X_[self.date_occurred_col]).dt.days.clip(lower=0)
        return X_.drop([self.date_occurred_col, self.date_reported_col], axis=1)

class VictimAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, age_col='Victim_Age'):
        self.age_col = age_col
        self.median_positive_age_ = None

    def fit(self, X, y=None):
        if self.age_col in X.columns:
            positive_ages = X.loc[X[self.age_col] > 0, self.age_col]
            if not positive_ages.empty:
                self.median_positive_age_ = positive_ages.median()
            else:
                self.median_positive_age_ = X[self.age_col].median()
            if pd.isna(self.median_positive_age_):
                self.median_positive_age_ = 30
        else:
            self.median_positive_age_ = 30
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.age_col in X_.columns:
            mask_invalid = (X_[self.age_col] <= 0) | (X_[self.age_col].isnull())
            X_.loc[mask_invalid, self.age_col] = self.median_positive_age_
        else:
            X_[self.age_col] = self.median_positive_age_
        return X_

class DistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col='Latitude', lon_col='Longitude', center=CITY_CENTER):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.center = center
        self.median_distance_ = None

    def _safe_geodesic(self, row):
        try:
            lat, lon = row[self.lat_col], row[self.lon_col]
            if pd.isnull(lat) or pd.isnull(lon):
                return np.nan
            if lat == 0 and lon == 0:
                return np.nan
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return np.nan
            return geodesic(self.center, (lat, lon)).km
        except Exception:
            return np.nan

    def fit(self, X, y=None):
        distances = X.apply(self._safe_geodesic, axis=1)
        valid_distances = distances[(~distances.isnull()) & (distances > 0)]
        self.median_distance_ = valid_distances.median() if not valid_distances.empty else 10
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Distance_to_Center'] = X_.apply(self._safe_geodesic, axis=1).fillna(self.median_distance_)
        return X_

class ModusOperandiBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, mo_col='Modus_Operandi'):
        self.mo_col = mo_col
        self.mlb = MultiLabelBinarizer(sparse_output=False)
        self.columns_ = None
        self.modus_operandi_mode_ = 'Unknown'

    def _prepare_input(self, series):
        return series.astype(str).apply(lambda x: x.split() if x and x.lower() != 'nan' else [])

    def fit(self, X, y=None):
        if self.mo_col in X.columns:
            mode_val = X[self.mo_col].mode()
            self.modus_operandi_mode_ = mode_val[0] if not mode_val.empty else 'Unknown'
            X_filled = X[self.mo_col].fillna(self.modus_operandi_mode_)
        else:
            X_filled = pd.Series(['Unknown'] * len(X), index=X.index)
            self.modus_operandi_mode_ = 'Unknown'

        X_split = self._prepare_input(X_filled)
        self.mlb.fit(X_split)
        self.columns_ = [f"MO_{cls}" for cls in self.mlb.classes_]
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.mo_col in X_.columns:
            X_filled = X_[self.mo_col].fillna(self.modus_operandi_mode_)
        else:
            X_filled = pd.Series([self.modus_operandi_mode_] * len(X_), index=X_.index)
        X_split = self._prepare_input(X_filled)

        try:
            X_encoded = self.mlb.transform(X_split)
        except ValueError:
            X_encoded = np.zeros((len(X_split), len(self.columns_ if self.columns_ else [])), dtype=int)
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.columns_, index=X_.index)

        if self.mo_col in X_.columns:
            X_ = X_.drop(columns=[self.mo_col])
        X_ = pd.concat([X_, X_encoded_df], axis=1)
        return X_

# Bạn có thể thêm CrimeDataPreprocessor nếu muốn, hoặc để pipeline đã train tổng hợp

# === KHỞI TẠO APP FLASK ===

app = Flask(__name__)

# Đường dẫn model
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_MODEL_PATH = os.path.join(MODEL_DIR, "crimecast_full_pipeline_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "crimecast_label_encoder.pkl")

model_pipeline = None
label_encoder_y = None

try:
    print(f"Loading model pipeline from {PIPELINE_MODEL_PATH} ...")
    model_pipeline = joblib.load(PIPELINE_MODEL_PATH)
    print("Model pipeline loaded!")
    print(f"Loading label encoder from {LABEL_ENCODER_PATH} ...")
    label_encoder_y = joblib.load(LABEL_ENCODER_PATH)
    print("Label encoder loaded!")
except Exception as e:
    print("Error loading models:", e)
    import traceback; traceback.print_exc()

@app.route('/')
def home():
    return "Crime Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline or not label_encoder_y:
        return jsonify({"error": "Models not loaded"}), 503
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON data received"}), 400
        if isinstance(json_data, dict):
            input_df = pd.DataFrame([json_data])
        elif isinstance(json_data, list):
            input_df = pd.DataFrame(json_data)
        else:
            return jsonify({"error": "Invalid JSON format"}), 400

        preds_enc = model_pipeline.predict(input_df)
        preds_text = label_encoder_y.inverse_transform(preds_enc)

        if isinstance(json_data, list):
            results = [{"prediction": p} for p in preds_text]
            return jsonify(results)
        else:
            return jsonify({"prediction": preds_text[0]})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": "Prediction error", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)