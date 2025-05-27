# app.py

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# --- CÁC IMPORT CẦN THIẾT CHO CUSTOM TRANSFORMERS CỦA BẠN ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MultiLabelBinarizer # LabelEncoder đã có trong code Flask rồi
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import warnings # Để dùng warnings.filterwarnings nếu cần trong custom class, dù ở đây không có

# Thư viện geopy cho DistanceCalculator
try:
    import geopy
    from geopy.distance import geodesic
except ImportError:
    print("Warning: geopy not found. DistanceCalculator might not work as expected if models depend on it.")
    def geodesic(point1, point2): # Định nghĩa giả để tránh lỗi nếu không được dùng
        print("Warning: geopy.distance.geodesic is not available. Returning dummy distance.")
        return 0

# --- Configuration (cho DistanceCalculator) ---
CITY_CENTER = (34.0522, -118.2437) # Los Angeles Center

# ==============================================================================
# >>> ĐỊNH NGHĨA CÁC CLASS CUSTOM TRANSFORMER TỪ CODE CỦA BẠN <<<
# ==============================================================================

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
        self.median_occurred_ = X_[self.date_occurred_col].median() if not X_[self.date_occurred_col].isnull().all() else pd.Timestamp('1970-01-01')
        self.median_reported_ = X_[self.date_reported_col].median() if not X_[self.date_reported_col].isnull().all() else pd.Timestamp('1970-01-01')
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.date_occurred_col] = pd.to_datetime(X_[self.date_occurred_col], errors='coerce')
        X_[self.date_reported_col] = pd.to_datetime(X_[self.date_reported_col], errors='coerce')
        X_[self.date_occurred_col].fillna(self.median_occurred_, inplace=True)
        X_[self.date_reported_col].fillna(self.median_reported_, inplace=True)
        X_['Day_O_Wday'] = X_[self.date_occurred_col].dt.weekday + 2
        X_['Day_R_Wday'] = X_[self.date_reported_col].dt.weekday + 2
        X_['Date_O_Month'] = (X_[self.date_occurred_col].dt.day - 1) // 7 + 1
        X_['Date_R_Month'] = (X_[self.date_reported_col].dt.day - 1) // 7 + 1
        X_['ReportingDelay'] = (X_[self.date_reported_col] - X_[self.date_occurred_col]).dt.days
        X_.loc[X_['ReportingDelay'] < 0, 'ReportingDelay'] = 0
        X_.drop([self.date_occurred_col, self.date_reported_col], axis=1, inplace=True)
        return X_

class VictimAgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, age_col='Victim_Age'):
        self.age_col = age_col
        self.median_positive_age_ = None

    def fit(self, X, y=None):
        # Ensure age_col exists, if not, use a default median
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
            X_.loc[X_[self.age_col] <= 0, self.age_col] = self.median_positive_age_
            X_[self.age_col].fillna(self.median_positive_age_, inplace=True)
        else:
            X_[self.age_col] = self.median_positive_age_
        return X_

class DistanceCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col='Latitude', lon_col='Longitude', center=CITY_CENTER): # Sử dụng CITY_CENTER đã định nghĩa ở trên
        self.lat_col = lat_col; self.lon_col = lon_col; self.center = center; self.median_distance_ = None

    def _safe_geodesic(self, row):
        try:
            lat = row[self.lat_col]; lon = row[self.lon_col]
            if pd.notnull(lat) and pd.notnull(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                if lat == 0 and lon == 0: return np.nan
                return geodesic(self.center, (lat, lon)).km
            return np.nan
        except (KeyError, ValueError):
            return np.nan

    def fit(self, X, y=None):
        X_temp = X.copy()
        if self.lat_col not in X_temp.columns: X_temp[self.lat_col] = np.nan
        if self.lon_col not in X_temp.columns: X_temp[self.lon_col] = np.nan
        temp_distances = X_temp.apply(self._safe_geodesic, axis=1)
        self.median_distance_ = temp_distances[temp_distances.notna() & (temp_distances > 0)].median()
        if pd.isna(self.median_distance_): self.median_distance_ = 10
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.lat_col not in X_.columns: X_[self.lat_col] = np.nan
        if self.lon_col not in X_.columns: X_[self.lon_col] = np.nan
        X_['Distance_to_Center'] = X_.apply(self._safe_geodesic, axis=1)
        X_['Distance_to_Center'].fillna(self.median_distance_, inplace=True)
        return X_

class ModusOperandiBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, mo_col='Modus_Operandi'):
        self.mo_col = mo_col; self.mlb = MultiLabelBinarizer(sparse_output=False); self.columns_ = None; self.modus_operandi_mode_ = 'Unknown'

    def _prepare_input(self, series):
        return series.astype(str).apply(lambda x: x.split(" ") if x and x.lower() != 'nan' else [])

    def fit(self, X, y=None):
        if self.mo_col in X.columns:
            mode_series = X[self.mo_col].fillna('Unknown_placeholder_for_mode_calc')
            if not mode_series.mode().empty:
                self.modus_operandi_mode_ = mode_series.mode()[0]
                if self.modus_operandi_mode_ == 'Unknown_placeholder_for_mode_calc': self.modus_operandi_mode_ = 'Unknown'
            X_filled = X[self.mo_col].fillna(self.modus_operandi_mode_)
        else:
            X_filled = pd.Series([''] * len(X), index=X.index, name=self.mo_col)
            self.modus_operandi_mode_ = ''
        X_split = self._prepare_input(X_filled); self.mlb.fit(X_split)
        self.columns_ = [f"MO_{cls}" for cls in self.mlb.classes_]; return self

    def transform(self, X):
        X_ = X.copy()
        if self.mo_col in X_.columns:
            X_filled = X_[self.mo_col].fillna(self.modus_operandi_mode_)
        else:
            X_filled = pd.Series([self.modus_operandi_mode_] * len(X_), index=X_.index, name=self.mo_col)
        X_split = self._prepare_input(X_filled)
        try:
            X_encoded = self.mlb.transform(X_split)
        except ValueError:
            X_encoded = np.zeros((len(X_split), len(self.columns_ if self.columns_ else [])), dtype=int) # Fallback
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.columns_, index=X_.index)
        if self.mo_col in X_.columns: X_ = X_.drop(self.mo_col, axis=1)
        X_ = pd.concat([X_, X_encoded_df], axis=1); return X_

class CrimeDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop_initial = [
            'Location', 'Cross_Street', 'Area_ID', 'Area_Name',
            'Weapon_Description', 'Premise_Description', 'Status_Description'
        ]
        self.cols_to_drop_final = ['Longitude', 'Latitude', 'Time_Occurred']
        self.col_victim_age = 'Victim_Age'
        self.col_weapon_code = 'Weapon_Used_Code'
        self.col_modus_operandi = 'Modus_Operandi'
        self.cols_mode_impute_cat = ['Victim_Sex', 'Victim_Descent']
        self.cols_ordinal_encode = ['Victim_Descent', 'Victim_Sex', 'Status']

        self.dt_creator_ = DatetimeFeatureCreator()
        self.dist_calc_ = DistanceCalculator() # CITY_CENTER sẽ được truyền từ hằng số toàn cục
        self.victim_age_imputer_ = VictimAgeImputer(age_col=self.col_victim_age)
        self.mo_binarizer_ = ModusOperandiBinarizer(mo_col=self.col_modus_operandi)
        self.mode_imputer_cat_ = SimpleImputer(strategy='most_frequent')
        self.median_imputer_numeric_ = SimpleImputer(strategy='median')
        self.ordinal_encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.variance_threshold_ = VarianceThreshold(threshold=1e-4)
        self.scaler_ = StandardScaler()
        self.original_input_columns_ = None
        self.fitted_columns_after_vt_ = None
        self.mo_columns_ = None
        self.columns_to_scale_ = None

    def _ensure_column_exists(self, df, column_name, default_val=np.nan):
        if column_name not in df.columns:
            df[column_name] = default_val
        return df

    def fit(self, X, y=None):
        print("Fitting CrimeDataPreprocessor...")
        self.original_input_columns_ = X.columns.tolist()
        X_ = X.copy()
        # Đảm bảo các cột cơ bản mà các transformer sẽ dùng đều có mặt
        essential_cols = ['Date_Occurred', 'Date_Reported', 'Latitude', 'Longitude',
                          self.col_victim_age, self.col_weapon_code, self.col_modus_operandi] \
                         + self.cols_mode_impute_cat + self.cols_ordinal_encode
        for col in essential_cols:
            X_ = self._ensure_column_exists(X_, col)

        X_.drop(columns=self.cols_to_drop_initial, inplace=True, errors='ignore')

        self.dist_calc_.fit(X_)
        X_temp_transform = self.dist_calc_.transform(X_)
        self.dt_creator_.fit(X_temp_transform)
        X_temp_transform = self.dt_creator_.transform(X_temp_transform)
        self.victim_age_imputer_.fit(X_temp_transform)
        X_temp_transform = self.victim_age_imputer_.transform(X_temp_transform)
        X_temp_transform[self.col_weapon_code] = X_temp_transform[self.col_weapon_code].fillna(-1)
        self.mo_binarizer_.fit(X_temp_transform)
        self.mo_columns_ = self.mo_binarizer_.columns_
        X_temp_transform = self.mo_binarizer_.transform(X_temp_transform)
        self.mode_imputer_cat_.fit(X_temp_transform[self.cols_mode_impute_cat])
        X_temp_transform[self.cols_mode_impute_cat] = self.mode_imputer_cat_.transform(X_temp_transform[self.cols_mode_impute_cat])
        self.ordinal_encoder_.fit(X_temp_transform[self.cols_ordinal_encode])
        X_temp_transform[self.cols_ordinal_encode] = self.ordinal_encoder_.transform(X_temp_transform[self.cols_ordinal_encode])
        X_temp_transform.drop(columns=self.cols_to_drop_final, inplace=True, errors='ignore')
        numeric_cols = X_temp_transform.select_dtypes(include=np.number).columns.tolist()
        numeric_cols_for_impute = [col for col in numeric_cols if not (self.mo_columns_ and col in self.mo_columns_)]
        if numeric_cols_for_impute and not X_temp_transform[numeric_cols_for_impute].empty:
            self.median_imputer_numeric_.fit(X_temp_transform[numeric_cols_for_impute])
            X_temp_transform[numeric_cols_for_impute] = self.median_imputer_numeric_.transform(X_temp_transform[numeric_cols_for_impute])
        else: self.median_imputer_numeric_ = None
        if X_temp_transform.empty:
            self.fitted_columns_after_vt_ = []; self.columns_to_scale_ = []; self.scaler_ = None; return self
        self.variance_threshold_.fit(X_temp_transform)
        self.fitted_columns_after_vt_ = X_temp_transform.columns[self.variance_threshold_.get_support()].tolist()
        X_temp_transform = X_temp_transform[self.fitted_columns_after_vt_]
        self.columns_to_scale_ = [col for col in self.fitted_columns_after_vt_ if not (self.mo_columns_ and col in self.mo_columns_)]
        if self.columns_to_scale_ and not X_temp_transform[self.columns_to_scale_].empty:
            self.scaler_.fit(X_temp_transform[self.columns_to_scale_])
        else: self.scaler_ = None; self.columns_to_scale_ = []
        print(f"CrimeDataPreprocessor fit complete. Features after VT: {len(self.fitted_columns_after_vt_)}")
        return self

    def transform(self, X):
        print(f"Transforming with CrimeDataPreprocessor... Input shape: {X.shape}")
        X_ = X.copy()
        if self.original_input_columns_: # Đảm bảo các cột từ lúc fit có mặt
            for col in self.original_input_columns_:
                if col not in X_.columns: X_[col] = np.nan
        X_.drop(columns=self.cols_to_drop_initial, inplace=True, errors='ignore')
        essential_cols = ['Date_Occurred', 'Date_Reported', 'Latitude', 'Longitude',
                          self.col_victim_age, self.col_weapon_code, self.col_modus_operandi] \
                         + self.cols_mode_impute_cat + self.cols_ordinal_encode
        for col in essential_cols: X_ = self._ensure_column_exists(X_, col)

        X_ = self.dist_calc_.transform(X_)
        X_ = self.dt_creator_.transform(X_)
        X_ = self.victim_age_imputer_.transform(X_)
        X_[self.col_weapon_code] = X_[self.col_weapon_code].fillna(-1)
        X_ = self.mo_binarizer_.transform(X_)
        X_[self.cols_mode_impute_cat] = self.mode_imputer_cat_.transform(X_[self.cols_mode_impute_cat])
        X_[self.cols_ordinal_encode] = self.ordinal_encoder_.transform(X_[self.cols_ordinal_encode])
        X_.drop(columns=self.cols_to_drop_final, inplace=True, errors='ignore')

        if self.median_imputer_numeric_ and hasattr(self.median_imputer_numeric_, 'feature_names_in_'):
            expected_numeric_cols = [c for c in self.median_imputer_numeric_.feature_names_in_ if not (self.mo_columns_ and c in self.mo_columns_)]
            for col in expected_numeric_cols:
                if col not in X_.columns:
                    try: idx = list(self.median_imputer_numeric_.feature_names_in_).index(col); X_[col] = self.median_imputer_numeric_.statistics_[idx]
                    except (ValueError, IndexError): X_[col] = 0
            cols_to_impute_now = [c for c in expected_numeric_cols if c in X_.columns]
            if cols_to_impute_now and not X_[cols_to_impute_now].empty:
                X_[cols_to_impute_now] = self.median_imputer_numeric_.transform(X_[cols_to_impute_now])
        cols_to_keep_after_vt = []
        if self.fitted_columns_after_vt_ is not None: # VT đã được fit
            # Đảm bảo các MO columns (nếu có) tồn tại trước khi chọn theo VT
            if self.mo_columns_:
                for mo_col in self.mo_columns_:
                    if mo_col not in X_.columns: X_[mo_col] = 0 # Thêm MO columns nếu thiếu
            for col_vt in self.fitted_columns_after_vt_:
                if col_vt not in X_.columns: X_[col_vt] = 0 # Thêm cột được VT chọn nếu thiếu
                cols_to_keep_after_vt.append(col_vt)
            if cols_to_keep_after_vt: X_ = X_[cols_to_keep_after_vt]
            elif not X_.empty : X_ = pd.DataFrame(index=X_.index)
        if self.scaler_ and self.columns_to_scale_:
            cols_to_scale_now = []
            for col_s in self.columns_to_scale_:
                if col_s not in X_.columns: X_[col_s] = 0.0
                cols_to_scale_now.append(col_s)
            if cols_to_scale_now and not X_[cols_to_scale_now].empty:
                 X_[cols_to_scale_now] = self.scaler_.transform(X_[cols_to_scale_now])
        print(f"CrimeDataPreprocessor transform complete. Output shape: {X_.shape}")
        return X_

# ==============================================================================
# >>> KẾT THÚC PHẦN CUSTOM TRANSFORMERS <<<
# ==============================================================================

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- Đường dẫn đến các file model ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_MODEL_PATH = os.path.join(MODEL_DIR, "crimecast_full_pipeline_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "crimecast_label_encoder.pkl")

# --- Load models ---
model_pipeline = None
label_encoder_y = None
try:
    print(f"Attempting to load pipeline model from: {PIPELINE_MODEL_PATH}")
    model_pipeline = joblib.load(PIPELINE_MODEL_PATH)
    print("Pipeline model loaded successfully!")
    print(f"Attempting to load label encoder from: {LABEL_ENCODER_PATH}")
    label_encoder_y = joblib.load(LABEL_ENCODER_PATH)
    print("Label encoder loaded successfully!")
except FileNotFoundError as e:
    print(f"MODEL LOADING ERROR - FileNotFoundError: {e}.")
except ModuleNotFoundError as e:
    print(f"MODEL LOADING ERROR - ModuleNotFoundError: {e}. Ensure all custom class definitions are in app.py before joblib.load().")
    import traceback; traceback.print_exc()
except Exception as e:
    print(f"MODEL LOADING ERROR - An unexpected error occurred: {e}")
    import traceback; traceback.print_exc()

# --- Định nghĩa các API Endpoints ---
@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline or not label_encoder_y:
        return jsonify({"error": "Models are not available. Please check server logs for loading errors."}), 503
    try:
        json_data = request.get_json()
        if not json_data: return jsonify({"error": "No input data provided in JSON format."}), 400
        if isinstance(json_data, list): input_df = pd.DataFrame(json_data)
        elif isinstance(json_data, dict): input_df = pd.DataFrame([json_data])
        else: return jsonify({"error": "Invalid JSON data format."}), 400
        print(f"Received data for prediction. Shape: {input_df.shape}, Columns: {input_df.columns.tolist()}")
        predictions_encoded = model_pipeline.predict(input_df)
        predictions_text = label_encoder_y.inverse_transform(predictions_encoded)
        if isinstance(json_data, list):
            results = [{"predicted_crime_category": pred} for pred in predictions_text]
            return jsonify(results)
        else:
            return jsonify({"predicted_crime_category": predictions_text[0]})
    except ValueError as e:
        print(f"ValueError during prediction: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Invalid data for prediction.", "details": str(e)}), 400
    except Exception as e:
        print("An unexpected error occurred during prediction:"); import traceback; traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred on the server.", "details": str(e)}), 500

# --- Chạy ứng dụng Flask ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)