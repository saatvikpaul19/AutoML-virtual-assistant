import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings

# Suppress sklearn warnings for cleaner execution
warnings.filterwarnings("ignore")

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        """Initialize the DataCleaner with a pandas DataFrame."""
        self.df = df.copy()
        
    def get_dataframe(self) -> pd.DataFrame:
        """Return the current state of the dataframe."""
        return self.df
        
    # ── Missing Value Handling ─────────────────────────────────────
    
    def handle_missing_mean(self, columns=None):
        """1. Replace missing numeric values with the column mean."""
        self._impute_simple(columns, strategy='mean')
        
    def handle_missing_median(self, columns=None):
        """2. Replace missing numeric values with the column median."""
        self._impute_simple(columns, strategy='median')
        
    def handle_missing_mode(self, columns=None):
        """3. Replace missing categorical/numeric values with the mode."""
        self._impute_simple(columns, strategy='most_frequent')
        
    def handle_missing_constant(self, constant_value, columns=None):
        """4. Replace missing values with a specific constant."""
        cols = self._get_cols(columns)
        self.df[cols] = self.df[cols].fillna(constant_value)
        
    def fill_missing_knn(self, n_neighbors=5, columns=None):
        """5. Replace missing numeric values using K-Nearest Neighbors imputation."""
        cols = self._get_cols(columns, numeric_only=True)
        if cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            self.df[cols] = imputer.fit_transform(self.df[cols])

    def drop_missing_rows(self, threshold=None):
        """6. Drop rows containing missing values (optionally with a threshold)."""
        if threshold is None:
            self.df.dropna(inplace=True)
        else:
            self.df.dropna(thresh=threshold, inplace=True)
            
    def drop_mostly_missing_columns(self, threshold_pct=0.5):
        """7. Drop columns where the percentage of missing values exceeds threshold."""
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > threshold_pct].index
        self.df.drop(columns=cols_to_drop, inplace=True)

    # ── Scaling and Normalization ──────────────────────────────────

    def standard_scale(self, columns=None):
        """8. Scale data to have mean=0 and std=1."""
        cols = self._get_cols(columns, numeric_only=True)
        if cols:
            self.df[cols] = StandardScaler().fit_transform(self.df[cols])
            
    def minmax_scale(self, columns=None):
        """9. Scale data to fit within 0 to 1."""
        cols = self._get_cols(columns, numeric_only=True)
        if cols:
            self.df[cols] = MinMaxScaler().fit_transform(self.df[cols])
            
    def robust_scale(self, columns=None):
        """10. Scale data robustly using IQR to handle outliers."""
        cols = self._get_cols(columns, numeric_only=True)
        if cols:
            self.df[cols] = RobustScaler().fit_transform(self.df[cols])
            
    def apply_log_transform(self, columns=None):
        """11. Apply natural log transformation (log(x+1)) to handle skewness."""
        cols = self._get_cols(columns, numeric_only=True)
        for col in cols:
            # Shift data to ensure positive inputs if there are negative values
            min_val = self.df[col].min()
            if min_val < 0:
                self.df[col] = np.log1p(self.df[col] - min_val + 1)
            else:
                self.df[col] = np.log1p(self.df[col])

    # ── Categorical Encoding ───────────────────────────────────────

    def label_encode(self, columns=None):
        """12. Encode categorical column values as integers."""
        cols = self._get_cols(columns, type_includes=['object', 'category'])
        for c in cols:
            le = LabelEncoder()
            # Handle possible NaNs by casting to string
            self.df[c] = le.fit_transform(self.df[c].astype(str))
            
    def one_hot_encode(self, columns=None, drop_first=True):
        """13. Apply one-hot encoding to categorical features (auto-adds new column definitions)."""
        cols = self._get_cols(columns, type_includes=['object', 'category'])
        if cols:
            self.df = pd.get_dummies(self.df, columns=cols, drop_first=drop_first)

    # ── Outlier Handling ───────────────────────────────────────────

    def remove_outliers_zscore(self, threshold=3.0, columns=None):
        """14. Remove rows where a numeric feature's Z-score > threshold."""
        cols = self._get_cols(columns, numeric_only=True)
        mask = pd.Series(True, index=self.df.index)
        for c in cols:
            z_scores = np.abs((self.df[c] - self.df[c].mean()) / self.df[c].std())
            mask = mask & (z_scores < threshold)
        self.df = self.df[mask]

    def remove_outliers_iqr(self, multiplier=1.5, columns=None):
        """15. Remove rows falling outside (Q1 - multiplier*IQR) and (Q3 + multiplier*IQR)."""
        cols = self._get_cols(columns, numeric_only=True)
        mask = pd.Series(True, index=self.df.index)
        for c in cols:
            q1 = self.df[c].quantile(0.25)
            q3 = self.df[c].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            mask = mask & ((self.df[c] >= lower_bound) & (self.df[c] <= upper_bound))
        self.df = self.df[mask]
        
    def cap_outliers(self, lower_percentile=0.01, upper_percentile=0.99, columns=None):
        """16. Cap/Clip extreme high and low numeric values to specified percentiles (Winsorization)."""
        cols = self._get_cols(columns, numeric_only=True)
        for c in cols:
            lower = self.df[c].quantile(lower_percentile)
            upper = self.df[c].quantile(upper_percentile)
            self.df[c] = self.df[c].clip(lower=lower, upper=upper)

    # ── Feature Engineering / Dimensionality ───────────────────────
    
    def drop_columns(self, columns):
        """17. Explicitly drop a column or list of columns."""
        cols = [c for c in (columns if isinstance(columns, list) else [columns]) if c in self.df.columns]
        self.df.drop(columns=cols, inplace=True)
        
    def pca_reduce(self, n_components=2, prefix="pca_"):
        """18. Apply Principal Component Analysis to reduce numeric dimensionality."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols: return
        
        # Must handle missing before PCA
        temp_df = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        # Must scale before PCA
        temp_df = StandardScaler().fit_transform(temp_df)
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(temp_df)
        
        # Drop old numeric, add new PCA cols
        self.df.drop(columns=numeric_cols, inplace=True)
        for i in range(n_components):
            self.df[f"{prefix}{i+1}"] = components[:, i]

    def remove_duplicates(self):
        """19. Drop exact duplicate rows in the dataset."""
        self.df.drop_duplicates(inplace=True)
        
    def convert_dtypes(self):
        """20. Attempt to automatically convert column types to their optimal format."""
        self.df = self.df.convert_dtypes()

    # ── Splitting ──────────────────────────────────────────────────
    
    def train_test_split_data(self, test_size=0.2, target_column=None):
        """21. Split data into X_train, X_test, y_train, y_test. Returns dict."""
        if target_column is None:
            target_column = self.df.columns[-1]
            
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
            
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Use stratify if categorical target
        stratify = None
        if y.nunique() < 30 and y.dtype in ['object', 'category', 'int64', 'int32']:
             stratify = y
             
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        }

    # ── Helpers ────────────────────────────────────────────────────
    
    def _get_cols(self, columns, numeric_only=False, type_includes=None):
        """Helper to resolve target columns from User request."""
        if columns is None:
            if numeric_only:
                return self.df.select_dtypes(include=[np.number]).columns.tolist()
            if type_includes:
                return self.df.select_dtypes(include=type_includes).columns.tolist()
            return self.df.columns.tolist()
            
        if isinstance(columns, str):
            columns = [columns]
            
        # Ensure cols exist
        existing = [c for c in columns if c in self.df.columns]
        if numeric_only:
            existing = self.df[existing].select_dtypes(include=[np.number]).columns.tolist()
        return existing

    def _impute_simple(self, columns, strategy):
        cols = self._get_cols(columns)
        if not cols: return
        # mean/median only work on numeric
        if strategy in ['mean', 'median']:
            cols = self._get_cols(cols, numeric_only=True)
            
        if cols:
            imputer = SimpleImputer(strategy=strategy)
            self.df[cols] = imputer.fit_transform(self.df[cols])
