import re
import urllib.parse
import numpy as np
import pandas as pd
import joblib

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from difflib import SequenceMatcher


# ======================================================
# Hybrid SQLi Detector
# ======================================================

class SQLiDetector:

    def __init__(self):

        self.w2v_model = None
        self.scaler = None
        self.rf_model = None

        # Signature patterns
        self.signature_patterns = {

            "union-based": [
                r"union\s+select",
                r"union\s+all\s+select"
            ],

            "error-based": [
                r"extractvalue\s*\(",
                r"updatexml\s*\(",
                r"floor\s*\(\s*rand\s*\(",
                r"group\s+by\s+.*rand",
                r"information_schema",
                r"mysql_fetch",
                r"syntax\s+error"
            ],

            "time-based": [
                r"sleep\s*\(",
                r"benchmark\s*\(",
                r"pg_sleep\s*\(",
                r"waitfor\s+delay"
            ]
        }

    # ======================================================
    # Data Cleaning
    # ======================================================

    def clean_data(self, df):

        print("Cleaning dataset...")

        df = df.dropna(subset=["payload"])
        df = df.drop_duplicates()

        df["payload"] = df["payload"].astype(str)

        df = df[df["payload"].str.strip() != ""]
        df = df[df["payload"].str.len() > 3]

        return df

    # ======================================================
    # Normalization
    # ======================================================

    def normalize(self, text):

        text = str(text)

        text = urllib.parse.unquote(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ======================================================
    # Skeletonization
    # ======================================================

    def skeletonize(self, text):

        text = re.sub(r'0x[0-9a-f]+', ' CONST_HEX ', text)

        text = re.sub(r"'[^']*'", ' CONST_STR ', text)
        text = re.sub(r'"[^"]*"', ' CONST_STR ', text)

        text = re.sub(r'\b(true|false|null)\b', ' CONST_BOOL ', text)

        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)

        return text

    # ======================================================
    # Tokenization
    # ======================================================

    def tokenize_sql(self, query):

        tokens = re.findall(
            r"[a-zA-Z_]+|\d+|!=|==|<=|>=|['\"].*?['\"]|[^\s]",
            str(query)
        )

        return tokens

    # ======================================================
    # Word2Vec Vector
    # ======================================================

    def get_vector(self, tokens):

        vectors = [
            self.w2v_model.wv[word]
            for word in tokens
            if word in self.w2v_model.wv
        ]

        if len(vectors) == 0:
            return np.zeros(self.w2v_model.vector_size)

        return np.mean(vectors, axis=0)

    # ======================================================
    # Fuzzy Similarity
    # ======================================================

    def fuzzy_similarity(self, a, b):

        return SequenceMatcher(None, a, b).ratio()

    # ======================================================
    # Signature Detection
    # ======================================================

    def signature_check(self, payload):

        payload = self.normalize(payload)

        for category, patterns in self.signature_patterns.items():

            for pattern in patterns:

                if re.search(pattern, payload):
                    return True, f"{category} signature"

                score = self.fuzzy_similarity(payload, pattern)

                if score > 0.75:
                    return True, f"fuzzy {category} signature"

        return False, None

    # ======================================================
    # Train + Evaluate
    # ======================================================

    def train_and_evaluate(self, df):

        df = self.clean_data(df)

        df["payload"] = df["payload"].apply(self.normalize)

        X = df["payload"]
        y = df["label"].values

        print("\nLabel Distribution")
        print(df["label"].value_counts())
        print("Total records:", len(df))

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.1,
            stratify=y,
            random_state=42
        )

        # ใช้ skeleton ก่อน tokenize
        X_train_tokens = X_train_raw.apply(
            lambda x: self.tokenize_sql(self.skeletonize(x))
        )

        X_test_tokens = X_test_raw.apply(
            lambda x: self.tokenize_sql(self.skeletonize(x))
        )

        print("\nTraining Word2Vec...")

        self.w2v_model = Word2Vec(
            sentences=X_train_tokens,
            vector_size=120,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=40
        )

        X_train_vec = np.vstack(X_train_tokens.apply(self.get_vector))
        X_test_vec = np.vstack(X_test_tokens.apply(self.get_vector))

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train_vec)
        X_test_scaled = self.scaler.transform(X_test_vec)

        print("\nTraining RandomForest...")

        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        self.rf_model.fit(X_train_scaled, y_train)

        self.evaluate(X_test_scaled, y_test)

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(self, X_test, y_test):

        print("\n==============================")
        print("Performance Results")
        print("==============================")

        y_pred = self.rf_model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

    # ======================================================
    # Save Model
    # ======================================================

    def save_model(self, path="models/sqli_detector.pkl"):

        data_to_save = {
            'w2v_model': self.w2v_model,
            'scaler': self.scaler,
            'rf_model': self.rf_model
        }

        joblib.dump(data_to_save, path)

        print(f"\n✅ Brain saved to {path}")

    # ======================================================
    # Load Model
    # ======================================================

    @staticmethod
    def load_model(path="models/sqli_detector.pkl"):

        data = joblib.load(path)

        new_detector = SQLiDetector()

        if isinstance(data, dict):

            new_detector.w2v_model = data.get('w2v_model')
            new_detector.scaler = data.get('scaler')
            new_detector.rf_model = data.get('rf_model')

            print("✅ Brain extracted and linked to detector.")

        else:
            new_detector = data

        print(f"✅ Model loaded from {path}")

        return new_detector

    # ======================================================
    # Predict Payload
    # ======================================================

    def predict_single(self, payload):

        if self.rf_model is None:
            raise Exception("Model not loaded")

        payload = self.normalize(payload)

        sig_detect, reason = self.signature_check(payload)

        if sig_detect:

            return {
                "prediction": "BLOCKED",
                "stage": "Signature Detection",
                "reason": reason
            }

        # ใช้ skeleton ก่อน tokenize
        payload = self.skeletonize(payload)

        tokens = self.tokenize_sql(payload)

        vec = self.get_vector(tokens)

        vec_scaled = self.scaler.transform([vec])

        pred = self.rf_model.predict(vec_scaled)[0]

        prob = self.rf_model.predict_proba(vec_scaled)[0][1]

        return {
            "prediction": "BLOCKED" if pred == 1 else "ALLOW",
            "stage": "RandomForest ML",
            "malicious_probability": float(prob)
        }
