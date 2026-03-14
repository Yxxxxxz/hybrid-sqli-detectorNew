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
#################################################### ต้นฉบับ + skeleton ########################################################333

# ======================================================
# Hybrid SQLi Detector
# ======================================================

class SQLiDetector:

    def __init__(self):

        self.w2v_model = None
        self.scaler = None
        self.rf_model = None

        # ======================================================
        # Signature patterns
        # ======================================================

        self.signature_patterns = {

            "union_based": [
                r"\bunion\s+(all\s+)?select\b"
            ],

            "error_based": [
                r"\b(updatexml|extractvalue|floor\(|geometrycollection|multipoint|exp\(|rand\()"
            ],

            "time_based": [
                r"\b(sleep|benchmark|waitfor\s+delay|pg_sleep|dbms_lock\.sleep)\b"
            ],

            "boolean_based": [
                r"\b(and|or)\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?"
            ]
        }

        # ======================================================
        # Fuzzy keywords
        # ======================================================

        self.fuzzy_keywords = [

            "union select",
            "sleep",
            "benchmark",
            "waitfor delay",
            "information_schema",
            "extractvalue",
            "updatexml"

        ]

        self.fuzzy_threshold = 0.85

    # ======================================================
    # Dataset Cleaning
    # ======================================================

    def clean_dataset(self, df):

        print("Cleaning dataset...")

        df = df.dropna(subset=["payload"])
        df = df.drop_duplicates()

        df["payload"] = df["payload"].astype(str)

        df = df[df["payload"].str.strip() != ""]
        df = df[df["payload"].str.len() >= 7]

        df = df[~df["payload"].str.match(r"^\d+$")]

        return df

    # ======================================================
    # Recursive URL Decode
    # ======================================================

    def recursive_decode(self, text, max_decode=5):

        count = 0

        while "%" in text and count < max_decode:

            new = urllib.parse.unquote(text)

            if new == text:
                break

            text = new
            count += 1

        return text

    # ======================================================
    # Remove SQL Comments
    # ======================================================

    def remove_comments(self, text):

        text = re.sub(r'(--|#).*?(\n|$)', ' ', text)
        text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)

        return text

    # ======================================================
    # Normalization
    # ======================================================

    def normalize(self, text):

        text = str(text)

        text = self.recursive_decode(text)

        text = text.lower()

        text = self.remove_comments(text)

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

    def tokenize(self, text):

        tokens = re.findall(r"[a-zA-Z_]+|CONST_[A-Z_]+", text)

        return tokens

    # ======================================================
    # Statistical Features
    # ======================================================

    def extract_stat_features(self, text):

        return np.array([
            len(text),
            text.count("'"),
            text.count(";"),
            text.count("="),
            text.count("--"),
            text.count("/*"),
            text.count(" or "),
            text.count(" and ")
        ])

    # ======================================================
    # Transform Pipeline
    # ======================================================

    def transform(self, text):

        normalized = self.normalize(text)

        skeleton = self.skeletonize(normalized)

        tokens = self.tokenize(skeleton)

        stat = self.extract_stat_features(skeleton)

        return tokens, stat

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

        for keyword in self.fuzzy_keywords:

            score = self.fuzzy_similarity(payload, keyword)

            if score >= self.fuzzy_threshold:
                return True, "fuzzy keyword"

        return False, None

    # ======================================================
    # Train + Evaluate
    # ======================================================

    def train_and_evaluate(self, df):

        df = self.clean_dataset(df)

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

        print("\nTransforming data...")

        X_train_tokens = []
        X_train_stats = []

        for text in X_train_raw:

            tokens, stat = self.transform(text)

            X_train_tokens.append(tokens)
            X_train_stats.append(stat)

        X_test_tokens = []
        X_test_stats = []

        for text in X_test_raw:

            tokens, stat = self.transform(text)

            X_test_tokens.append(tokens)
            X_test_stats.append(stat)

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

        X_train_vec = np.vstack([self.get_vector(t) for t in X_train_tokens])
        X_test_vec = np.vstack([self.get_vector(t) for t in X_test_tokens])

        X_train_stats = np.vstack(X_train_stats)
        X_test_stats = np.vstack(X_test_stats)

        X_train_final = np.hstack([X_train_vec, X_train_stats])
        X_test_final = np.hstack([X_test_vec, X_test_stats])

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train_final)
        X_test_scaled = self.scaler.transform(X_test_final)

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

        data = {
            "w2v_model": self.w2v_model,
            "scaler": self.scaler,
            "rf_model": self.rf_model
        }

        joblib.dump(data, path)

        print(f"\nModel saved to {path}")

    # ======================================================
    # Load Model
    # ======================================================

    @staticmethod
    def load_model(path="models/sqli_detector.pkl"):

        data = joblib.load(path)

        detector = SQLiDetector()

        detector.w2v_model = data["w2v_model"]
        detector.scaler = data["scaler"]
        detector.rf_model = data["rf_model"]

        print("Model loaded successfully")

        return detector

    # ======================================================
    # Predict Payload
    # ======================================================

    def predict_single(self, payload):

        payload_norm = self.normalize(payload)

        sig_detect, reason = self.signature_check(payload_norm)

        if sig_detect:

            return {
                "prediction": "BLOCKED",
                "stage": "Signature Detection",
                "reason": reason
            }

        tokens, stat = self.transform(payload)

        vec = self.get_vector(tokens)

        features = np.hstack([vec, stat])

        vec_scaled = self.scaler.transform([features])

        pred = self.rf_model.predict(vec_scaled)[0]

        prob = self.rf_model.predict_proba(vec_scaled)[0][1]

        return {
            "prediction": "BLOCKED" if pred == 1 else "ALLOW",
            "stage": "RandomForest ML",
            "malicious_probability": float(prob)
        }
