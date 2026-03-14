# ==========================================================
# Hybrid SQL Injection Detector
# Signature Detection + Machine Learning
# Improved Version
# ==========================================================
######################################### โมเดลต้นฉบับ ใช้การ preprocessing ใหม่ #############################3
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


class SQLiDetector:

    def __init__(self):

        self.w2v_model = None
        self.scaler = None
        self.rf_model = None

        # Signature attacks
        self.signature_patterns = {

            "union_based": [
                r"\bunion\s+(all\s+)?select\b"
            ],

            "error_based": [
                r"\b(updatexml|extractvalue)\s*\("
            ],

            "time_based": [
                r"\b(sleep|benchmark|waitfor\s+delay|pg_sleep)\b"
            ],

            # NEW
            "conditional_sqli": [
                r"\bcase\s+when\b",
                r"\bselect\b.*\bselect\b"
            ]
        }

        self.fuzzy_keywords = [
            "union select",
            "sleep",
            "benchmark",
            "waitfor delay"
        ]

        self.fuzzy_threshold = 0.85


# ======================================================
# Dataset Cleaning
# ======================================================

    def clean_dataset(self, df):

        df = df.dropna(subset=["payload"])
        df = df.drop_duplicates()

        df["payload"] = df["payload"].astype(str)

        df = df[df["payload"].str.strip() != ""]
        df = df[df["payload"].str.len() >= 5]

        return df


# ======================================================
# Recursive URL Decode
# ======================================================

    def recursive_decode(self, text, max_decode=3):

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

        text = re.sub(r'0[xX][0-9a-fA-F]+', ' CONST_HEX ', text)
        text = re.sub(r'0[bB][0-9a-fA-F]+', ' CONST_HEX ', text)

        text = re.sub(r"'[^']*'", ' CONST_STR ', text)
        text = re.sub(r'"[^"]*"', ' CONST_STR ', text)

        text = re.sub(r'\b\d+(\.\d+)?\b', ' CONST_NUM ', text)

        return text


# ======================================================
# Tokenization (ปรับตรงนี้)
# ======================================================

    def tokenize(self, text):

        tokens = re.findall(
            r"select|union|case|when|then|else|end|or|and|from|where|CONST_[A-Z_]+|[a-zA-Z_]+|[=+\-*/()|]",
            text
        )

        return tokens


# ======================================================
# Statistical Features (เพิ่ม SQL logic)
# ======================================================

    def extract_stat_features(self, text):

        return np.array([

            len(text),
            text.count("'"),
            text.count("="),
            text.count(";"),
            text.count("--"),

            text.count(" or "),
            text.count(" and "),

            text.count("select"),
            text.count("union"),

            # NEW
            text.count("case"),
            text.count("when"),
            text.count("then"),
            text.count("else")

        ])


# ======================================================
# Transform
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
# Signature Detection
# ======================================================

    def signature_check(self, text):

        text = self.normalize(text)

        for name, patterns in self.signature_patterns.items():

            for pattern in patterns:

                if re.search(pattern, text):

                    return True, name

        tokens = re.findall(r"[a-zA-Z0-9_]+", text)

        for keyword in self.fuzzy_keywords:

            for token in tokens:

                score = SequenceMatcher(None, token, keyword).ratio()

                if score >= self.fuzzy_threshold:

                    return True, keyword

        return False, None


# ======================================================
# Train Model
# ======================================================

    def train_and_evaluate(self, df):

        df = self.clean_dataset(df)

        X = df["payload"]
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,
            test_size=0.1,
            stratify=y,
            random_state=42
        )

        X_train_trans = X_train.apply(self.transform)
        X_test_trans = X_test.apply(self.transform)

        X_train_tokens = X_train_trans.apply(lambda x: x[0])
        X_test_tokens = X_test_trans.apply(lambda x: x[0])

        X_train_stat = np.vstack(X_train_trans.apply(lambda x: x[1]))
        X_test_stat = np.vstack(X_test_trans.apply(lambda x: x[1]))

        print("Training Word2Vec...")

        self.w2v_model = Word2Vec(

            sentences=X_train_tokens,
            vector_size=120,
            window=5,
            min_count=1,   # ปรับ
            workers=4,
            sg=1,
            epochs=40

        )

        X_train_vec = np.vstack(X_train_tokens.apply(self.get_vector))
        X_test_vec = np.vstack(X_test_tokens.apply(self.get_vector))

        X_train_all = np.hstack([X_train_vec, X_train_stat])
        X_test_all = np.hstack([X_test_vec, X_test_stat])

        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train_all)
        X_test_scaled = self.scaler.transform(X_test_all)

        print("Training RandomForest...")

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

        y_pred = self.rf_model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
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
# Predict
# ======================================================

    def predict_single(self, payload):

        if self.rf_model is None:
            raise Exception("Model not loaded")

        sig_detect, reason = self.signature_check(payload)

        if sig_detect:

            return {

                "prediction": "BLOCKED",
                "stage": "Signature Detection",
                "reason": reason

            }

        tokens, stat = self.transform(payload)

        vec = self.get_vector(tokens)

        features = np.hstack([vec, stat])

        features_scaled = self.scaler.transform([features])

        pred = self.rf_model.predict(features_scaled)[0]

        prob = self.rf_model.predict_proba(features_scaled)[0][1]

        return {

            "prediction": "BLOCKED" if pred == 1 else "ALLOW",
            "stage": "Machine Learning",
            "malicious_probability": float(prob)

        }