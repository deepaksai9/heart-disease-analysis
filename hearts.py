import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

def clean_column_names(df):
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    return df

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)

def preprocess_heart_data(df):
    df = clean_column_names(df)
    for col in df.columns:
        if df[col].dtype == "object":
            nunique = df[col].nunique()
            if nunique <= 20:
                df[col] = df[col].astype("category").cat.codes
            else:
                top_vals = df[col].value_counts().nlargest(10).index
                df[col + "_grouped"] = df[col].apply(lambda x: x if x in top_vals else "other")
                dummies = pd.get_dummies(df[col + "_grouped"], prefix=sanitize_filename(col), drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col, col + "_grouped"])
    return df

def descriptive_analysis(df):
    print("\n====== DESCRIPTIVE ANALYSIS ======\n")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print(df.describe(include='all').T)
    print("\nValue counts for each column:\n")
    for col in df.columns:
        print(f"{col}:")
        print(df[col].value_counts())
        print()

def eda_charts(df, output_dir="eda_charts"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEDA charts will be saved in: {output_dir}/\n")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        fname_hist = sanitize_filename(col) + "_hist.png"
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname_hist}")
        plt.close()

        fname_box = sanitize_filename(col) + "_box.png"
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname_box}")
        plt.close()

    if len(num_cols) > 1:
        plt.figure(figsize=(10,8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()
    print("Charts saved. Review them for visual insights.\n")

def predictive_analysis(df):
    print("\n====== PREDICTIVE ANALYSIS ======\n")
    target_col = "target"
    if target_col not in df.columns:
        print("Target column not found. Skipping predictive analysis.")
        return
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"Random Forest Classifier - Accuracy: {acc:.2f}\n")
    print(report)
    print("Confusion Matrix:\n", cm)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("Top features influencing heart disease predictions:\n")
    print(feature_importances.sort_values(ascending=False).head(7))
    return feature_importances

def prescriptive_analysis(feature_importances):
    print("\n====== PRESCRIPTIVE ANALYSIS ======\n")
    high_impact = feature_importances.sort_values(ascending=False).head(3).index.tolist()
    msg = (
        f"Actionable Insights:\n"
        f"- Most influential features: {', '.join(high_impact)}.\n"
        f"- Focus on these features for early diagnosis or prevention.\n"
    )
    print(msg)

def main():
    if len(sys.argv) != 2:
        print("Usage: python heart_disease_analysis.py heart.csv")
        return
    fname = sys.argv[1]
    print(f"Loading data from {fname} ...\n")
    df = pd.read_csv(fname)
    df_processed = preprocess_heart_data(df)
    print("Preprocessing complete. Columns after preprocessing:\n")
    print(list(df_processed.columns))
    descriptive_analysis(df_processed)
    eda_charts(df_processed)
    feature_importances = predictive_analysis(df_processed)
    if feature_importances is not None:
        prescriptive_analysis(feature_importances)

if __name__ == "__main__":
    main()
