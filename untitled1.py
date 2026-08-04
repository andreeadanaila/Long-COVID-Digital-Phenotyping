

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -----------------------------------------------------------------
#  Incarcare si curatare
# -----------------------------------------------------------------
FILE_PATH = "C:/Users/monic/Desktop/mdpi/Neuro-Long COVID-212.xlsx"
SHEET_NAME = "Results_EN"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=0)


df = df.drop(index=0).reset_index(drop=True)

# -----------------------------------------------------------------
#  Definim cele 2 grupuri de coloane
# -----------------------------------------------------------------
BINARY_SYMPTOM_COLS = [
    "Difficulty concentrating",
    "Slowed thinking",
    "Confusion",
    "Forgetfulness",
    "Feeling disoriented",
    "Difficulty making decisions",
    "Difficulty retaining new information",
]

LIKERT_SCALE_COLS = [
    "The extent of your difficulty in remembering tasks or activities you intend to perform.",
    "The extent of your difficulty in recalling events that occurred to you in the past week.",
    "The extent of your difficulty in remembering the names of individuals you interact with daily.",
    "The extent of your difficulty in recognizing individuals you have previously met.",
    "The extent of your difficulty in remembering the reason for leaving your house.",
    "The extent of your difficulty during conversations: in forgetting the topic of discussion and going off track.",
    "The extent of your difficulty in performing two tasks simultaneously without getting distracted.",
    "The extent of your difficulty in effectively learning new skills.",
    "The extent of your difficulty in maintaining focus due to minor distractions and ambient noise.",
    "The extent of your difficulty in fully assessing situations when making decisions.",
    "The extent of your difficulty in distinguishing between important and unimportant aspects while performing a task.",
    "The extent of your difficulty in finding items because you placed them in the wrong location and cannot remember where.",
    "The extent of your difficulty in concentrating on studying a single topic for more than ten minutes.",
    "The extent of your difficulty in taking notes while simultaneously listening to a lecture.",
]


for col in BINARY_SYMPTOM_COLS + LIKERT_SCALE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.dropna(subset=BINARY_SYMPTOM_COLS + LIKERT_SCALE_COLS)

# -----------------------------------------------------------------
#  Construim target-ul - scor total din cele 14 intrebari,
# apoi il transformam in 0 (usor) / 1 (sever) folosind mediana
# -----------------------------------------------------------------
df["total_score"] = df[LIKERT_SCALE_COLS].sum(axis=1)

median_score = df["total_score"].median()
df["target"] = (df["total_score"] > median_score).astype(int)

print(f"Prag folosit (mediana scorului total): {median_score}")
print(f"Distributie target:\n{df['target'].value_counts()}")


X = df[BINARY_SYMPTOM_COLS]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------------------------------------------
#  Antrenare si evaluare - 2 modele simple (baseline-uri)
# -----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print("Acuratete:", round(accuracy_score(y_test, preds), 3))
    print("AUROC:", round(roc_auc_score(y_test, probs), 3))
    print(classification_report(y_test, preds, zero_division=0))

