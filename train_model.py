import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.config import DATA_DIR, METRICS_PATH, MODEL_PATH, VECTORIZER_PATH
from app.text_utils import nettoyer_texte


def charger_donnees() -> pd.DataFrame:
    textes = pd.read_csv(DATA_DIR / "textes.csv")
    images = pd.read_csv(DATA_DIR / "images.csv")
    audios = pd.read_csv(DATA_DIR / "audios.csv")

    principal = textes[["texte", "sentiment_ref"]].rename(
        columns={"texte": "texte_brut"}
    )
    supplement = pd.concat(
        [
            images[["texte_attendu_ocr", "sentiment_ref"]].rename(
                columns={"texte_attendu_ocr": "texte_brut"}
            ),
            audios[["transcription_reference", "sentiment_ref"]].rename(
                columns={"transcription_reference": "texte_brut"}
            ),
        ],
        ignore_index=True,
    )
    donnees = pd.concat([principal, supplement], ignore_index=True)
    donnees["texte_propre"] = donnees["texte_brut"].map(nettoyer_texte)
    return donnees.dropna(subset=["texte_propre", "sentiment_ref"])


def entrainer() -> dict:
    donnees = charger_donnees()
    x_train, x_test, y_train, y_test = train_test_split(
        donnees["texte_propre"],
        donnees["sentiment_ref"],
        test_size=0.25,
        random_state=42,
        stratify=donnees["sentiment_ref"],
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        lowercase=True,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    modele = LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    modele.fit(x_train_tfidf, y_train)
    predictions = modele.predict(x_test_tfidf)

    rapport = classification_report(y_test, predictions, output_dict=True)
    matrice = confusion_matrix(y_test, predictions, labels=list(modele.classes_))
    pipeline = Pipeline([("tfidf", vectorizer), ("modele", modele)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(
        pipeline, donnees["texte_propre"], donnees["sentiment_ref"], cv=cv
    ).mean()

    metriques = {
        "nb_exemples": int(len(donnees)),
        "nb_train": int(len(x_train)),
        "nb_test": int(len(x_test)),
        "accuracy": float(rapport["accuracy"]),
        "accuracy_cv5": float(cv_accuracy),
        "labels": list(modele.classes_),
        "rapport_classification": rapport,
        "matrice_confusion": matrice.tolist(),
    }

    x_complet = vectorizer.fit_transform(donnees["texte_propre"])
    modele.fit(x_complet, donnees["sentiment_ref"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(
        json.dumps(metriques, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metriques


if __name__ == "__main__":
    resultats = entrainer()
    print(f"Exemples : {resultats['nb_exemples']} ({resultats['nb_train']} train / {resultats['nb_test']} test)")
    print(f"Accuracy (holdout 25 %) : {resultats['accuracy']:.2%}")
    print(f"Accuracy (validation croisée 5 plis) : {resultats['accuracy_cv5']:.2%}")
    print(json.dumps(resultats["rapport_classification"], ensure_ascii=False, indent=2))
    print(f"Modèles sauvegardés : {MODEL_PATH.name}, {VECTORIZER_PATH.name}")
