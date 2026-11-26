from functools import lru_cache

import joblib
import numpy as np

from app.config import MODEL_PATH, VECTORIZER_PATH
from app.text_utils import nettoyer_texte


@lru_cache(maxsize=1)
def charger_classifieur():
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise RuntimeError("Artefacts manquants : lancez d'abord `python train_model.py`.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predire_sentiment(texte: str) -> dict:
    modele, vectorizer = charger_classifieur()
    texte_propre = nettoyer_texte(texte)
    vecteur = vectorizer.transform([texte_propre])
    probabilites = modele.predict_proba(vecteur)[0]
    index = int(np.argmax(probabilites))
    return {
        "sentiment_predit": str(modele.classes_[index]),
        "confiance": round(float(probabilites[index]), 4),
        "texte_nettoye": texte_propre,
    }
