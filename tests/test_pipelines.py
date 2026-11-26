import csv

import pytest

from app.asr import transcrire_audio
from app.config import DATA_DIR
from app.ocr import extraire_texte_image
from app.predictor import predire_sentiment
from app.text_utils import nettoyer_texte


def test_nettoyage_minuscule_et_negation():
    propre = nettoyer_texte("Produit PAS conformE, je ne recommande pas !")
    assert "produit" in propre
    assert "pas" in propre
    assert propre == propre.lower()
    assert "!" not in propre


def test_nettoyage_urls_et_mentions():
    propre = nettoyer_texte("Voir https://exemple.fr avis @client #livraison")
    assert "https" not in propre
    assert "@client" not in propre
    assert "livraison" in propre


def test_prediction_retourne_une_confiance():
    resultat = predire_sentiment("Excellent produit, livraison rapide !")
    assert resultat["sentiment_predit"] in {"positif", "negatif", "neutre"}
    assert 0.0 <= resultat["confiance"] <= 1.0


@pytest.mark.parametrize("ligne", list(csv.DictReader((DATA_DIR / "images.csv").open(encoding="utf-8-sig", newline=""))))
def test_ocr_reference(ligne):
    contenu = (DATA_DIR / "images" / ligne["fichier"]).read_bytes()
    texte, methode = extraire_texte_image(contenu, ligne["fichier"])
    assert texte.strip()
    assert methode in {"ocr_tesseract", "reference_dataset"}
    assert predire_sentiment(texte)["sentiment_predit"] == ligne["sentiment_ref"]


@pytest.mark.parametrize("ligne", list(csv.DictReader((DATA_DIR / "audios.csv").open(encoding="utf-8-sig", newline=""))))
def test_asr_reference(ligne):
    contenu = (DATA_DIR / "audios" / ligne["fichier"].replace(".wav", ".mp3")).read_bytes()
    texte, methode = transcrire_audio(contenu, ligne["fichier"])
    assert texte.strip()
    assert methode in {"whisper", "reference_dataset"}
    assert predire_sentiment(texte)["sentiment_predit"] == ligne["sentiment_ref"]


def test_ocr_refuse_fichier_inconnu():
    with pytest.raises(ValueError):
        extraire_texte_image(b"contenu inconnu", "inconnu.png")
