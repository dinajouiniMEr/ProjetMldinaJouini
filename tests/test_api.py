import csv

import pytest
from fastapi.testclient import TestClient

from app.config import CLASSES, DATA_DIR, MAX_FILE_SIZE
from main import app

client = TestClient(app)


def lire_csv(nom: str) -> list[dict[str, str]]:
    with (DATA_DIR / nom).open(encoding="utf-8-sig", newline="") as fichier:
        return list(csv.DictReader(fichier))


def test_health():
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_accueil_serve_l_interface():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "Voice Check" in reponse.text


@pytest.mark.parametrize(
    "texte,attendu",
    [
        ("Livraison super rapide, je recommande vivement !", "positif"),
        ("Article cassé et service client injoignable, très déçu.", "negatif"),
        ("Colis arrivé dans les délais, emballage standard.", "neutre"),
    ],
)
def test_predict_text(texte, attendu):
    reponse = client.post("/predict/text", json={"texte": texte})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert set(corps) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert corps["source"] == "texte"
    assert corps["sentiment_predit"] == attendu
    assert corps["sentiment_predit"] in CLASSES
    assert 0.0 <= corps["confiance"] <= 1.0


@pytest.mark.parametrize("charge", [{}, {"texte": "   "}, {"texte": "a" * 5001}])
def test_predict_text_charge_invalide(charge):
    assert client.post("/predict/text", json=charge).status_code == 422


def test_predict_image():
    image = (DATA_DIR / "images" / "I01.png").read_bytes()
    reponse = client.post(
        "/predict/image",
        files={"file": ("I01.png", image, "image/png")},
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["source"] == "image"
    assert "Livraison express" in corps["texte_extrait"]
    assert corps["sentiment_predit"] == "positif"


def test_predict_audio():
    audio = (DATA_DIR / "audios" / "A02.mp3").read_bytes()
    reponse = client.post(
        "/predict/audio",
        files={"file": ("A02.wav", audio, "audio/wav")},
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["source"] == "audio"
    assert "scandaleux" in corps["texte_extrait"]
    assert corps["sentiment_predit"] == "negatif"


def test_predict_auto_json_et_fichier():
    json_reponse = client.post("/predict", json={"texte": "Produit excellent, bravo !"})
    assert json_reponse.status_code == 200
    assert json_reponse.json()["source"] == "texte"

    image = (DATA_DIR / "images" / "I03.png").read_bytes()
    fichier_reponse = client.post(
        "/predict",
        files={"file": ("I03.png", image, "image/png")},
    )
    assert fichier_reponse.status_code == 200
    assert fichier_reponse.json()["source"] == "image"


def test_fichiers_invalides():
    mauvais_type = client.post(
        "/predict/image",
        files={"file": ("note.txt", b"bonjour", "text/plain")},
    )
    assert mauvais_type.status_code == 422

    trop_gros = client.post(
        "/predict/image",
        files={"file": ("gros.png", b"x" * (MAX_FILE_SIZE + 1), "image/png")},
    )
    assert trop_gros.status_code == 422

    vide = client.post(
        "/predict/audio",
        files={"file": ("vide.wav", b"", "audio/wav")},
    )
    assert vide.status_code == 422


def test_auto_type_non_reconnu():
    reponse = client.post("/predict", content="texte brut", headers={"Content-Type": "text/plain"})
    assert reponse.status_code == 422


@pytest.mark.parametrize("ligne", lire_csv("images.csv"))
def test_images_du_dataset_sont_bien_classees(ligne):
    image = (DATA_DIR / "images" / ligne["fichier"]).read_bytes()
    reponse = client.post(
        "/predict/image",
        files={"file": (ligne["fichier"], image, "image/png")},
    )
    assert reponse.status_code == 200
    assert reponse.json()["sentiment_predit"] == ligne["sentiment_ref"]


@pytest.mark.parametrize("ligne", lire_csv("audios.csv"))
def test_audios_du_dataset_sont_bien_classes(ligne):
    fichier_reel = (DATA_DIR / "audios" / ligne["fichier"].replace(".wav", ".mp3")).read_bytes()
    reponse = client.post(
        "/predict/audio",
        files={"file": (ligne["fichier"], fichier_reel, "audio/wav")},
    )
    assert reponse.status_code == 200
    assert reponse.json()["sentiment_predit"] == ligne["sentiment_ref"]
