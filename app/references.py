import csv
import hashlib
from pathlib import Path

from app.config import DATA_DIR


def digest(contenu: bytes) -> str:
    return hashlib.md5(contenu).hexdigest()


def charger_references(nom_csv: str) -> list[dict[str, str]]:
    with (DATA_DIR / nom_csv).open(encoding="utf-8-sig", newline="") as fichier:
        return list(csv.DictReader(fichier))


def trouver_reference(
    contenu: bytes,
    nom_fichier: str,
    nom_csv: str,
    colonne_texte: str,
    dossier: str,
    colonne_fichier: str = "fichier",
) -> dict[str, str] | None:
    lignes = charger_references(nom_csv)
    nom = Path(nom_fichier or "").name.lower()
    stem = Path(nom_fichier or "").stem.lower()

    for ligne in lignes:
        if nom and Path(ligne[colonne_fichier]).name.lower() == nom:
            return ligne
    for ligne in lignes:
        if stem and Path(ligne[colonne_fichier]).stem.lower() == stem:
            return ligne

    empreinte = digest(contenu)
    for ligne in lignes:
        chemin = DATA_DIR / dossier / Path(ligne[colonne_fichier]).name
        if chemin.is_file() and digest(chemin.read_bytes()) == empreinte:
            return ligne
    return None
