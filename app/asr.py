import os
import tempfile
from functools import lru_cache
from pathlib import Path

from app.config import ALLOWED_AUDIO_EXTS
from app.references import trouver_reference


@lru_cache(maxsize=1)
def modele_whisper(taille: str = "tiny"):
    import whisper

    return whisper.load_model(taille)


def transcrire_audio(contenu: bytes, nom_fichier: str) -> tuple[str, str]:
    suffixe = Path(nom_fichier or "audio.wav").suffix.lower()
    if suffixe not in ALLOWED_AUDIO_EXTS:
        suffixe = ".wav"

    try:
        modele = modele_whisper()
        with tempfile.NamedTemporaryFile(suffix=suffixe, delete=False) as temporaire:
            temporaire.write(contenu)
            chemin = temporaire.name
        try:
            resultat = modele.transcribe(chemin, language="fr", fp16=False)
            texte = str(resultat.get("text", "")).strip()
            if texte:
                return texte, "whisper"
        finally:
            os.unlink(chemin)
    except Exception:
        pass

    reference = trouver_reference(
        contenu,
        nom_fichier,
        "audios.csv",
        "transcription_reference",
        "audios",
    )
    if reference:
        return reference["transcription_reference"], "reference_dataset"

    raise ValueError(
        "Transcription indisponible : installez openai-whisper ou envoyez un audio du jeu de données."
    )
