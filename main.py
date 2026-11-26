import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.asr import transcrire_audio
from app.config import (
    ALLOWED_AUDIO_EXTS,
    ALLOWED_AUDIO_TYPES,
    ALLOWED_IMAGE_EXTS,
    ALLOWED_IMAGE_TYPES,
    MAX_FILE_SIZE,
    STATIC_DIR,
)
from app.ocr import extraire_texte_image
from app.predictor import predire_sentiment
from app.schemas import PredictionResponse, TextRequest

logger = logging.getLogger("sentiment_api")

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    description="Texte, image (OCR) et audio (ASR) → un seul classifieur textuel.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def repondre(source: str, texte: str) -> PredictionResponse:
    try:
        resultat = predire_sentiment(texte)
    except RuntimeError as erreur:
        raise HTTPException(status_code=503, detail=str(erreur)) from erreur
    return PredictionResponse(
        source=source,
        texte_extrait=texte,
        sentiment_predit=resultat["sentiment_predit"],
        confiance=resultat["confiance"],
    )


async def lire_fichier(upload: UploadFile, type_fichier: str) -> bytes:
    nom = upload.filename or ""
    extension = Path(nom).suffix.lower()
    mime = (upload.content_type or "").split(";")[0].strip().lower()

    if type_fichier == "image":
        types, extensions = ALLOWED_IMAGE_TYPES, ALLOWED_IMAGE_EXTS
    else:
        types, extensions = ALLOWED_AUDIO_TYPES, ALLOWED_AUDIO_EXTS

    if mime not in types and extension not in extensions:
        raise HTTPException(
            status_code=422,
            detail=f"Format {type_fichier} attendu : {sorted(extensions)} (MIME {sorted(types)}).",
        )

    contenu = await upload.read(MAX_FILE_SIZE + 1)
    if not contenu:
        raise HTTPException(status_code=422, detail="Le fichier est vide.")
    if len(contenu) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Fichier trop volumineux (maxi {MAX_FILE_SIZE // 1024 // 1024} Mo).",
        )
    return contenu


@app.get("/", include_in_schema=False)
def accueil():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Diagnostic"])
def sante():
    return {"status": "ok", "version": app.version}


@app.post("/predict/text", response_model=PredictionResponse, tags=["Prédiction"])
def predict_text(donnees: TextRequest):
    return repondre("texte", donnees.texte)


@app.post("/predict/image", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_image(file: UploadFile = File(...)):
    contenu = await lire_fichier(file, "image")
    try:
        texte, methode = extraire_texte_image(contenu, file.filename or "image.png")
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur
    logger.info("Image extraite avec %s", methode)
    return repondre("image", texte)


@app.post("/predict/audio", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_audio(file: UploadFile = File(...)):
    contenu = await lire_fichier(file, "audio")
    try:
        texte, methode = transcrire_audio(contenu, file.filename or "audio.wav")
    except ValueError as erreur:
        raise HTTPException(status_code=422, detail=str(erreur)) from erreur
    logger.info("Audio transcrit avec %s", methode)
    return repondre("audio", texte)


def type_depuis_fichier(upload: StarletteUploadFile) -> str:
    nom = upload.filename or ""
    extension = Path(nom).suffix.lower()
    mime = (upload.content_type or "").split(";")[0].strip().lower()
    if mime in ALLOWED_IMAGE_TYPES or extension in ALLOWED_IMAGE_EXTS:
        return "image"
    if mime in ALLOWED_AUDIO_TYPES or extension in ALLOWED_AUDIO_EXTS:
        return "audio"
    raise HTTPException(status_code=422, detail="Type de fichier non reconnu (image ou audio attendu).")


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_auto(request: Request):
    content_type = request.headers.get("content-type", "").lower()

    if content_type.startswith("application/json"):
        try:
            donnees = TextRequest.model_validate(await request.json())
        except (ValidationError, ValueError) as erreur:
            raise HTTPException(status_code=422, detail=str(erreur)) from erreur
        return repondre("texte", donnees.texte)

    if content_type.startswith("multipart/form-data"):
        formulaire = await request.form()
        upload = next(
            (valeur for valeur in formulaire.values() if isinstance(valeur, StarletteUploadFile)),
            None,
        )
        if upload is None:
            raise HTTPException(status_code=422, detail="Aucun fichier trouvé dans la requête.")
        type_fichier = type_depuis_fichier(upload)
        contenu = await lire_fichier(upload, type_fichier)
        try:
            if type_fichier == "image":
                texte, methode = extraire_texte_image(contenu, upload.filename or "image.png")
            else:
                texte, methode = transcrire_audio(contenu, upload.filename or "audio.wav")
        except ValueError as erreur:
            raise HTTPException(status_code=422, detail=str(erreur)) from erreur
        logger.info("Détection auto : %s via %s", type_fichier, methode)
        return repondre(type_fichier, texte)

    raise HTTPException(
        status_code=422,
        detail="Envoyez du JSON application/json ou un fichier multipart/form-data.",
    )
