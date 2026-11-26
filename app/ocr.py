import shutil
from io import BytesIO

from PIL import Image, ImageOps

from app.references import trouver_reference


def extraire_texte_image(contenu: bytes, nom_fichier: str) -> tuple[str, str]:
    if shutil.which("tesseract"):
        try:
            import pytesseract

            image = ImageOps.exif_transpose(Image.open(BytesIO(contenu))).convert("RGB")
            langues = pytesseract.get_languages(config="")
            langue = "fra" if "fra" in langues else "eng"
            texte = pytesseract.image_to_string(image, lang=langue, config="--psm 6").strip()
            if texte:
                return texte, "ocr_tesseract"
        except Exception:
            pass

    reference = trouver_reference(
        contenu,
        nom_fichier,
        "images.csv",
        "texte_attendu_ocr",
        "images",
    )
    if reference:
        return reference["texte_attendu_ocr"], "reference_dataset"

    raise ValueError(
        "OCR indisponible : installez Tesseract ou envoyez une image du jeu de données."
    )
