import re
import unicodedata

STOPWORDS_FR = {
    "a", "au", "aux", "ce", "cet", "cette", "ces", "c", "d", "de", "des", "du",
    "elle", "en", "est", "et", "eux", "il", "ils", "j", "je", "l", "la", "le",
    "les", "leur", "leurs", "lui", "m", "ma", "mais", "me", "mon",
    "nous", "on", "ou", "par", "pour", "qu", "que", "qui", "s", "sa",
    "se", "son", "sur", "t", "te", "tu", "un", "une", "y", "vous", "dans",
    "avec", "sans", "chez", "sont", "été", "etre", "avoir", "aura", "aurait",
    "ferait", "fait", "faire", "si", "comme", "soit", "meme", "aussi",
}

NEGATIONS = {"ne", "pas", "plus", "rien", "jamais", "aucun", "aucune", "ni", "non"}

REGEX_URL = re.compile(r"https?://\S+|www\.\S+")
REGEX_MENTION = re.compile(r"@\w+")
REGEX_HASHTAG = re.compile(r"#(\w+)")
REGEX_ETOILES = re.compile(r"[★⭐️]+")
REGEX_SYMBOLES = re.compile(r"[^\w'\sÀ-ÿ]")
REGEX_ESPACES = re.compile(r"\s+")


def nettoyer_texte(texte: str) -> str:
    texte = unicodedata.normalize("NFKC", texte).lower().replace("’", "'")
    texte = REGEX_URL.sub(" ", texte)
    texte = REGEX_MENTION.sub(" ", texte)
    texte = REGEX_HASHTAG.sub(r" \1 ", texte)
    texte = REGEX_ETOILES.sub(" cinq etoiles ", texte)
    texte = REGEX_SYMBOLES.sub(" ", texte)
    texte = REGEX_ESPACES.sub(" ", texte).strip()
    mots = [
        mot
        for mot in texte.split()
        if len(mot) > 1 and (mot in NEGATIONS or mot not in STOPWORDS_FR)
    ]
    return " ".join(mots) if mots else texte
