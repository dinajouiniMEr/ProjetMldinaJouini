# Fiche de révision — Classifieur de Sentiment Multi-source

Aide personnelle pour réviser le projet et répondre aux questions du professeur.
Ouvrir dans VS Code : `code FICHE_SOUTENANCE.md` depuis le dossier du projet.

## 1. Le projet en une phrase

Une plateforme reçoit des avis sous trois formats (texte, image, audio). Les images
sont converties en texte par OCR, les audios par ASR, puis **un seul classifieur
textuel** décide du sentiment : `positif`, `negatif` ou `neutre`.

```
texte ────────────────────────┐
image → OCR (pytesseract) → texte ─┼→ nettoyage → TF-IDF → régression logistique → {sentiment, confiance}
audio → ASR (whisper) → texte ────┘
```

L'idée clé : OCR et ASR sont des **adaptateurs d'entrée**, pas des modèles de
sentiment. Ajouter une quatrième source ne demandera qu'un nouveau module
d'extraction, sans toucher au classifieur.

## 2. Les outils et leur rôle

| Outil | Rôle dans le projet |
|---|---|
| Python 3.13 | seul langage de programmation pour le calcul et l'API |
| venv + pip | environnement isolé, reproductible (`requirements.txt`) |
| pandas | lecture des 3 fichiers CSV du professeur |
| scikit-learn | `TfidfVectorizer`, `LogisticRegression`, évaluation, split |
| numpy | `argmax` sur les probabilités |
| joblib | sauvegarde de `model.pkl` et `vectorizer.pkl` |
| Pillow | ouverture/redressement de l'image |
| pytesseract + Tesseract | extraction du texte dans l'image (Tesseract absent ici) |
| openai-whisper | transcription audio (absent ici) |
| FastAPI | déclaration des routes REST |
| Pydantic | validation des corps JSON et des réponses |
| Starlette | socle de FastAPI, sert les fichiers statiques |
| Uvicorn | serveur HTTP qui exécute l'application |
| httpx + pytest | tests unitaires et d'intégration (`TestClient`) |
| HTML/CSS/JS natif | interface web servie par FastAPI elle-même |
| Swagger/OpenAPI | documentation interactive générée automatiquement |

## 3. Le trajet d'une requête (à réciter en 30 secondes)

1. Le navigateur appelle une route (`main.py:83` à `121`).
2. Pydantic vérifie le JSON ou le fichier : mauvais format → **422** (`main.py:46`).
3. Image → `extraire_texte_image()` (`app/ocr.py:9`) ; audio → `transcrire_audio()`
   (`app/asr.py:17`) ; texte → rien à convertir.
4. `repondre()` (`main.py:33`) appelle `predire_sentiment()` (`app/predictor.py:17`) :
   **le même classifieur pour les trois sources**.
5. Le classifieur charge les artefacts une seule fois (`@lru_cache`, `app/predictor.py:11`),
   nettoie avec **la même fonction qu'à l'entraînement** (`app/text_utils.py:24`),
   puis prend `argmax(predict_proba)`.
6. Réponse JSON unifiée : `source`, `texte_extrait`, `sentiment_predit`, `confiance`.

Le point commun de toute l'architecture est donc `app/predictor.py:17`.

## 4. Les 5 étapes du projet

### Étape 1 — Modélisation (`train_model.py:15`, `train_model.py:39`)
- 24 textes de `textes.csv` + 18 transcriptions de `images.csv`/`audios.csv` = **42 exemples**.
- Nettoyage : minuscules, URLs/mentions/punctuation supprimées, **négations conservées**
  (`ne`, `pas`, `plus`, `rien`), étoiles `★★★★★` transformées en `cinq etoiles`.
- `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))` : robuste à la morphologie
  française et aux petites variations (`cassé`/`cassés`) sur un petit corpus.
- `LogisticRegression(C=1, class_weight="balanced", max_iter=3000)` : multiclasse,
  rapide, et fournit `predict_proba()` pour la confiance.
- Évaluation : split stratifié 75/25 (**63,64 %** sur 11 exemples) et validation
  croisée 5 plis (**76,39 %**, plus fiable). Puis réentraînement sur les 42 exemples
  avant sérialisation. Métriques dans `models/metrics.json`.

### Étape 2 — OCR et ASR
- `app/ocr.py:9` : Pillow + pytesseract, langue `fra` si disponible, sinon `eng`.
- `app/asr.py:17` : fichier temporaire + Whisper `tiny`, `language="fr"`.
- Mode de repli (`app/references.py:17`) : recherche de la référence dans les CSV
  par **nom de fichier** puis par **empreinte MD5** (gère le décalage `.wav` annoncé /
  `.mp3` livré). Si rien n'est trouvé : `ValueError` → HTTP 422.

### Étape 3 — API FastAPI (`main.py`)
| Route | Rôle |
|---|---|
| `GET /` | interface web |
| `GET /health` | diagnostic |
| `POST /predict/text` | JSON `{"texte": "..."}` |
| `POST /predict/image` | fichier image multipart |
| `POST /predict/audio` | fichier audio multipart |
| `POST /predict` | détection automatique JSON ou fichier |

Validation : `TextRequest` (1 à 5 000 caractères, non vide) et `lire_fichier()`
(MIME ou extension autorisés, non vide, ≤ 5 Mo). Erreurs : 422 pour les entrées
invalides, 503 si les `.pkl` manquent.

### Étape 4 — Interface (`static/index.html:1`)
Quatre onglets, appel `fetch()` vers la route correspondante, badge coloré, barre
de confiance et texte extrait affiché. Aucun Streamlit à installer. `/docs` et
`/redoc` sont fournis automatiquement par FastAPI.

### Étape 5 — Tests
`python -m pytest -q` → **53 passed**.
- `tests/test_pipelines.py:12` : nettoyage, négations, OCR/ASR des 18 fichiers, erreur.
- `tests/test_api.py:17` : 200 sur toutes les routes, structure JSON exacte, 422 sur
  charges invalides, et les 18 avis du dataset conformes à `sentiment_ref`.

## 5. Questions probables et réponses courtes

**Quel problème résolvez-vous ?** — Trois formats d'avis, une seule décision métier,
sans entraîner trois modèles.

**C'est quoi l'architecture en cascade ?** — OCR/ASR produisent du texte ; un seul
classifieur décide. Nouvelle source = nouvel adaptateur uniquement.

**Pourquoi des n-grammes de caractères ?** — 42 exemples seulement : les mots sont trop
dispersés, les caractères généralisent mieux. CV 76,4 % vs ~64 % en mots.

**Pourquoi la régression logistique ?** — Multiclasse native, légère, `predict_proba()`
donne une confiance, `class_weight="balanced"` équilibre les classes.

**Pourquoi 63 % et 76 % ?** — Holdout de 11 phrases bruité ; la validation croisée est
plus représentative. Le modèle final est réentraîné sur tout le jeu après réglage.

**Pourquoi 41 % de confiance ?** — Maximum des 3 probabilités, pas un % de justesse.
Exemple : `positif 0,416 / neutre 0,389 / negatif 0,196`. Ligne de base = 33 %.

**Comment marche l'OCR ?** — Pillow + pytesseract ; repli par nom/MD5 si Tesseract absent.

**Comment marche l'ASR et le .wav/.mp3 ?** — Whisper tiny ; le CSV annonce `.wav` mais
les fichiers sont `.mp3`, la correspondance se fait par racine `A01` puis MD5.

**Comment gérez-vous les erreurs ?** — 422 Pydantic pour le JSON, 422 explicites pour
les fichiers, 503 si artefacts absents. Aucune entrée ne donne de 500.

**`/predict` vs `/predict/text` ?** — `/predict` regarde le `Content-Type` puis
l'extension/MIME pour deviner la source.

**C'est quoi Swagger ?** — Documentation générée par FastAPI depuis les routes et
schémas Pydantic : toujours synchronisée avec le code, et exécutable.

**Pas de base de données ?** — Non : CSV + `.pkl` suffisent. SQLite ne servirait que
pour un historique des prédictions.

**Pourquoi pas CamemBERT ?** — Corpus trop petit et modèle à télécharger ; évolution
future avec plus de données.

## 6. Démo live (3 minutes)

1. `/docs` → `POST /predict/text` → Try it out → avis négatif → montrer le **200** et le JSON.
2. Interface : texte `positif`, puis `I02.png` → `negatif` ; souligner que le **texte
   extrait est affiché** pour vérifier l'OCR.
3. Terminal : logs `200 OK`, puis `python -m pytest -q` → `53 passed`.

Phrase-clé : « Les trois modalités convergent vers un seul point de décision,
`predire_sentiment()` dans `app/predictor.py:17`. »

## 7. Limites et améliorations à annoncer

- Petit corpus → confiances modérées (40–55 %) : score relatif, pas fiabilité absolue.
- OCR/ASR réels nécessitent Tesseract/Whisper ; le repli garantit la démo hors ligne.
- Pas de persistance des prédictions ni de gestion d'utilisateurs.
- Améliorations : plus de données, CamemBERT, calibration, SQLite, logs structurés.

## 8. Arborescence du livrable

```
classifieur_sentiment_multimodal/
├── main.py, train_model.py, requirements.txt, pytest.ini, README.md
├── FICHE_SOUTENANCE.md   (ce fichier — aide de révision personnelle)
├── app/                  config, text_utils, predictor, ocr, asr, references, schemas
├── data/                 textes.csv, images.csv, audios.csv, images/, audios/
├── models/               model.pkl, vectorizer.pkl, metrics.json
├── static/index.html     mini-app web
├── captures/             01-interface-texte.png, 02-swagger.png, 03-image-negatif.png
└── tests/                test_api.py, test_pipelines.py
```

Le `.venv`, `__pycache__` et `.pytest_cache` ne doivent **pas** être inclus dans le ZIP.
