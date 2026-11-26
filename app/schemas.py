from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    texte: str = Field(min_length=1, max_length=5000)

    @field_validator("texte")
    @classmethod
    def texte_non_vide(cls, valeur: str) -> str:
        valeur_nettoyee = valeur.strip()
        if not valeur_nettoyee:
            raise ValueError("Le texte ne peut pas être vide.")
        return valeur_nettoyee


class PredictionResponse(BaseModel):
    source: str
    texte_extrait: str
    sentiment_predit: str
    confiance: float = Field(ge=0.0, le=1.0)
