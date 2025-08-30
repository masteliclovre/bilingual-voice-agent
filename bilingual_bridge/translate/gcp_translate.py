from google.cloud import translate_v3 as translate
from typing import Optional

class Translator:
    def __init__(self, project_id: str, glossary_id: Optional[str] = None, location: str = "global"):
        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/{location}"
        self.glossary = (
            f"{self.parent}/glossaries/{glossary_id}" if glossary_id else None
        )


    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text:
            return ""
        req = {
            "parent": self.parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": src,
            "target_language_code": tgt,
        }
        if self.glossary:
            req["glossary_config"] = {"glossary": self.glossary}
        resp = self.client.translate_text(request=req)
        if self.glossary and resp.glossary_translations:
            return resp.glossary_translations[0].translated_text
        return resp.glossary_translations[0].translated_text if self.glossary else resp.translations[0].translated_text
