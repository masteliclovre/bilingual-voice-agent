from google.cloud import translate_v3 as translate


class Translator:
    def __init__(self, project_id: str):
        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/global"


    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text:
            return ""
        resp = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": src,
                "target_language_code": tgt,
            }
        )
        return resp.translations[0].translated_text if resp.translations else ""

