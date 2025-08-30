"""Create a Google Translate v3 Glossary from a local CSV file and a target pair.
Usage:
    python tools/create_glossary.py \
        --project $GCP_PROJECT_ID \
        --glossary-id hr-en-domain \
        --source hr --target en \
        --csv ./glossary.csv \
        --location global

Requires GOOGLE_APPLICATION_CREDENTIALS env var.
"""
import argparse
import csv
import os
from google.cloud import translate_v3 as translate
from google.cloud import storage


def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"

def create_glossary(project: str, glossary_id: str, src: str, tgt: str, gcs_uri: str, location: str = "global"):
    client = translate.TranslationServiceClient()
    parent = f"projects/{project}/locations/{location}"
    g = translate.Glossary(
        name=f"{parent}/glossaries/{glossary_id}",
        language_pair=translate.Glossary.LanguagePair(
            source_language_code=src, target_language_code=tgt
        ),
        input_config=translate.GlossaryInputConfig(gcs_source=translate.GcsSource(input_uri=gcs_uri)),
    )
    op = client.create_glossary(parent=parent, glossary=g)
    result = op.result(timeout=600)
    print("Created glossary:", result.name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--glossary-id", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--bucket", required=True, help="GCS bucket to upload CSV")
    p.add_argument("--location", default="global")
    args = p.parse_args()


    blob_name = f"glossaries/{args.glossary_id}.csv"
    gcs_uri = upload_to_gcs(args.csv, args.bucket, blob_name)
    create_glossary(args.project, args.glossary_id, args.source, args.target, gcs_uri, args.location)


if __name__ == "__main__":
    main()