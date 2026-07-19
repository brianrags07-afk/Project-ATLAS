#!/bin/bash

set -e

BUCKET="gs://atlas-mlb-data-brian-4817"

echo "Uploading Project ATLAS files..."

gcloud storage cp -r data/master "${BUCKET}/master/" || true
gcloud storage cp -r data/daily "${BUCKET}/daily/" || true
gcloud storage cp -r reports "${BUCKET}/reports/" || true

echo "Upload complete!"
