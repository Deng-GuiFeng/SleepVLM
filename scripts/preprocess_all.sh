#!/usr/bin/env bash
# Preprocess all MASS subsets: render waveform images and compute band power targets.
#
# Usage:
#   bash scripts/preprocess_all.sh
#
# This script processes all 5 MASS subsets:
#   - SS1 (labeled, test set)
#   - SS2 (unlabeled, Phase 1 WPT)
#   - SS3 (labeled, Phase 2 SFT)
#   - SS4 (unlabeled, Phase 1 WPT)
#   - SS5 (unlabeled, Phase 1 WPT)
#
# For labeled subsets (SS1, SS3), sleep stage annotations are loaded from Base.edf.
# For unlabeled subsets (SS2, SS4, SS5), epochs are numbered without stage labels.
# Band power targets are computed for SS2, SS4, SS5 (used in Phase 1 WPT training).

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
NUM_WORKERS="${NUM_WORKERS:-32}"

echo "========================================"
echo "MASS Dataset Preprocessing"
echo "========================================"
echo "Data directory: ${DATA_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "========================================"

# --- Labeled subsets (SS1, SS3): render with stage labels ---
for SS in SS1 SS3; do
    echo ""
    echo "--- Processing MASS-${SS} (labeled) ---"
    python -m sleepvlm.data.preprocess \
        --input_dir "${DATA_DIR}/MASS/${SS}" \
        --output_dir "${DATA_DIR}/MASS/${SS}/images" \
        --mode labeled \
        --num_workers "${NUM_WORKERS}"
done

# --- Unlabeled subsets (SS2, SS4, SS5): render without labels + compute band power ---
for SS in SS2 SS4 SS5; do
    echo ""
    echo "--- Processing MASS-${SS} (unlabeled + band power) ---"
    python -m sleepvlm.data.preprocess \
        --input_dir "${DATA_DIR}/MASS/${SS}" \
        --output_dir "${DATA_DIR}/MASS/${SS}/images" \
        --mode unlabeled \
        --wpt_features_dir "${DATA_DIR}/MASS/${SS}/wpt_features" \
        --num_workers "${NUM_WORKERS}"
done

echo ""
echo "========================================"
echo "Preprocessing Complete"
echo "========================================"
echo "Rendered images:"
for SS in SS1 SS2 SS3 SS4 SS5; do
    COUNT=$(find "${DATA_DIR}/MASS/${SS}/images" -name "*.png" 2>/dev/null | wc -l)
    echo "  MASS-${SS}: ${COUNT} images"
done
echo ""
echo "Band power JSONs:"
for SS in SS1 SS2 SS3 SS4 SS5; do
    COUNT=$(find "${DATA_DIR}/MASS/${SS}/wpt_features" -name "*.json" 2>/dev/null | wc -l)
    echo "  MASS-${SS}: ${COUNT} files"
done
echo "========================================"
