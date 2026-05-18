#!/usr/bin/env bash
# Pull paper-related files from sibling branches into the current working tree.
# Read-only on those branches (uses `git show`), so it never modifies them.
#
# Usage:
#     ./scripts/consolidate_paper_branches.sh
#
# Reviews after running:  git status && git diff --stat

set -euo pipefail
cd "$(dirname "$0")/.."

declare -A FILES=(
    ["writing/intro-and-setup:paper/INTRODUCTION_DRAFT.tex"]="paper/INTRODUCTION_DRAFT.tex"
    ["writing/intro-and-setup:paper/WRITING_STATUS.md"]="paper/WRITING_STATUS.md"
    ["feature/advanced-physics:paper/sensor_localization.tex"]="paper/sensor_localization.tex"
    ["feature/advanced-physics:docs/SPATIAL_PHYSICS_INTERPRETATION.md"]="docs/SPATIAL_PHYSICS_INTERPRETATION.md"
    ["feature/xai-integration:research/XAI_PAPER_CONTRIBUTION.md"]="research/XAI_PAPER_CONTRIBUTION.md"
    ["feature/xai-integration:research/XAI_PLAN.md"]="research/XAI_PLAN.md"
    ["feature/edge-simulation:research/temperature_field_estimation.md"]="research/temperature_field_estimation.md"
)

for src in "${!FILES[@]}"; do
    dst="${FILES[$src]}"
    mkdir -p "$(dirname "$dst")"
    if git show "$src" > "$dst" 2>/dev/null; then
        echo "OK   $src -> $dst"
    else
        echo "MISS $src (skipped)"
        rm -f "$dst"
    fi
done

echo
echo "Done. Review with:  git status"
