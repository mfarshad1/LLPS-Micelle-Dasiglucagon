#!/usr/bin/env bash

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$REPO/reproduced_figures"

mkdir -p "$OUT"
rm -f "$OUT"/*.pdf

echo "============================================================"
echo " LLPS manuscript — repository-only reproduction"
echo "============================================================"
echo

echo "Checking plotting scripts for external CRC paths..."

if grep -R "/groups/jwhitme1" "$REPO/plotting" --include="*.py"; then
    echo "ERROR: plotting script contains CRC path."
    exit 1
fi

echo "PASS: all plotting scripts are repository-only."
echo

FAIL=0

run_one () {
    LABEL="$1"
    SCRIPT="$2"
    EXPECT="$3"

    echo "------------------------------------------------------------"
    echo "$LABEL"
    echo "------------------------------------------------------------"

    if python "$REPO/plotting/$SCRIPT"; then

        if [ -s "$OUT/$EXPECT" ]; then
            echo "PASS: $EXPECT"
            ls -lh "$OUT/$EXPECT"
        else
            echo "FAIL: expected PDF not created."
            FAIL=1
        fi

    else
        echo "FAIL: $SCRIPT"
        FAIL=1
    fi

    echo
}

run_one \
 "Figure 3 — charge mapping" \
 "charge_mapping.py" \
 "figure3_charge_mapping.pdf"

run_one \
 "Figure 5 — phase behavior" \
 "phase_behavior.py" \
 "figure5_phase_behavior.pdf"

run_one \
 "Figure 6b — density profiles" \
 "density_profiles.py" \
 "figure6b_density_profiles.pdf"

run_one \
 "Figure 7 — binding analysis" \
 "binding_analysis.py" \
 "figure7_binding_analysis.pdf"

run_one \
 "Figure 8 — RDF" \
 "rdf_analysis.py" \
 "figure8_rdf.pdf"

run_one \
 "Figure 9 — MSD and mobility" \
 "msd_mobility.py" \
 "figure9_msd_mobility.pdf"

echo "============================================================"

if [ "$FAIL" -eq 0 ]; then
    echo "ALL REPOSITORY-BASED PYTHON FIGURES PASSED"
else
    echo "ONE OR MORE FIGURES FAILED"
fi

echo "============================================================"
echo
echo "Generated PDFs:"
find "$OUT" -maxdepth 1 -name "*.pdf" -printf '%f\n' | sort

exit "$FAIL"
