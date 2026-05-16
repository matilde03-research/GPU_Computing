#!/usr/bin/env bash
# Download the 11 SuiteSparse matrices used in the SpMV deliverable into ./data/.
# Each archive is fetched as <Group>/<Name>.tar.gz from the SuiteSparse mirror
# and extracted in place; existing matrices are skipped.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data
cd data

BASE="https://suitesparse-collection-website.herokuapp.com/MM"

# group:name pairs taken from the headers of the .mtx files in data/.
MATRICES=(
    "GHS_indef:boyd2"
    "Oberwolfach:bone010"
    "PARSEC:Ga41As41H72"
    "Rucci:Rucci1"
    "Freescale:FullChip"
    "Rajat:rajat31"
    "GHS_psdef:ldoor"
    "Williams:webbase-1M"
    "Sandia:ASIC_680ks"
    "LAW:eu-2005"
)

for entry in "${MATRICES[@]}"; do
    group="${entry%%:*}"
    name="${entry##*:}"

    if [[ -f "${name}/${name}.mtx" || -f "${name}.mtx" ]]; then
        echo "[skip] ${name}: already present"
        continue
    fi

    url="${BASE}/${group}/${name}.tar.gz"
    echo "[get ] ${group}/${name}"
    curl -fL --retry 3 -o "${name}.tar.gz" "${url}"
    tar -xzf "${name}.tar.gz"
    rm -f "${name}.tar.gz"

    # If the extracted folder holds a single file, flatten it to data/<name>.mtx.
    if [[ -d "${name}" ]]; then
        count=$(find "${name}" -maxdepth 1 -type f | wc -l)
        if [[ "${count}" -eq 1 ]]; then
            mv "${name}/${name}.mtx" "./${name}.mtx"
            rmdir "${name}"
        fi
    fi
done

echo "done; matrices in $(pwd)"
