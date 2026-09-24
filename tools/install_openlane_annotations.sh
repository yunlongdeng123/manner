#!/usr/bin/env bash
set -euo pipefail

# Install the official OpenLane-V2 subset-A annotation layer only.  Camera
# images stay in AutoDL Public Data and are extracted selectively later.
ARCHIVE="${1:-/root/autodl-tmp/datasets/_archives/OpenLane-V2_subset_A_info.tar}"
DATA_ROOT="${2:-/root/autodl-tmp/datasets/openlanev2/OpenLane-V2}"
DEVKIT_DATA="${OPENLANEV2_DEVKIT_DATA:-/root/autodl-tmp/third_party/OpenLane-V2/data/OpenLane-V2}"
EXPECTED_MD5="95bf28ccf22583d20434d75800be065d"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "archive not found: $ARCHIVE" >&2
  exit 2
fi

ACTUAL_MD5="$(md5sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]]; then
  echo "checksum mismatch: expected=$EXPECTED_MD5 actual=$ACTUAL_MD5" >&2
  exit 3
fi

if [[ ! -f "$DEVKIT_DATA/data_dict_subset_A.json" ]]; then
  echo "official devkit data dictionary not found: $DEVKIT_DATA/data_dict_subset_A.json" >&2
  exit 4
fi

mkdir -p "$DATA_ROOT"
if find "$DATA_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "target must be empty: $DATA_ROOT" >&2
  exit 4
fi

echo "verified md5=$ACTUAL_MD5"
tar -xf "$ARCHIVE" -C "$DATA_ROOT"
cp "$DEVKIT_DATA/data_dict_subset_A.json" "$DATA_ROOT/data_dict_subset_A.json"
cp "$DEVKIT_DATA/openlanev2.md5" "$DATA_ROOT/openlanev2.md5"
echo "OPENLANEV2_ROOT=$DATA_ROOT"
echo "OPENLANEV2_DATA_DICT=$DATA_ROOT/data_dict_subset_A.json"
