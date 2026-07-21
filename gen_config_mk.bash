#!/usr/bin/env bash

set -eu

script_dir="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"
include_dir="${script_dir}/include"

include_dir="${include_dir//\$/\$\$}"
include_dir="${include_dir//#/\\#}"

printf 'CPATH := %s:$(CPATH)\nexport CPATH\n' "${include_dir}"
