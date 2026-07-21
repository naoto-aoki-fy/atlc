# Source this file to add this repository's include directory to CPATH.
#
# Usage:
#   source setup-cpath.sh

_atlc_setup_cpath_script_dir="$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")"
_atlc_setup_cpath_include_dir="${_atlc_setup_cpath_script_dir}/include"

case ":${CPATH:-}:" in
  *":${_atlc_setup_cpath_include_dir}:"*) ;;
  *)
    if [ -n "${CPATH:-}" ]; then
      export CPATH="${_atlc_setup_cpath_include_dir}:${CPATH}"
    else
      export CPATH="${_atlc_setup_cpath_include_dir}"
    fi
    ;;
esac

unset _atlc_setup_cpath_script_dir _atlc_setup_cpath_include_dir
