#!/usr/bin/env bash
set -euo pipefail

# All operators must use the designated deployment runner. flock serializes
# wrapper operations there; Kamal also locks each remote operation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .kamal
exec 9>"${MANCALA_AI_DEPLOY_LOCK:-.kamal/deploy.lock}"
flock -n 9 || { echo "another mancala-ai deployment is running" >&2; exit 1; }

: "${KAMAL_HOST:?required}"
: "${KAMAL_WIREGUARD_IP:?required}"
KAMAL_SSH_USER="${KAMAL_SSH_USER:-root}"
KAMAL_SSH_PORT="${KAMAL_SSH_PORT:-22}"
KAMAL_MEMORY="${KAMAL_MEMORY:-2g}"
KAMAL_CPUS="${KAMAL_CPUS:-2.0}"
KAMAL_BUILDER_DRIVER="${KAMAL_BUILDER_DRIVER:-docker-container}"
export KAMAL_SSH_USER KAMAL_SSH_PORT KAMAL_MEMORY KAMAL_CPUS KAMAL_BUILDER_DRIVER
state="${MANCALA_AI_DEPLOY_STATE:-.kamal/mancala-ai-deployment}"
config_store="${MANCALA_AI_CONFIG_STORE:-.kamal/config-revisions}"
previous_version=""
previous_assets=""
previous_config=""
rollback_version=""
rollback_assets=""
rollback_config=""
if [[ -f "$state" ]]; then
  IFS=$'\t' read -r previous_version previous_assets previous_config \
    rollback_version rollback_assets rollback_config < "$state"
fi
if [[ -n "$previous_config" ]]; then
  values_file="$config_store/$previous_config/values.sh"
  [[ -f "$values_file" ]] || { echo "recorded deployment values are missing: $values_file" >&2; exit 2; }
  recorded_identity="$(bash -c 'source "$1"; printf "%s\t%s\t%s" "$KAMAL_HOST" "$KAMAL_SSH_USER" "$KAMAL_SSH_PORT"' _ "$values_file")" || exit 2
  printf -v requested_identity '%s\t%s\t%s' "$KAMAL_HOST" "$KAMAL_SSH_USER" "$KAMAL_SSH_PORT"
  [[ "$requested_identity" == "$recorded_identity" ]] || {
    echo "deployment target drift: configured SSH target differs from recorded target $recorded_identity" >&2
    exit 2
  }
fi
ssh_args=(-o BatchMode=yes -l "$KAMAL_SSH_USER" -p "$KAMAL_SSH_PORT")
config_path() {
  local revision="$1"
  if [[ -n "$revision" ]]; then
    printf '%s/%s/deploy.yml' "$config_store" "$revision"
  else
    printf '%s' "config/deploy.yml"
  fi
}
active_config="$(config_path "$previous_config")"
kamal() {
  bundle exec kamal "$@" -c "$active_config" -d mancala-ai
}
running_container="$(ssh "${ssh_args[@]}" "$KAMAL_HOST" \
  "docker ps --filter label=service=mancala-ai --filter label=destination=mancala-ai --filter status=running --format '{{.ID}}'")"
running_version=""
if [[ -n "$running_container" ]]; then
  running_version="$(ssh "${ssh_args[@]}" "$KAMAL_HOST" \
    "docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' '$running_container'" \
    | awk -F= '$1 == "KAMAL_VERSION" { print $2 }')"
  if [[ -z "$running_version" || "$running_version" != "$previous_version" ]]; then
    echo "running image version does not match the recorded deployment state" >&2
    exit 2
  fi
fi
old_version="${running_version:-$previous_version}"
if [[ -n "$old_version" && ( -z "$previous_version" || -z "$previous_assets" || -z "$previous_config" ) ]]; then
  echo "existing service lacks wrapper release state; record its image and asset release before updating" >&2
  exit 2
fi

if [[ "${1:-deploy}" == "rollback" ]]; then
  [[ -n "$rollback_version" && -n "$rollback_assets" && -n "$rollback_config" ]] || {
    echo "no previous complete release is recorded" >&2
    exit 2
  }
  export MANCALA_AI_ASSET_RELEASE="$rollback_assets"
  export VERSION="$rollback_version"
  active_config="$(config_path "$rollback_config")"
  [[ -f "$active_config" ]] || {
    echo "previous deployment configuration snapshot is missing: $active_config" >&2
    exit 2
  }
elif [[ "${1:-deploy}" != "deploy" ]]; then
  echo "usage: deploy/deploy.sh [deploy|rollback]" >&2
  exit 2
else
  : "${MANCALA_AI_ASSET_RELEASE:?select an absolute versioned release directory}"
fi
[[ "$MANCALA_AI_ASSET_RELEASE" = /* ]] || {
  echo "MANCALA_AI_ASSET_RELEASE must be an absolute remote directory" >&2
  exit 2
}
[[ "$MANCALA_AI_ASSET_RELEASE" != *$'\t'* && "$MANCALA_AI_ASSET_RELEASE" != *$'\n'* && "$MANCALA_AI_ASSET_RELEASE" != *$'\r'* ]] || {
  echo "MANCALA_AI_ASSET_RELEASE cannot contain tab or newline characters" >&2
  exit 2
}
quoted_asset_path="$(python -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$MANCALA_AI_ASSET_RELEASE")"
remote_asset_check="test -d $quoted_asset_path && test -r $quoted_asset_path && test -x $quoted_asset_path"
ssh "${ssh_args[@]}" "$KAMAL_HOST" "$remote_asset_check" || {
  echo "remote asset directory is missing or inaccessible: $MANCALA_AI_ASSET_RELEASE" >&2
  exit 2
}

if [[ "${1:-deploy}" == "deploy" ]]; then
  config_revision="$(python - <<'PY'
import hashlib, json, os, subprocess
keys = ("KAMAL_HOST", "KAMAL_WIREGUARD_IP", "KAMAL_SSH_USER", "KAMAL_SSH_PORT", "KAMAL_IMAGE", "KAMAL_REGISTRY_SERVER", "KAMAL_REGISTRY_USERNAME", "KAMAL_MEMORY", "KAMAL_CPUS", "KAMAL_BUILDER_DRIVER", "MANCALA_AI_ASSET_RELEASE")
payload = json.dumps({"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "values": {key: os.environ.get(key, "") for key in keys}}, sort_keys=True)
print(hashlib.sha256(payload.encode()).hexdigest())
PY
)"
  git diff --quiet -- config/deploy.yml config/deploy.mancala-ai.yml || {
    echo "deployment configuration must be committed before deployment" >&2
    exit 2
  }
  config_dir="$config_store/$config_revision"
  if [[ ! -d "$config_dir" ]]; then
    mkdir -p "$config_dir"
    cp config/deploy.yml config/deploy.mancala-ai.yml "$config_dir/"
    python - "$config_dir/values.sh" <<'PY'
import os, shlex, sys
keys = ("KAMAL_HOST", "KAMAL_WIREGUARD_IP", "KAMAL_SSH_USER", "KAMAL_SSH_PORT", "KAMAL_IMAGE", "KAMAL_REGISTRY_SERVER", "KAMAL_REGISTRY_USERNAME", "KAMAL_MEMORY", "KAMAL_CPUS", "KAMAL_BUILDER_DRIVER", "MANCALA_AI_ASSET_RELEASE")
with open(sys.argv[1], "w", encoding="utf-8") as output:
    for key in keys:
        if key in os.environ:
            output.write(f"export {key}={shlex.quote(os.environ[key])}\n")
PY
    chmod 0444 "$config_dir"/*.yml
    chmod 0444 "$config_dir/values.sh"
    chmod 0555 "$config_dir"
  elif ! cmp -s config/deploy.yml "$config_dir/deploy.yml" || \
    ! cmp -s config/deploy.mancala-ai.yml "$config_dir/deploy.mancala-ai.yml" || \
    [[ ! -f "$config_dir/values.sh" ]]; then
    echo "stored deployment configuration snapshot does not match its revision" >&2
    exit 2
  fi
  active_config="$config_dir/deploy.yml"
  [[ -f "$config_dir/values.sh" ]] || { echo "deployment values snapshot is missing" >&2; exit 2; }
else
  config_revision="$rollback_config"
  # Restore the previous release's resolved values, not today's environment.
  # shellcheck source=/dev/null
  source "$config_store/$rollback_config/values.sh"
fi
deployment_ok=0
service_stopped=0
mark_recovery_failed() {
  local failed_state="${state}.recovery-failed.$$"
  if printf 'RECOVERY_FAILED\t\t\t\t\t\n' > "$failed_state" && mv "$failed_state" "$state"; then
    echo "deployment state marked as requiring operator reconciliation" >&2
  else
    rm -f "$failed_state"
    echo "recovery failure could not be recorded in deployment state" >&2
  fi
}
recover() {
  status=$?
  if (( status != 0 && deployment_ok == 0 && service_stopped == 1 )) && [[ -n "$old_version" ]]; then
    echo "replacement failed; restoring the recorded complete release" >&2
    # A failed post-start check or state write may leave the replacement bound
    # to the fixed port. Stop it using the attempted release configuration.
    if ! bundle exec kamal app stop -c "$active_config" -d mancala-ai; then
      echo "recovery failed: could not stop replacement and release port 8080" >&2
      mark_recovery_failed
      exit "$status"
    fi
    export MANCALA_AI_ASSET_RELEASE="$previous_assets"
    export VERSION="$previous_version"
    active_config="$(config_path "$previous_config")"
    # Roll back with all values captured for the prior running release.
    # shellcheck source=/dev/null
    source "$config_store/$previous_config/values.sh"
    if ! bundle exec kamal rollback "$previous_version" -c "$active_config" -d mancala-ai; then
      echo "recovery failed: Kamal could not restore previous release $previous_version" >&2
      mark_recovery_failed
      exit "$status"
    fi
    python -c 'import time,urllib.request; u="http://"+__import__("os").environ["KAMAL_WIREGUARD_IP"]+":8080/health/ready"; end=time.monotonic()+180
while time.monotonic()<end:
 try: urllib.request.urlopen(u,timeout=3); break
 except Exception: time.sleep(1)
else: raise SystemExit("previous release did not become ready")' || {
      echo "recovery failed: previous release did not pass readiness verification" >&2
      mark_recovery_failed
      exit "$status"
    }
  fi
  exit "$status"
}
trap recover EXIT

kamal config >/dev/null
if [[ -n "$running_container" ]]; then
  # Release the fixed published port before Kamal boots a replacement.
  stop_config="$(config_path "$previous_config")"
  service_stopped=1
  (
    # Evaluate the incumbent's recorded values while issuing its stop command.
    # shellcheck source=/dev/null
    source "$config_store/$previous_config/values.sh"
    bundle exec kamal app stop -c "$stop_config" -d mancala-ai
  )
fi

if [[ "${1:-deploy}" == "rollback" ]]; then
  kamal rollback "$rollback_version"
else
  kamal deploy
fi
python -c 'import time,urllib.request; u="http://"+__import__("os").environ["KAMAL_WIREGUARD_IP"]+":8080/health/ready"; end=time.monotonic()+180
while time.monotonic()<end:
 try: urllib.request.urlopen(u,timeout=3); break
 except Exception: time.sleep(1)
else: raise SystemExit("replacement did not become ready")'

version="${VERSION:-$(git rev-parse HEAD)}"
temporary="${state}.tmp.$$"
if [[ "${1:-deploy}" == "rollback" ]]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rollback_version" "$rollback_assets" "$rollback_config" \
    "$previous_version" "$previous_assets" "$previous_config" > "$temporary"
else
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$version" "$MANCALA_AI_ASSET_RELEASE" "$config_revision" \
    "$previous_version" "$previous_assets" "$previous_config" > "$temporary"
fi
mv "$temporary" "$state"
deployment_ok=1
trap - EXIT
