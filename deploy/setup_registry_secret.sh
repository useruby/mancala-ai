#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
identity_dir="$HOME/.config/mancala-ai"
identity="$identity_dir/age-identity.txt"
secret_dir="$ROOT/deploy/secrets"
recipient_file="$secret_dir/ghcr-recipient.txt"
secret_file="$secret_dir/ghcr-registry-password.age"
kamal_dir="$ROOT/.kamal"
kamal_secrets="$kamal_dir/secrets.mancala-ai"
kamal_secret_line='KAMAL_REGISTRY_PASSWORD=$KAMAL_REGISTRY_PASSWORD'
rotate=0

if [[ "${1:-}" == "--rotate" && $# == 1 ]]; then
  rotate=1
elif [[ $# -ne 0 ]]; then
  echo "usage: deploy/setup_registry_secret.sh [--rotate]" >&2
  exit 2
fi

command -v age >/dev/null || {
  echo "age is required (Arch Linux: sudo pacman -S age)" >&2
  exit 1
}
command -v age-keygen >/dev/null || {
  echo "age-keygen is required (Arch Linux: sudo pacman -S age)" >&2
  exit 1
}

umask 077
mkdir -p "$identity_dir" "$secret_dir"
chmod 0700 "$identity_dir"

if [[ -e "$identity" && ( ! -f "$identity" || -L "$identity" ) ]]; then
  echo "age identity must be a regular file: $identity" >&2
  exit 1
fi
if [[ ! -e "$identity" ]]; then
  age-keygen -o "$identity" >/dev/null
fi
chmod 0600 "$identity"

recipient="$(age-keygen -y "$identity")"
if [[ -e "$recipient_file" ]]; then
  [[ "$(<"$recipient_file")" == "$recipient" ]] || {
    echo "stored age recipient does not match the local identity" >&2
    exit 1
  }
else
  printf '%s\n' "$recipient" > "$recipient_file"
  chmod 0644 "$recipient_file"
fi

if [[ -e "$kamal_secrets" && ( ! -f "$kamal_secrets" || -L "$kamal_secrets" ) ]]; then
  echo "Kamal secrets path must be a regular file: $kamal_secrets" >&2
  exit 1
fi
if [[ -f "$kamal_secrets" ]] && \
  ! grep -Fqx "$kamal_secret_line" "$kamal_secrets" && \
  grep -q '^KAMAL_REGISTRY_PASSWORD=' "$kamal_secrets"; then
  echo "KAMAL_REGISTRY_PASSWORD is already configured differently in $kamal_secrets" >&2
  exit 1
fi

if [[ -f "$secret_file" && "$rotate" -ne 1 ]]; then
  if ! existing_token="$(age --decrypt --identity "$identity" "$secret_file")"; then
    echo "could not decrypt existing GHCR registry credential" >&2
    exit 1
  fi
  [[ -n "$existing_token" ]] || {
    echo "existing GHCR registry credential decrypts to an empty value" >&2
    exit 1
  }
  unset existing_token
else
  read -r -s -p 'GHCR token with write:packages permission: ' token
  printf '\n'
  [[ -n "$token" ]] || {
    echo "registry token cannot be empty" >&2
    exit 1
  }

  temporary="$(mktemp "$secret_dir/.ghcr-registry-password.XXXXXX")"
  trap 'rm -f "$temporary"; unset token decrypted_token' EXIT
  printf '%s' "$token" | age --encrypt --recipient "$recipient" --output "$temporary"
  if ! decrypted_token="$(age --decrypt --identity "$identity" "$temporary")"; then
    echo "could not verify encrypted GHCR registry credential" >&2
    exit 1
  fi
  [[ "$decrypted_token" == "$token" ]] || {
    echo "encrypted GHCR registry credential verification failed" >&2
    exit 1
  }
  unset token decrypted_token
  chmod 0644 "$temporary"
  mv -f "$temporary" "$secret_file"
  trap - EXIT
fi

mkdir -p "$kamal_dir"
if [[ ! -f "$kamal_secrets" ]]; then
  printf '%s\n' "$kamal_secret_line" > "$kamal_secrets"
elif ! grep -Fqx "$kamal_secret_line" "$kamal_secrets"; then
  printf '%s\n' "$kamal_secret_line" >> "$kamal_secrets"
fi
chmod 0600 "$kamal_secrets"

printf 'Encrypted GHCR credential saved to %s\n' "$secret_file"
printf 'Private age identity remains outside the repository at %s\n' "$identity"
printf 'Kamal secret loader configured at %s\n' "$kamal_secrets"
