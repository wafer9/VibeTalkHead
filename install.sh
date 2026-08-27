#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME="${SCRIPT_NAME:-zcode-install}"
DEBUG="${DEBUG:-0}"

CLAUDE_INSTALL_URL="${CLAUDE_INSTALL_URL:-https://claude.ai/install.sh}"
CLAUDE_NPM_PACKAGE="${CLAUDE_NPM_PACKAGE:-@anthropic-ai/claude-code}"
CLAUDE_NPM_PREFIX="${CLAUDE_NPM_PREFIX:-$HOME/.local}"
CLAUDE_RELEASE_URL_TEMPLATE="${CLAUDE_RELEASE_URL_TEMPLATE:-https://code.yukework.com/releases/claude/%s/%s}"
ZCODE_SCRIPT_URL="${ZCODE_SCRIPT_URL:-https://code.yukework.com/zcode.sh}"
INTERNAL_FORWARD_IP="10.57.20.8"
HOSTS_FILE="${HOSTS_FILE:-/etc/hosts}"
CURL_CONNECT_TIMEOUT_SECONDS=5
CURL_MAX_TIME_SECONDS=300
PREFERRED_INSTALL_DIR="${PREFERRED_INSTALL_DIR:-/usr/local/bin}"
FALLBACK_INSTALL_DIR="${FALLBACK_INSTALL_DIR:-$HOME/.local/bin}"
INSTALL_NAME="${INSTALL_NAME:-zcode}"
INSTALL_PATH="${INSTALL_PATH:-}"
ZCODE_HOME="${ZCODE_HOME:-$HOME/.zcode}"
SETTINGS_PATH="${ZYB_SETTINGS_PATH:-$ZCODE_HOME/settings.json}"
MARKETPLACE_SOURCE="${MARKETPLACE_SOURCE:-https://git.yukework.com/pkg/zcode-marketplace.git}"
MARKETPLACE_SCOPE="${MARKETPLACE_SCOPE:-user}"
DEFAULT_PLUGINS="${DEFAULT_PLUGINS:-zyb-observability}"
GIT_SSH_HOST="${GIT_SSH_HOST:-git.yukework.com}"
PLUGIN_SCOPE="${PLUGIN_SCOPE:-user}"

# Static (offline) marketplace: install plugins from a downloaded tarball placed
# under ~/.zcode/plugins and referenced via a local "directory" source, instead
# of cloning the git marketplace. Enabled with --static-market.
STATIC_MARKETPLACE="${STATIC_MARKETPLACE:-0}"
MARKETPLACE_ARCHIVE_URL="${MARKETPLACE_ARCHIVE_URL:-https://code.yukework.com/zcode-market.tgz}"
LOCAL_MARKETPLACE_DIR="${LOCAL_MARKETPLACE_DIR:-$ZCODE_HOME/plugins/zyb-plugins}"
LOCAL_MARKETPLACE_NAME="${LOCAL_MARKETPLACE_NAME:-zyb-plugins}"

SKIP_CLAUDE_INSTALL=0
SKIP_INIT=0
SKIP_MARKETPLACE=0
SKIP_MCP=1
ADDED_HOSTS=()

CYAN='\033[0;36m'
NC='\033[0m'
USE_COLOR=0
COLOR_RESET=""
COLOR_DIM=""
COLOR_GREEN=""
COLOR_YELLOW=""
COLOR_RED=""
COLOR_ORANGE=""


print_zcode_logo() {
    echo -e "${CYAN}"
    cat << "EOF"
███████╗ ██████╗  ██████╗  ██████╗  ███████╗
╚══███╔╝██╔════╝ ██╔═══██╗ ██║  ██╗ ██╔════╝
  ███╔╝ ██║      ██║   ██║ ██║  ██║ █████╗
 ███╔╝  ██║      ██║   ██║ ██║  ██║ ██╔══╝
███████╗╚██████╗ ╚██████╔╝ ██████╔╝ ███████╗
╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝  ╚══════╝
EOF
    echo -e "${NC}"
}


init_colors() {
  if [[ -n "${NO_COLOR:-}" ]]; then
    return 0
  fi

  if [[ -t 1 || -t 2 ]]; then
    USE_COLOR=1
    COLOR_RESET=$'\033[0m'
    COLOR_DIM=$'\033[2m'
    COLOR_GREEN=$'\033[32m'
    COLOR_YELLOW=$'\033[33m'
    COLOR_RED=$'\033[31m'
    COLOR_ORANGE=$'\033[38;5;214m'
  fi
}

colorize() {
  local color="$1"
  shift

  if (( USE_COLOR == 1 )); then
    printf '%s%s%s' "$color" "$*" "$COLOR_RESET"
  else
    printf '%s' "$*"
  fi
}

style_command() {
  colorize "$COLOR_ORANGE" "$*"
}

log() {
  if (( USE_COLOR == 1 )); then
    printf '  %s %s\n' "$(colorize "$COLOR_DIM" ".")" "$*" >&2
  else
    printf '  . %s\n' "$*" >&2
  fi
}

debug() {
  if [[ "$DEBUG" == "1" ]]; then
    if (( USE_COLOR == 1 )); then
      printf '  %s %s\n' "$(colorize "$COLOR_DIM" ".")" "$(colorize "$COLOR_DIM" "Debug: $*")" >&2
    else
      printf '  . Debug: %s\n' "$*" >&2
    fi
  fi
}

warn() {
  printf '%s %s\n' "$(colorize "$COLOR_YELLOW" "!")" "$*" >&2
}

warn_step() {
  if (( USE_COLOR == 1 )); then
    printf '  %s %s\n' "$(colorize "$COLOR_YELLOW" "!")" "$*" >&2
  else
    printf '  ! %s\n' "$*" >&2
  fi
}

die() {
  printf '%s %s\n' "$(colorize "$COLOR_RED" "✘")" "$*" >&2
  exit 1
}

done_ok() {
  printf '%s %s\n' "$(colorize "$COLOR_GREEN" "✔")" "$*"
}

usage() {
  cat <<EOF
Usage:
  curl -fsSL <internal-install-url> | bash

Optional flags:
  --debug                Enable debug logs
  --skip-claude          Do not install Claude Code
  --skip-init            Do not initialize the default settings
  --skip-market          Do not install the default plugins / marketplace
  --static-market        Install the marketplace from a static tarball
                         (local directory source) instead of git
  --skip-mcp             Do not configure MCP servers
  --install-dir DIR      Override the target bin directory
  --install-path PATH    Override the full target path
  --zcode-url URL        Override the runtime wrapper download URL
  --claude-url URL       Override the Claude install script URL
  -h, --help             Show this help

Environment overrides:
  ZCODE_SCRIPT_URL
  CLAUDE_INSTALL_URL
  CLAUDE_NPM_PACKAGE
  CLAUDE_NPM_PREFIX
  INSTALL_PATH
  ZYB_SETTINGS_PATH
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

curl_fetch() {
  local url="$1"
  shift

  curl --connect-timeout "$CURL_CONNECT_TIMEOUT_SECONDS" --max-time "$CURL_MAX_TIME_SECONDS" "$@" "$url"
}

hosts_entry_exists() {
  local host="$1"

  grep -Eq "^[[:space:]]*${INTERNAL_FORWARD_IP//./\\.}[[:space:]]+${host//./\\.}([[:space:]]|\$)" "$HOSTS_FILE" 2>/dev/null
}

append_hosts_entry() {
  local host="$1"
  local entry="$INTERNAL_FORWARD_IP $host"

  if hosts_entry_exists "$host"; then
    debug "Hosts entry already present: $entry"
    return 0
  fi

  if [[ -w "$HOSTS_FILE" ]]; then
    printf '%s\n' "$entry" >> "$HOSTS_FILE"
    ADDED_HOSTS+=("$host")
    debug "Hosts entry added: $entry"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1; then
    printf '%s\n' "$entry" | sudo tee -a "$HOSTS_FILE" >/dev/null
    ADDED_HOSTS+=("$host")
    debug "Hosts entry added with sudo: $entry"
    return 0
  fi

  return 1
}

remove_hosts_entry() {
  local host="$1"
  local tmp_file

  tmp_file="$(mktemp "${TMPDIR:-/tmp}/zcode.hosts.XXXXXX")" || return 1

  if ! awk -v ip="$INTERNAL_FORWARD_IP" -v host="$host" '
    !($1 == ip && $2 == host) { print }
  ' "$HOSTS_FILE" > "$tmp_file"; then
    rm -f "$tmp_file"
    return 1
  fi

  if [[ -w "$HOSTS_FILE" ]]; then
    cat "$tmp_file" > "$HOSTS_FILE" || {
      rm -f "$tmp_file"
      return 1
    }
  elif command -v sudo >/dev/null 2>&1; then
    sudo tee "$HOSTS_FILE" >/dev/null < "$tmp_file" || {
      rm -f "$tmp_file"
      return 1
    }
  else
    rm -f "$tmp_file"
    return 1
  fi

  rm -f "$tmp_file"
  debug "Hosts entry removed: $INTERNAL_FORWARD_IP $host"
}

ensure_internal_hosts_entries() {
  append_hosts_entry "claude.ai" || return 1
  append_hosts_entry "storage.googleapis.com" || return 1
}

cleanup_internal_hosts_entries() {
  local host
  local cleanup_failed=0

  if (( ${#ADDED_HOSTS[@]} == 0 )); then
    return 0
  fi

  debug "Cleaning up temporary internal hosts overrides"
  for host in "${ADDED_HOSTS[@]}"; do
    if ! remove_hosts_entry "$host"; then
      warn "Failed to remove temporary hosts override for $host."
      cleanup_failed=1
    fi
  done

  ADDED_HOSTS=()
  return "$cleanup_failed"
}

cleanup_internal_hosts_entries_on_exit() {
  local exit_code=$?

  cleanup_internal_hosts_entries || true
  return "$exit_code"
}

install_claude_via_npm() {
  require_cmd npm
  if [[ "$DEBUG" == "1" ]]; then
    npm install -g --prefix "$CLAUDE_NPM_PREFIX" "$CLAUDE_NPM_PACKAGE"
  else
    npm install -g --silent --prefix "$CLAUDE_NPM_PREFIX" "$CLAUDE_NPM_PACKAGE"
  fi
}

get_claude_release_url() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m | tr '[:upper:]' '[:lower:]')"
  printf "$CLAUDE_RELEASE_URL_TEMPLATE" "$os" "$arch"
}

install_claude_from_release() {
  local url target_dir target_file

  url="$(get_claude_release_url)"
  target_dir="$HOME/.local/bin"
  target_file="$target_dir/claude"

  debug "Downloading Claude from: $url"

  mkdir -p "$target_dir"

  if ! download_file "$url" "$target_file"; then
    return 1
  fi

  chmod +x "$target_file"
  debug "Claude installed to: $target_file"
}

resolve_install_path() {
  local parent_dir

  if [[ -n "$INSTALL_PATH" ]]; then
    printf '%s' "$INSTALL_PATH"
    return 0
  fi

  parent_dir="$(dirname "$PREFERRED_INSTALL_DIR")"

  if [[ -d "$PREFERRED_INSTALL_DIR" && -w "$PREFERRED_INSTALL_DIR" ]]; then
    printf '%s/%s' "$PREFERRED_INSTALL_DIR" "$INSTALL_NAME"
    return 0
  fi

  if [[ ! -d "$PREFERRED_INSTALL_DIR" && -w "$parent_dir" ]]; then
    printf '%s/%s' "$PREFERRED_INSTALL_DIR" "$INSTALL_NAME"
    return 0
  fi

  # Fallback to user-writable directory instead of requiring sudo
  mkdir -p "$FALLBACK_INSTALL_DIR"
  printf '%s/%s' "$FALLBACK_INSTALL_DIR" "$INSTALL_NAME"
}

resolve_claude_bin() {
  local candidate
  for candidate in "claude" "/usr/local/bin/claude" "/opt/homebrew/bin/claude" "$HOME/.local/bin/claude"; do
    if [[ "$candidate" == *"/"* ]]; then
      if [[ -x "$candidate" ]]; then
        printf '%s' "$candidate"
        return 0
      fi
      continue
    fi

    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

download_file() {
  local url="$1"
  local output="$2"

  curl_fetch "$url" --fail --silent --show-error --location --retry 2 --retry-delay 1 -o "$output"
}

download_zcode_script() {
  local output="$1"

  debug "Download URL: $ZCODE_SCRIPT_URL"
  download_file "$ZCODE_SCRIPT_URL" "$output"

  [[ -s "$output" ]] || die "Downloaded zcode script is empty"
  [[ "$(head -n1 "$output")" == "#!/usr/bin/env bash" ]] || die "Downloaded zcode script is not a valid bash script"
}

run_claude_native_installer() {
  if [[ "$DEBUG" == "1" ]]; then
    curl_fetch "$CLAUDE_INSTALL_URL" --fail --silent --show-error --location | bash
  else
    curl_fetch "$CLAUDE_INSTALL_URL" --fail --silent --show-error --location 2>/dev/null | bash >/dev/null 2>&1
  fi
}

install_claude_if_needed() {
  local existing_bin

  if (( SKIP_CLAUDE_INSTALL == 1 )); then
    debug "Skipped Claude Code installation by flag"
    return 0
  fi

  existing_bin="$(resolve_claude_bin || true)"
  if [[ -n "$existing_bin" ]]; then
    log "Claude Code detected. Skipping installation."
    debug "Claude Code path: $existing_bin"
    return 0
  fi

  log "Installing Claude Code..."

  # Try 1: Download from internal release server
  if install_claude_from_release; then
    existing_bin="$(resolve_claude_bin || true)"
    if [[ -n "$existing_bin" ]]; then
      debug "Claude Code installed from internal release: $existing_bin"
      return 0
    fi
  fi

  warn_step "Internal release download failed. Trying official installer..."

  # Try 2: Official installer directly
  if run_claude_native_installer; then
    existing_bin="$(resolve_claude_bin || true)"
    if [[ -n "$existing_bin" ]]; then
      debug "Claude Code installed from official installer: $existing_bin"
      return 0
    fi
  fi

  warn_step "Official installer failed. Retrying via internal proxy..."
  warn_step "The internal proxy can be slow and may not show progress. Please wait patiently."

  # Try 3: Official installer with internal hosts override
  if ensure_internal_hosts_entries && run_claude_native_installer; then
    existing_bin="$(resolve_claude_bin || true)"
    if [[ -n "$existing_bin" ]]; then
      debug "Claude Code installed via internal proxy: $existing_bin"
      return 0
    fi
  fi

  warn_step "Internal proxy failed. Falling back to npm install."

  # Try 4: npm install as last resort
  install_claude_via_npm

  existing_bin="$(resolve_claude_bin || true)"
  if [[ -n "$existing_bin" ]]; then
    debug "Claude Code installed: $existing_bin"
  else
    warn "Claude Code installation finished, but the command was not found yet. Reopen the terminal and try again."
  fi
}

install_runtime_wrapper() {
  local target_path="$1"
  local target_dir target_parent_dir tmp_file

  target_dir="$(dirname "$target_path")"
  target_parent_dir="$(dirname "$target_dir")"
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/zcode.install.XXXXXX")"

  log "Installing ZCode..."
  download_zcode_script "$tmp_file"
  chmod +x "$tmp_file"

  if [[ -d "$target_dir" ]]; then
    :
  elif [[ -w "$target_parent_dir" ]]; then
    mkdir -p "$target_dir"
  else
    die "Cannot write to $target_dir. Please specify a writable directory via --install-dir or run with appropriate permissions."
  fi

  if [[ -w "$target_dir" ]]; then
    install -m 755 "$tmp_file" "$target_path"
  else
    die "Cannot write to $target_dir. Please specify a writable directory via --install-dir or run with appropriate permissions."
  fi

  rm -f "$tmp_file"

  debug "zcode installed to: $target_path"
}

init_default_settings() {
  local target_path="$1"

  if (( SKIP_INIT == 1 )); then
    debug "Skipped settings initialization by flag"
    return 0
  fi

  log "Initializing settings..."
  "$target_path" init >/dev/null 2>/dev/null
  debug "Settings file: $SETTINGS_PATH"
}

has_ssh_credential() {
  local key

  if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
    return 0
  fi

  for key in id_ed25519 id_rsa id_ecdsa id_dsa identity; do
    if [[ -f "$HOME/.ssh/$key" ]]; then
      return 0
    fi
  done

  return 1
}

ensure_git_ssh_known_host() {
  local known_hosts_file="$HOME/.ssh/known_hosts"
  local host="${GIT_SSH_HOST##*@}"

  # check if host key already exists
  if ssh-keygen -F "$host" -f "$known_hosts_file" >/dev/null 2>&1; then
    debug "SSH host key already trusted: $host"
    return 0
  fi

  # ensure .ssh directory exists
  if [[ ! -d "$HOME/.ssh" ]]; then
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
  fi

  # Clean up any broken or hashed records for this host first
  ssh-keygen -R "$host" >/dev/null 2>&1 || true

  # scan and add host key (explicitly requesting ecdsa to prevent 'No ECDSA host key' errors)
  debug "Adding SSH host key for: $host"
  if ssh-keyscan -t rsa,ecdsa,ed25519 "$host" >> "$known_hosts_file" 2>/dev/null; then
    chmod 644 "$known_hosts_file" 2>/dev/null || true
    debug "SSH host key added successfully"
  else
    warn "Failed to add SSH host key for $host. Git operations may require manual confirmation."
  fi
}

remove_broken_git_url_rewrite() {
  local host="$1"
  local broken_key="url.\"git@${host}:\".insteadOf"

  git config --global --get-all "$broken_key" >/dev/null 2>&1 || return 0

  if git config --global --unset-all "$broken_key" >/dev/null 2>&1; then
    debug "Removed broken git URL rewrite left by an older installer: $host"
  else
    warn_step "Found a broken git URL rewrite for $host but could not remove it. Please clean up ~/.gitconfig manually."
  fi
}

ensure_git_url_rewrite() {
  local host="$1"
  local rewrite_key="url.git@${host}:.insteadOf"
  local rewrite_value="https://${host}/"

  if git config --global --get-all "$rewrite_key" 2>/dev/null | grep -Fxq "$rewrite_value"; then
    debug "Git URL rewrite already configured: $rewrite_value"
    return 0
  fi

  if git config --global --add "$rewrite_key" "$rewrite_value" >/dev/null 2>&1; then
    debug "Configured git URL rewrite: $rewrite_value -> git@${host}:"
  else
    warn_step "Failed to configure git URL rewrite for $host"
  fi
}

setup_internal_git_access() {
  local host="${GIT_SSH_HOST##*@}"

  if ! command -v git >/dev/null 2>&1; then
    warn_step "Git is not installed. Skipping global URL rewrite config."
    return 0
  fi

  remove_broken_git_url_rewrite "$host"

  if (( STATIC_MARKETPLACE == 1 )); then
    debug "Static marketplace mode. Skipping the git URL rewrite for $host."
    return 0
  fi

  if ! has_ssh_credential; then
    log "No SSH key found. Keeping https for $host."
    log "To switch to SSH later: git config --global --add 'url.git@${host}:.insteadOf' 'https://${host}/'"
    return 0
  fi

  ensure_git_ssh_known_host
  ensure_git_url_rewrite "$host"
}

marketplace_is_configured() {
  local claude_bin="$1"
  local output

  output="$(CLAUDE_CONFIG_DIR="$ZCODE_HOME" "$claude_bin" plugin marketplace list --json 2>/dev/null || printf '[]')"
  printf '%s' "$output" | grep -Fq "$MARKETPLACE_SOURCE"
}

is_plugin_installed() {
  local claude_bin="$1"
  local name="$2"

  CLAUDE_CONFIG_DIR="$ZCODE_HOME" "$claude_bin" plugin list --json 2>/dev/null | grep -Fq "\"$name\"" 2>/dev/null
}

install_default_plugins() {
  local claude_bin="$1"
  local plugin failed=0

  for plugin in $DEFAULT_PLUGINS; do
    if is_plugin_installed "$claude_bin" "$plugin"; then
      debug "Already installed: $plugin"
      continue
    fi

    if ! CLAUDE_CONFIG_DIR="$ZCODE_HOME" "$claude_bin" plugin install "$plugin" --scope "$PLUGIN_SCOPE" >/dev/null 2>&1; then
      warn_step "Failed to install plugin: $plugin"
      failed=1
    fi
  done

  return "$failed"
}

install_static_marketplace() {
  local claude_bin="$1"
  local tmp_archive tmp_extract old_backup

  tmp_archive="$(mktemp "${TMPDIR:-/tmp}/zcode.market.XXXXXX")"
  tmp_extract="$(mktemp -d "${TMPDIR:-/tmp}/zcode.market.d.XXXXXX")"

  debug "Downloading marketplace archive: $MARKETPLACE_ARCHIVE_URL"
  if ! download_file "$MARKETPLACE_ARCHIVE_URL" "$tmp_archive"; then
    warn "Failed to download the static marketplace archive: $MARKETPLACE_ARCHIVE_URL"
    rm -rf "$tmp_archive" "$tmp_extract"
    return 1
  fi

  if ! tar -xzf "$tmp_archive" -C "$tmp_extract" 2>/dev/null; then
    warn "Downloaded marketplace archive is not a valid tar.gz"
    rm -rf "$tmp_archive" "$tmp_extract"
    return 1
  fi
  rm -f "$tmp_archive"

  if [[ ! -f "$tmp_extract/.claude-plugin/marketplace.json" ]]; then
    warn "Marketplace archive is missing .claude-plugin/marketplace.json"
    rm -rf "$tmp_extract"
    return 1
  fi

  # Atomically swap the extracted tree into the stable local directory.
  mkdir -p "$(dirname "$LOCAL_MARKETPLACE_DIR")"
  old_backup="${LOCAL_MARKETPLACE_DIR}.old.$$"
  [[ -e "$LOCAL_MARKETPLACE_DIR" ]] && mv "$LOCAL_MARKETPLACE_DIR" "$old_backup"
  if ! mv "$tmp_extract" "$LOCAL_MARKETPLACE_DIR"; then
    warn "Failed to install marketplace into $LOCAL_MARKETPLACE_DIR"
    [[ -e "$old_backup" ]] && mv "$old_backup" "$LOCAL_MARKETPLACE_DIR"
    rm -rf "$tmp_extract"
    return 1
  fi
  rm -rf "$old_backup"
  debug "Marketplace installed at: $LOCAL_MARKETPLACE_DIR"
  return 0
}

finalize_static_marketplace_settings() {
  # Force zyb-plugins to a clean local "directory" source. zcode's settings
  # normalization only overwrites a *git* source, so a directory source
  # survives future `zcode` runs; this also stops Claude from trying to
  # git-update a plain directory (autoUpdate disabled).
  [[ -f "$SETTINGS_PATH" ]] || return 0
  LOCAL_DIR="$LOCAL_MARKETPLACE_DIR" NAME="$LOCAL_MARKETPLACE_NAME" \
    perl -MJSON::PP - "$SETTINGS_PATH" <<'PERL'
use strict; use warnings; use JSON::PP;
my ($file) = @ARGV;
open my $fh, '<', $file or exit 0;
local $/; my $raw = <$fh>; close $fh;
my $doc = eval { JSON::PP->new->utf8->decode($raw) };
exit 0 if $@ || ref($doc) ne 'HASH';
$doc->{extraKnownMarketplaces} = {} unless ref($doc->{extraKnownMarketplaces}) eq 'HASH';
$doc->{extraKnownMarketplaces}{$ENV{NAME}} = {
  autoUpdate => JSON::PP::false,
  source     => { source => 'directory', path => $ENV{LOCAL_DIR} },
};
open my $out, '>', $file or exit 0;
print $out JSON::PP->new->utf8->canonical->pretty->encode($doc);
close $out;
PERL
}

bootstrap_plugins_static() {
  local claude_bin="$1"

  log "Installing Plugins (static marketplace)..."

  if ! install_static_marketplace "$claude_bin"; then
    warn "Static marketplace install failed. You can retry later with: zcode market update"
    return 0
  fi

  # Register the local directory as the marketplace (overrides any git entry
  # written by `zcode init`, and populates known_marketplaces.json so plugin
  # installs resolve).
  if ! CLAUDE_CONFIG_DIR="$ZCODE_HOME" "$claude_bin" plugin marketplace add "$LOCAL_MARKETPLACE_DIR" --scope "$MARKETPLACE_SCOPE" >/dev/null 2>&1; then
    warn "Failed to register local marketplace directory: $LOCAL_MARKETPLACE_DIR"
    return 0
  fi

  finalize_static_marketplace_settings

  install_default_plugins "$claude_bin" || true
}

bootstrap_plugins() {
  local claude_bin="$1"

  [[ -n "$claude_bin" ]] || return 0

  if (( SKIP_MARKETPLACE == 1 )); then
    debug "Skipped marketplace/plugins installation by flag"
    return 0
  fi

  if (( STATIC_MARKETPLACE == 1 )); then
    bootstrap_plugins_static "$claude_bin"
    return 0
  fi

  log "Installing Plugins..."

  # ensure marketplace
  if marketplace_is_configured "$claude_bin"; then
    debug "Marketplace already configured: $MARKETPLACE_SOURCE"
    debug "Refreshing marketplace content..."
    local update_out
    if ! update_out=$(env CLAUDE_CONFIG_DIR="$ZCODE_HOME" GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" "$claude_bin" plugin marketplace update 2>&1); then
      debug "Marketplace update failed (non-fatal): $update_out"
    fi
  else
    debug "Adding marketplace: $MARKETPLACE_SOURCE"
    local out
    if ! out=$(env CLAUDE_CONFIG_DIR="$ZCODE_HOME" GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" "$claude_bin" plugin marketplace add "$MARKETPLACE_SOURCE" --scope "$MARKETPLACE_SCOPE" 2>&1); then
      warn "Failed to add the marketplace. You can retry later."
      warn "Marketplace error details: $out"
      return 0
    fi
  fi

  install_default_plugins "$claude_bin" || true
}

INF_MCP_NAME="inf-mcp"
INF_MCP_URL="https://inf-mcp.yukework.com/inf-mcp/mcp"

setup_mcp_servers() {
  local claude_bin="$1"

  [[ -n "$claude_bin" ]] || return 0

  if (( SKIP_MCP == 1 )); then
    debug "Skipped MCP server configuration by flag"
    return 0
  fi

  log "Configuring MCP servers..."

  if [[ -f "$SETTINGS_PATH" ]] && grep -Fq "\"$INF_MCP_NAME\"" "$SETTINGS_PATH" 2>/dev/null; then
    debug "MCP server already configured: $INF_MCP_NAME"
    return 0
  fi

  # ensure settings file exists with valid JSON
  mkdir -p "$(dirname "$SETTINGS_PATH")"
  if [[ ! -f "$SETTINGS_PATH" ]] || [[ ! -s "$SETTINGS_PATH" ]]; then
    printf '{}' > "$SETTINGS_PATH"
  fi

  # add MCP server entry via perl (available on macOS/Linux)
  local tmp_file
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/zcode.mcp.XXXXXX")" || {
    warn_step "Failed to configure MCP server: $INF_MCP_NAME"
    return 0
  }

  if perl -MJSON::PP -e '
    my $file = $ARGV[0];
    my $name = $ARGV[1];
    my $url  = $ARGV[2];

    local $/;
    open my $fh, "<", $file or die;
    my $raw = <$fh>;
    close $fh;

    my $json = JSON::PP->new->utf8->decode($raw);
    $json->{mcpServers} //= {};
    $json->{mcpServers}{$name} = {
      type => "streamable-http",
      url  => $url,
    };

    print JSON::PP->new->utf8->pretty->canonical->encode($json);
  ' "$SETTINGS_PATH" "$INF_MCP_NAME" "$INF_MCP_URL" > "$tmp_file" 2>/dev/null; then
    cat "$tmp_file" > "$SETTINGS_PATH"
    debug "MCP server added: $INF_MCP_NAME"
  else
    warn_step "Failed to configure MCP server: $INF_MCP_NAME"
  fi

  rm -f "$tmp_file"
}

read_installed_version() {
  local target_path="$1"
  local installed_version

  installed_version="$("$target_path" version 2>/dev/null || true)"
  printf '%s' "$installed_version"
}

print_next_steps() {
  local target_path="$1"
  local target_dir command_name installed_version

  target_dir="$(dirname "$target_path")"
  command_name="$(basename "$target_path")"
  installed_version="$(read_installed_version "$target_path")"

  printf '\n'
  if [[ -n "$installed_version" ]]; then
    done_ok "$(style_command "$command_name") v$installed_version is ready"
  else
    done_ok "$(style_command "$command_name") is ready"
  fi
  log "Settings file: $SETTINGS_PATH"

  if [[ ":$PATH:" != *":$target_dir:"* ]]; then
    local shell_rc="${HOME}/.profile"
    local shell_name="$(basename "${SHELL:-/bin/bash}")"

    if [[ "$shell_name" == "zsh" ]]; then
      shell_rc="${HOME}/.zshrc"
    elif [[ "$shell_name" == "bash" ]]; then
      if [[ -f "${HOME}/.bashrc" ]]; then
        shell_rc="${HOME}/.bashrc"
      elif [[ -f "${HOME}/.bash_profile" ]]; then
        shell_rc="${HOME}/.bash_profile"
      fi
    fi

    if [[ -w "$shell_rc" || -w "$(dirname "$shell_rc")" ]]; then
      if ! grep -q "$target_dir" "$shell_rc" 2>/dev/null; then
        printf '\n# Added by %s installation\nexport PATH="%s:$PATH"\n' "$command_name" "$target_dir" >> "$shell_rc"
        log "Added $target_dir to PATH in $shell_rc."

        # Try to automatically source the rc file where possible
        if [[ -f "$shell_rc" ]]; then
          source "$shell_rc" 2>/dev/null || true
        fi
      fi
    fi
  fi

  log "Start with $(style_command "$command_name"), or run $(style_command "$command_name login") to sign in"
}

parse_flags() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --debug)
        DEBUG=1
        shift
        ;;
      --skip-claude)
        SKIP_CLAUDE_INSTALL=1
        shift
        ;;
      --skip-init)
        SKIP_INIT=1
        shift
        ;;
      --skip-market)
        SKIP_MARKETPLACE=1
        shift
        ;;
      --static-market)
        STATIC_MARKETPLACE=1
        shift
        ;;
      --skip-mcp)
        SKIP_MCP=1
        shift
        ;;
      --install-dir)
        shift
        [[ -n "${1:-}" ]] || die "--install-dir requires a value"
        PREFERRED_INSTALL_DIR="$1"
        INSTALL_PATH=""
        shift
        ;;
      --install-path)
        shift
        [[ -n "${1:-}" ]] || die "--install-path requires a value"
        INSTALL_PATH="$1"
        shift
        ;;
      --zcode-url)
        shift
        [[ -n "${1:-}" ]] || die "--zcode-url requires a value"
        ZCODE_SCRIPT_URL="$1"
        shift
        ;;
      --claude-url)
        shift
        [[ -n "${1:-}" ]] || die "--claude-url requires a value"
        CLAUDE_INSTALL_URL="$1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

main() {
  print_zcode_logo
  local target_path claude_bin

  init_colors
  trap cleanup_internal_hosts_entries_on_exit EXIT
  parse_flags "$@"
  require_cmd bash
  require_cmd curl
  require_cmd install

  target_path="$(resolve_install_path)"
  debug "Install path: $target_path"

  install_claude_if_needed
  install_runtime_wrapper "$target_path"
  init_default_settings "$target_path"
  setup_internal_git_access
  claude_bin="$(resolve_claude_bin || true)"
  bootstrap_plugins "$claude_bin"
  setup_mcp_servers "$claude_bin"
  cleanup_internal_hosts_entries || true
  print_next_steps "$target_path"
}

main "$@"