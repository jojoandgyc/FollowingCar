#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/.." && pwd)"

package="/Users/dylon/Downloads/Filez/WebTool/2.3.2/release/rknn-toolkit2-v2.3.2-2025-04-09.tgz"
python_tag="cp310"
wheel_arch="x86_64"
image_name="rk-car-rknn-toolkit2:2.3.2"
base_image="ubuntu:22.04"
platform="linux/amd64"
proxy=""
no_proxy="localhost,127.0.0.1,host.docker.internal,192.168.0.64"
stage_only=0

usage() {
    cat <<'EOF'
Usage: scripts/build_rknn_toolkit2_image.sh [options]

Build the RKNN-Toolkit2 development image from a local Rockchip .tgz or .zip.

Options:
  --package PATH       Toolkit package archive. Supports .tgz/.tar.gz/.zip.
  --python-tag TAG     Wheel Python tag, default: cp310.
  --wheel-arch ARCH    Wheel architecture fragment, default: x86_64.
  --image-name NAME    Docker image tag, default: rk-car-rknn-toolkit2:2.3.2.
  --base-image NAME    Docker base image, default: ubuntu:22.04.
  --platform PLATFORM  Docker build platform, default: linux/amd64.
  --proxy URL          Build HTTP/HTTPS proxy.
  --no-proxy LIST      Build NO_PROXY list.
  --stage-only         Extract the wheel but do not run docker build.
  -h, --help           Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --package)
            package="$2"
            shift 2
            ;;
        --package=*)
            package="${1#*=}"
            shift
            ;;
        --python-tag)
            python_tag="$2"
            shift 2
            ;;
        --python-tag=*)
            python_tag="${1#*=}"
            shift
            ;;
        --wheel-arch)
            wheel_arch="$2"
            shift 2
            ;;
        --wheel-arch=*)
            wheel_arch="${1#*=}"
            shift
            ;;
        --image-name)
            image_name="$2"
            shift 2
            ;;
        --image-name=*)
            image_name="${1#*=}"
            shift
            ;;
        --base-image)
            base_image="$2"
            shift 2
            ;;
        --base-image=*)
            base_image="${1#*=}"
            shift
            ;;
        --platform)
            platform="$2"
            shift 2
            ;;
        --platform=*)
            platform="${1#*=}"
            shift
            ;;
        --proxy)
            proxy="$2"
            shift 2
            ;;
        --proxy=*)
            proxy="${1#*=}"
            shift
            ;;
        --no-proxy)
            no_proxy="$2"
            shift 2
            ;;
        --no-proxy=*)
            no_proxy="${1#*=}"
            shift
            ;;
        --stage-only)
            stage_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$package" ]]; then
    echo "RKNN Toolkit2 package not found: $package" >&2
    exit 1
fi

cache="${repo}/.cache/rknn-toolkit2"
wheel_stage="${cache}/packages"
mkdir -p "$wheel_stage"

wheel_rel="$(
    python3 - "$repo" "$package" "$wheel_stage" "$python_tag" "$wheel_arch" <<'PY'
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
package = Path(sys.argv[2]).resolve()
wheel_stage = Path(sys.argv[3]).resolve()
python_tag = sys.argv[4]
wheel_arch = sys.argv[5]

def wanted(name: str) -> bool:
    base = os.path.basename(name)
    if not base.endswith(".whl"):
        return False
    if not base.startswith("rknn_toolkit2-"):
        return False
    if f"-{python_tag}-" not in base:
        return False
    return not wheel_arch or wheel_arch in base

def choose(names):
    matches = sorted(name for name in names if wanted(name))
    if not matches:
        raise SystemExit(
            f"No RKNN Toolkit2 wheel found for python_tag={python_tag!r} "
            f"wheel_arch={wheel_arch!r} in {package}"
        )
    return matches[0]

wheel_stage.mkdir(parents=True, exist_ok=True)
if zipfile.is_zipfile(package):
    with zipfile.ZipFile(package) as archive:
        entry = choose(archive.namelist())
        out = wheel_stage / os.path.basename(entry)
        with archive.open(entry) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
elif tarfile.is_tarfile(package):
    with tarfile.open(package) as archive:
        entry = choose(archive.getnames())
        member = archive.getmember(entry)
        out = wheel_stage / os.path.basename(entry)
        src = archive.extractfile(member)
        if src is None:
            raise SystemExit(f"Could not extract archive member: {entry}")
        with src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
else:
    raise SystemExit(f"Unsupported package archive: {package}")

try:
    rel = out.resolve().relative_to(repo)
except ValueError:
    raise SystemExit(f"Staged wheel is outside repo: {out}")
print(rel.as_posix())
PY
)"

echo "Using RKNN wheel: ${repo}/${wheel_rel}"
if ((stage_only)); then
    echo "Stage only requested; skipping docker build."
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or not on PATH; wheel has been staged but image was not built." >&2
    exit 127
fi

docker_args=(
    build
    --platform "${platform}"
    -f "${repo}/docker/rknn-toolkit2/Dockerfile"
    --build-arg "BASE_IMAGE=${base_image}"
    --build-arg "RKNN_WHL=${wheel_rel}"
    -t "${image_name}"
)

if [[ -n "$proxy" ]]; then
    docker_args+=(
        --build-arg "HTTP_PROXY=${proxy}"
        --build-arg "HTTPS_PROXY=${proxy}"
        --build-arg "http_proxy=${proxy}"
        --build-arg "https_proxy=${proxy}"
        --build-arg "NO_PROXY=${no_proxy}"
        --build-arg "no_proxy=${no_proxy}"
    )
fi

docker_args+=("${repo}")

echo "Building Docker image: ${image_name}"
docker "${docker_args[@]}"
