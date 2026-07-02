"""Generate Kaggle notebooks for the Duck harness workflow.

Generated artifacts:

``wheels/``
    Internet-on utility notebook that downloads the vLLM wheelhouse into its
    output. Re-run when runtime package pins change.

``code/``
    Internet-on utility notebook that clones this repo and emits a TAAF source
    bundle from the current committed code plus the bundled benchmark pickles.
    Re-run after code changes.

``submission.ipynb``
    Internet-off scored notebook. It attaches the competition source, the code
    utility notebook, the wheels utility notebook, and the model dataset, then
    runs Tufa's TAAF benchmark runner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPETITION_SLUG = "arc-prize-2026-arc-agi-3"
SOURCE_MARKER = "taaf-kaggle-bundle.json"
COMPETITION_WHEELHOUSE = f"/kaggle/input/competitions/{COMPETITION_SLUG}/arc_agi_3_wheels"
SHARE_NOTEBOOK_TEMPLATE = HERE / "tufa-arc-agi-framework" / "src" / "taaf" / "kaggle" / "taaf_kaggle_run_share.ipynb"


@dataclass
class Config:
    kaggle_username: str = "mariogemoll"
    github_repo: str = "mariogemoll/arc-prize-2026-arc-agi-3"
    deploy_key_secret: str = "GITHUB_DEPLOY_KEY_ARC_PRIZE_2026"

    wheels_slug: str = "arc-prize-2026-arc-agi-3-wheels"
    code_slug: str = "arc-prize-2026-arc-agi-3-code"
    submission_slug: str = "arc-prize-2026-arc-agi-3"

    # Tufa's public share used this model dataset. Keep it attached separately
    # from the wheel/code utility kernels.
    model_dataset_ref: str = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"

    served_model_name: str = "vrfai/Qwen3.6-27B-FP8"
    vllm_port: int = 1234
    vllm_max_model_len: int = 65536
    analyzer_context_window: int = 32768
    tensor_parallel_size: int = 1

    # Match Tufa's share notebook defaults.
    benchmark_label: str = "arc-prize-2026-arc-agi-3"
    true_submission_n_passes: int = 1
    interactive_n_passes: int = 1

    @property
    def wheels_ref(self) -> str:
        return f"{self.kaggle_username}/{self.wheels_slug}"

    @property
    def code_ref(self) -> str:
        return f"{self.kaggle_username}/{self.code_slug}"

    @property
    def submission_ref(self) -> str:
        return f"{self.kaggle_username}/{self.submission_slug}"


def env_info_cell() -> str:
    return """\
import os, platform, subprocess, sys

print("python      :", sys.version)
print("platform    :", platform.platform())
try:
    import torch
    print("torch       :", torch.__version__)
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        print("gpu         :", torch.cuda.get_device_name(0))
except ImportError:
    print("torch       : not installed")

try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("nvidia-smi  :", r.stdout.strip())
except FileNotFoundError:
    pass
print("kaggle env  :", {k: v for k, v in os.environ.items() if k.startswith("KAGGLE")})"""


def build_wheels(cfg: Config) -> None:
    intro = """\
# ARC Prize 2026 wheelhouse

Downloads the runtime wheelhouse used by the Duck submission into
`/kaggle/working`. Run this notebook with internet enabled and a GPU image.
Downstream submissions install vLLM from this notebook output with
`--no-index --find-links`.
"""

    download = """\
import subprocess, sys
from pathlib import Path

dest = Path("/kaggle/working")

# These pins mirror the setup stamp in Tufa's public share bundle. If Kaggle's
# image changes underneath us, edit this list and re-run the wheels notebook.
requirements = dest / "requirements.lock"
requirements.write_text(
    "\\n".join([
        "vllm==0.19.0",
        "torch==2.10.0",
        "flashinfer-python==0.6.6",
        "transformers",
        "accelerate",
        "peft",
        "pillow",
        "numpy",
        "scipy",
        "matplotlib",
        "requests",
        "python-dotenv",
        "imageio",
        "imageio-ffmpeg",
    ]) + "\\n",
    encoding="utf-8",
)

subprocess.run(
    [
        sys.executable, "-m", "pip", "download",
        "--requirement", str(requirements),
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
        "-d", str(dest),
    ],
    check=True,
)

wheels = sorted(dest.glob("*.whl"))
print(f"{len(wheels)} wheels downloaded")
assert any(w.name.startswith("vllm-") for w in wheels), "no vLLM wheel found"
subprocess.run(["du", "-sh", str(dest)], check=True)"""

    nb = new_notebook(cells=[new_markdown_cell(intro), new_code_cell(env_info_cell()), new_code_cell(download)])
    write_notebook(nb, HERE / "wheels" / "wheels.ipynb")
    write_metadata(
        {
            "id": cfg.wheels_ref,
            "title": cfg.wheels_slug,
            "code_file": "wheels.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "competition_sources": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        HERE / "wheels" / "kernel-metadata.json",
    )


def setup_command(cfg: Config) -> str:
    # A single shell command, written as a heredoc inside JSON, matching TAAF's
    # setup hook contract.
    return f'''"$PYTHON" - <<'PYSETUP'
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WHEELHOUSE_REF = {cfg.wheels_ref!r}
MODEL_REF = {cfg.model_dataset_ref!r}
SERVED_MODEL_NAME = {cfg.served_model_name!r}
VLLM_HOST = "127.0.0.1"
VLLM_PORT = {cfg.vllm_port}
VLLM_BASE_URL = f"http://{{VLLM_HOST}}:{{VLLM_PORT}}/v1"
VLLM_MAX_MODEL_LEN = {cfg.vllm_max_model_len}
ANALYZER_CONTEXT_WINDOW = {cfg.analyzer_context_window}
VLLM_TENSOR_PARALLEL_SIZE = {cfg.tensor_parallel_size}
WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
SITE_PACKAGES = WORKING_DIR / "vllm-site-packages"
VLLM_SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
VLLM_SERVER_PID = WORKING_DIR / "vllm-openai-server.pid"
INSTALL_STAMP = SITE_PACKAGES / ".duck-harness-wheelhouse"


def input_paths() -> dict[str, Path]:
    raw = os.getenv("TAAF_KAGGLE_INPUT_PATHS", "").strip()
    data = json.loads(raw) if raw else {{}}
    return {{str(ref): Path(str(path)) for ref, path in data.items()}}


def resolve_ref(ref: str) -> Path:
    mapped = input_paths().get(ref)
    if mapped is not None:
        return mapped
    owner, slug = ref.split("/", 1)
    for candidate in (
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets") / owner / slug,
        Path("/kaggle/input/notebooks") / owner / slug,
        Path("/kaggle/usr/lib/notebooks") / owner / slug,
    ):
        if candidate.exists():
            return candidate
    return Path("/kaggle/input") / slug


WHEELHOUSE = resolve_ref(WHEELHOUSE_REF)
MODEL_PATH = resolve_ref(MODEL_REF)


def vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SITE_PACKAGES) if not existing else f"{{SITE_PACKAGES}}{{os.pathsep}}{{existing}}"
    env.update({{
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "VLLM_NO_USAGE_STATS": "1",
    }})
    return env


def cached_install_is_usable() -> bool:
    if not INSTALL_STAMP.exists():
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import vllm, torch; print(f'cached vLLM {{vllm.__version__}}, torch {{torch.__version__}}')"],
        env=vllm_env(),
        text=True,
    )
    return result.returncode == 0


def install_vllm_wheelhouse() -> None:
    requirements = WHEELHOUSE / "requirements.lock"
    if not requirements.exists():
        raise FileNotFoundError(f"Missing wheelhouse lock file: {{requirements}}")
    if cached_install_is_usable():
        print(f"Using cached vLLM target install at {{SITE_PACKAGES}}", flush=True)
        return
    shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-index",
            "--find-links", str(WHEELHOUSE),
            "--requirement", str(requirements),
            "--target", str(SITE_PACKAGES),
            "--upgrade",
            "--ignore-installed",
            "--only-binary", ":all:",
            "--no-compile",
            "--disable-pip-version-check",
            "--no-warn-conflicts",
        ],
        check=True,
    )
    INSTALL_STAMP.write_text("ok\\n", encoding="utf-8")


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={{"Content-Type": "application/json"}})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tail_server_log(lines: int = 80) -> str:
    if not VLLM_SERVER_LOG.exists():
        return ""
    return "\\n".join(VLLM_SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def wait_for_vllm_server(timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"{{VLLM_BASE_URL}}/models"
    while time.monotonic() < deadline:
        if VLLM_SERVER_PID.exists():
            try:
                os.kill(int(VLLM_SERVER_PID.read_text().strip()), 0)
            except OSError as exc:
                raise RuntimeError(f"vLLM server process is not alive: {{exc}}\\n{{tail_server_log()}}") from exc
        try:
            print("vLLM server ready:", request_json(url, timeout=5), flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for vLLM server at {{url}}.\\n{{tail_server_log()}}")


def start_vllm_server() -> None:
    install_vllm_wheelhouse()
    VLLM_SERVER_PID.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODEL_PATH),
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(VLLM_TENSOR_PARALLEL_SIZE),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--generation-config", "vllm",
        "--enable-prefix-caching",
        "--default-chat-template-kwargs", "{{\\"preserve_thinking\\": true}}",
        "--reasoning-parser", "qwen3",
        "--max-model-len", str(VLLM_MAX_MODEL_LEN),
    ]
    print("Starting vLLM OpenAI server:", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=vllm_env(), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    VLLM_SERVER_PID.write_text(str(process.pid), encoding="utf-8")
    wait_for_vllm_server()


def run_vllm_api_smoke_test() -> None:
    payload = {{
        "model": SERVED_MODEL_NAME,
        "messages": [{{"role": "user", "content": "Answer in one short sentence: what is 2 + 2?"}}],
        "temperature": 0.0,
        "max_tokens": 96,
        "chat_template_kwargs": {{"enable_thinking": False}},
    }}
    response = request_json(f"{{VLLM_BASE_URL}}/chat/completions", payload=payload, timeout=120)
    print("VLLM smoke test:", response["choices"][0]["message"].get("content", "").strip(), flush=True)


print(f"vLLM wheelhouse path: {{WHEELHOUSE}}", flush=True)
print(f"Qwen model path: {{MODEL_PATH}}", flush=True)
missing = [str(path) for path in (WHEELHOUSE, MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError("Missing attached path(s): " + ", ".join(missing))
start_vllm_server()
run_vllm_api_smoke_test()
setup_env_path = Path(os.environ["TAAF_KAGGLE_SETUP_ENV"])
existing_setup_env = json.loads(setup_env_path.read_text(encoding="utf-8")) if setup_env_path.exists() else {{}}
existing_setup_env.update({{
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "PYTHONPATH": str(SITE_PACKAGES) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    "LOCAL_ANALYZER_BASE_URL": VLLM_BASE_URL,
    "OPENAI_BASE_URL": VLLM_BASE_URL,
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str(ANALYZER_CONTEXT_WINDOW),
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}})
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding="utf-8")
PYSETUP'''


def teardown_command() -> str:
    return '''"$PYTHON" - <<'PYTEARDOWN'
import os
import shutil
import signal
import time
from pathlib import Path

WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
pid_path = WORKING_DIR / "vllm-openai-server.pid"
site_packages = WORKING_DIR / "vllm-site-packages"
if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        print("Stopping vLLM server", flush=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        print(f"Could not stop vLLM server cleanly: {exc!r}", flush=True)
    pid_path.unlink(missing_ok=True)
shutil.rmtree(site_packages, ignore_errors=True)
print(f"Removed temporary vLLM install at {site_packages}", flush=True)
PYTEARDOWN'''


def build_code(cfg: Config) -> None:
    intro = f"""\
# ARC Prize 2026 source bundle

Clones `{cfg.github_repo}`, snapshots `ARC3-Inference/` and
`tufa-arc-agi-framework/`, and emits the TAAF Kaggle source bundle consumed by
the submission notebook. Re-run after every code change.
"""

    clone_and_bundle = f"""\
import base64
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path

clone_url = "https://github.com/{cfg.github_repo}.git"
try:
    from kaggle_secrets import UserSecretsClient
    raw = UserSecretsClient().get_secret({cfg.deploy_key_secret!r})
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(exist_ok=True)
    key_path = ssh_dir / "id_ed25519"
    key_path.write_bytes(base64.b64decode(raw))
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    subprocess.run("ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null", shell=True, check=True)
    clone_url = "git@github.com:{cfg.github_repo}.git"
    print("Using SSH deploy key.", flush=True)
except Exception:
    print("No deploy key secret found; using HTTPS.", flush=True)

repo = Path("/kaggle/working/repo")
if repo.exists():
    shutil.rmtree(repo)
subprocess.run(["git", "clone", "--depth", "1", clone_url, str(repo)], check=True)

out = Path("/kaggle/working")
for name in ["src"]:
    shutil.rmtree(out / name, ignore_errors=True)
(out / "src").mkdir(parents=True, exist_ok=True)

for name in ["ARC3-Inference", "tufa-arc-agi-framework"]:
    shutil.copytree(repo / name, out / "src" / name, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))

for name in ["benchmark_initial.pkl", "deploy_target.pkl"]:
    shutil.copy(repo / "example-run" / name, out / name)

git_status = subprocess.run(
    ["git", "-C", str(repo), "log", "-1", "--oneline"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
(out / "git_status.txt").write_text(git_status + "\\n", encoding="utf-8")
(out / "preamble.txt").write_text(
    "benchmark.label : {cfg.benchmark_label}\\n"
    "benchmark.solver: duck-harness\\n"
    f"git status: {{git_status}}\\n",
    encoding="utf-8",
)
(out / "setup_commands.json").write_text(json.dumps([{setup_command(cfg)!r}], indent=2) + "\\n", encoding="utf-8")
(out / "teardown_commands.json").write_text(json.dumps([{teardown_command()!r}], indent=2) + "\\n", encoding="utf-8")
(out / "{SOURCE_MARKER}").write_text(
    json.dumps({{
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "benchmark_label": {cfg.benchmark_label!r},
        "notebook": "submission.ipynb",
    }}, indent=2) + "\\n",
    encoding="utf-8",
)
(out / "README.dataset.md").write_text("# Duck Harness Kaggle Source Bundle\\n", encoding="utf-8")
shutil.rmtree(repo)
print("Source bundle staged in /kaggle/working", flush=True)"""

    nb = new_notebook(cells=[new_markdown_cell(intro), new_code_cell(clone_and_bundle)])
    write_notebook(nb, HERE / "code" / "code.ipynb")
    write_metadata(
        {
            "id": cfg.code_ref,
            "title": cfg.code_slug,
            "code_file": "code.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "competition_sources": [],
            "dataset_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
        HERE / "code" / "kernel-metadata.json",
    )


def build_submission(cfg: Config) -> None:
    nb = render_share_notebook(cfg)
    write_notebook(nb, HERE / "submission.ipynb")
    write_metadata(
        {
            "id": cfg.submission_ref,
            "title": cfg.submission_slug,
            "code_file": "submission.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "competition_sources": [COMPETITION_SLUG],
            "dataset_sources": [cfg.model_dataset_ref],
            "kernel_sources": [cfg.wheels_ref, cfg.code_ref],
            "model_sources": [],
        },
        HERE / "kernel-metadata.json",
    )


def render_share_notebook(cfg: Config) -> dict:
    """Render Tufa's share notebook, with code-kernel source-bundle support."""
    nb = json.loads(SHARE_NOTEBOOK_TEMPLATE.read_text(encoding="utf-8"))
    replacements = {
        "__TAAF_KAGGLE_WORKING_DIR__": repr("/kaggle/working"),
        "__TAAF_DATASET_SOURCES__": repr([cfg.model_dataset_ref]),
        "__TAAF_KERNEL_SOURCES__": repr([cfg.wheels_ref, cfg.code_ref]),
        "__TAAF_DATASET_BUNDLE_MARKER__": repr(SOURCE_MARKER),
        "__TAAF_COMPETITION_WHEELHOUSE__": repr(COMPETITION_WHEELHOUSE),
    }
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if isinstance(source, str):
            for token, value in replacements.items():
                source = source.replace(token, value)
            cell["source"] = source

    # Tufa's share template expects the source bundle to be the first dataset
    # source. Our workflow publishes it as the code utility notebook output, so
    # keep the rest of their notebook and only widen bundle discovery/mapping.
    nb["cells"][6]["source"] = locate_bundle_cell(cfg)
    return nb


def locate_bundle_cell(cfg: Config) -> str:
    return f"""\
DATASET_SOURCES = [{cfg.model_dataset_ref!r}]
KERNEL_SOURCES = [{cfg.wheels_ref!r}, {cfg.code_ref!r}]
DATASET_BUNDLE_MARKER = {SOURCE_MARKER!r}
SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"


def _find_bundle_dir() -> Path:
    roots = [Path("/kaggle/input"), Path("/kaggle/input/notebooks"), Path("/kaggle/usr/lib/notebooks")]
    for root in roots:
        if not root.exists():
            continue
        for marker in root.rglob(DATASET_BUNDLE_MARKER):
            return marker.parent
    raise RuntimeError("Duck source bundle not found in attached Kaggle inputs.")


def _dataset_mount_candidates(ref: str) -> list[Path]:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _kernel_mount_candidates(ref: str) -> list[Path]:
    owner, slug = ref.split("/", 1)
    return [
        Path("/kaggle/input/notebooks") / owner / slug,
        Path("/kaggle/usr/lib/notebooks") / owner / slug,
        Path("/kaggle/input") / slug,
    ]


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((c for c in candidates if c.exists()), None)


BUNDLE_DIR = _find_bundle_dir()
print(f"taaf.kaggle: source bundle = {{BUNDLE_DIR}}")
kaggle_input_paths: dict[str, str] = {{}}
for ref in DATASET_SOURCES:
    candidates = _dataset_mount_candidates(ref)
    kaggle_input_paths[ref] = str(_first_existing(candidates) or candidates[0])
for ref in KERNEL_SOURCES:
    candidates = _kernel_mount_candidates(ref)
    resolved = BUNDLE_DIR if ref == {cfg.code_ref!r} else _first_existing(candidates)
    kaggle_input_paths[ref] = str(resolved or candidates[0])

setup_env = {{
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps(KERNEL_SOURCES),
}}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\\n")
print(f"taaf.kaggle: input paths = {{setup_env['TAAF_KAGGLE_INPUT_PATHS']}}")"""


def new_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def new_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def new_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(HERE)}")


def write_metadata(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(HERE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", nargs="?", choices=["all", "wheels", "code", "submission"], default="all")
    args = parser.parse_args()
    cfg = Config()
    if args.which in ("all", "wheels"):
        build_wheels(cfg)
    if args.which in ("all", "code"):
        build_code(cfg)
    if args.which in ("all", "submission"):
        build_submission(cfg)


if __name__ == "__main__":
    main()
