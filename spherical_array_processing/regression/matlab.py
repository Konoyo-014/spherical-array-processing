from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MatlabRuntime:
    executable: str
    source: str


@dataclass
class OctaveRuntime:
    executable: str
    source: str


@dataclass
class MatlabProbeResult:
    status: str  # ok | login_required | timeout | error | not_found
    message: str
    stdout_tail: str = ""
    stderr_tail: str = ""


def _search_standard_matlab_macos() -> MatlabRuntime | None:
    apps = Path("/Applications")
    if not apps.exists():
        return None
    cands = sorted(apps.glob("MATLAB_R*/bin/matlab"), reverse=True)
    for cand in cands:
        if cand.exists():
            return MatlabRuntime(executable=str(cand), source="macOS /Applications")
    return None


def detect_matlab() -> MatlabRuntime | None:
    env_bin = os.environ.get("MATLAB_BIN")
    if env_bin and Path(env_bin).exists():
        return MatlabRuntime(executable=env_bin, source="MATLAB_BIN")
    env_root = os.environ.get("MATLAB_ROOT")
    if env_root:
        cand = Path(env_root) / "bin" / "matlab"
        if cand.exists():
            return MatlabRuntime(executable=str(cand), source="MATLAB_ROOT")
    for cmd in ("matlab",):
        p = shutil.which(cmd)
        if p:
            return MatlabRuntime(executable=p, source="PATH")
    return _search_standard_matlab_macos()


def matlab_available() -> bool:
    return detect_matlab() is not None


def detect_octave() -> OctaveRuntime | None:
    env_bin = os.environ.get("OCTAVE_BIN")
    if env_bin and Path(env_bin).exists():
        return OctaveRuntime(executable=env_bin, source="OCTAVE_BIN")
    p = shutil.which("octave")
    if p:
        return OctaveRuntime(executable=p, source="PATH")
    return None


def run_matlab_batch(script: str, cwd: str | Path | None = None, timeout_s: int = 600) -> subprocess.CompletedProcess[str]:
    rt = detect_matlab()
    if rt is None:
        raise RuntimeError("MATLAB not found. Set MATLAB_BIN or MATLAB_ROOT.")
    cmd = [rt.executable, "-batch", script]
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout_s)


def probe_matlab_cli(timeout_s: int = 15) -> MatlabProbeResult:
    """Probe whether MATLAB CLI can execute commands without interactive login prompts.

    Uses a lightweight legacy invocation because `-batch` may hang silently when CLI login
    is required on some installations.
    """
    rt = detect_matlab()
    if rt is None:
        return MatlabProbeResult(status="not_found", message="MATLAB not found")
    cmd = [rt.executable, "-nojvm", "-nodisplay", "-nosplash", "-r", "disp('CODEX_MATLAB_PROBE_OK');exit"]
    try:
        cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return MatlabProbeResult(
            status="timeout",
            message=f"MATLAB CLI probe timed out after {timeout_s}s",
            stdout_tail=out[-1000:],
            stderr_tail=err[-1000:],
        )

    out = cp.stdout or ""
    err = cp.stderr or ""
    text = (out + "\n" + err).lower()
    login_markers = [
        "mathworks account",
        "email address",
        "登录失败",
        "电子邮件地址",
        "是否要重试",
    ]
    if any(m.lower() in text for m in login_markers):
        return MatlabProbeResult(
            status="login_required",
            message="MATLAB CLI appears to require interactive MathWorks account login",
            stdout_tail=out[-1000:],
            stderr_tail=err[-1000:],
        )
    if "codex_matlab_probe_ok" in text and cp.returncode == 0:
        return MatlabProbeResult(status="ok", message="MATLAB CLI probe succeeded", stdout_tail=out[-1000:], stderr_tail=err[-1000:])
    return MatlabProbeResult(
        status="error",
        message=f"MATLAB CLI probe failed with return code {cp.returncode}",
        stdout_tail=out[-1000:],
        stderr_tail=err[-1000:],
    )


def run_octave_eval(expr: str, cwd: str | Path | None = None, timeout_s: int = 600) -> subprocess.CompletedProcess[str]:
    rt = detect_octave()
    if rt is None:
        raise RuntimeError("Octave not found. Set OCTAVE_BIN or add octave to PATH.")
    cmd = [rt.executable, "--quiet", "--eval", expr]
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout_s)
