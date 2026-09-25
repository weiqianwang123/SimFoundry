# SPDX-License-Identifier: Apache-2.0
"""Codex CLI stand-in for the Gemini wrapper.

With ``SIMFOUNDRY_VLM_BACKEND=codex``, every ``Gemini(...)`` constructed by the
pipeline is replaced by a :class:`CodexVLM` that runs ``codex exec`` instead of
calling Vertex AI / the Gemini API. It keeps the parts of the Gemini interface
the stages use: calling it with ``prompt`` / ``image_paths``,
``get_result_text``, ``get_result_images`` and ``IMAGE_SHAPES``.

- Text models (detection, frame selection, front picking, physics estimates):
  one read-only ``codex exec`` call with the images attached; the final agent
  message is the response text.
- Image models (object removal, object upsampling): ``codex exec`` with the
  ``image_generation`` feature enabled, asked to save one PNG into a scratch
  workspace, which is returned as the response image.

Sampling controls (temperature, top_p, seed) have no Codex equivalent and are
ignored, so runs are not bit-reproducible. Responses are not cached.

Environment:
    SIMFOUNDRY_CODEX_BIN        codex executable (default: the ChatGPT desktop
                                app's bundled CLI if present, else ``codex``)
    SIMFOUNDRY_CODEX_MODEL      model (default: gpt-6-astra)
    SIMFOUNDRY_CODEX_REASONING  reasoning effort (default: medium)
    SIMFOUNDRY_CODEX_TIMEOUT_S  per-call timeout in seconds (default: 900)
    SIMFOUNDRY_CODEX_WORKDIR    parent dir for scratch workspaces (default: tmp)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image as PILImage

DESKTOP_APP_CODEX = "/usr/lib/chatgpt/resources/codex"

TEXT_TEMPLATE = """You are serving as a vision-language model behind a program's API.
Answer the request below using only the attached image(s) and the text.
Do not run shell commands, read or write files, or use any tool.
Your final message is passed verbatim to a parser, so follow the requested
output format exactly and add nothing before or after it.

<request>
{prompt}
</request>
"""

IMAGE_TEMPLATE = """You are serving as an image-editing model behind a program's API.
Use your image generation tool to produce exactly ONE image that fulfils the
request below, using the attached image(s) as the input to edit. Unless the
request says otherwise, keep the framing of the first attached image and output
it at {width}x{height} pixels. Save the result as a PNG file at exactly this
path:

    {output_path}

Then reply with the single word DONE.

<request>
{prompt}
</request>
"""


class CodexCallFailed(RuntimeError):
    """``codex exec`` did not produce a usable response."""


@dataclass
class CodexResult:
    """The response of one call."""

    text: str
    images: list = field(default_factory=list)


def default_codex_bin() -> str:
    """Resolve the codex executable."""
    explicit = os.environ.get("SIMFOUNDRY_CODEX_BIN")
    if explicit:
        return explicit
    if os.access(DESKTOP_APP_CODEX, os.X_OK):
        return DESKTOP_APP_CODEX
    found = shutil.which("codex")
    if found is None:
        raise FileNotFoundError("no codex executable; set SIMFOUNDRY_CODEX_BIN")
    return found


class CodexVLM:
    """Drop-in replacement for :class:`simfoundry.models.vlm.Gemini`."""

    def __init__(
        self,
        gemini_model,
        image_output,
        image_shapes,
        resolutions,
        codex_bin=None,
        model=None,
        reasoning_effort=None,
        timeout_s=None,
        verbose=False,
    ):
        """
        Args:
            gemini_model (str): The Gemini model this instance replaces (for logs).
            image_output (bool): Whether the replaced model returns images.
            image_shapes (set): The replaced model's IMAGE_SHAPES (stages read it).
            resolutions (dict): The replaced model's RESOLUTIONS.
        """
        self.gemini_model = gemini_model
        self.image_output = image_output
        self.IMAGE_SHAPES = image_shapes
        self.RESOLUTIONS = resolutions
        self.codex_bin = codex_bin or default_codex_bin()
        self.model = model or os.environ.get("SIMFOUNDRY_CODEX_MODEL", "gpt-6-astra")
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "SIMFOUNDRY_CODEX_REASONING", "medium"
        )
        self.timeout_s = float(
            timeout_s or os.environ.get("SIMFOUNDRY_CODEX_TIMEOUT_S", 900)
        )
        self.verbose = verbose

    def __call__(
        self,
        prompt,
        image_paths=None,
        temperature=0,
        top_p=0,
        seed=0,
        n_retries=3,
        print_results=False,
    ):
        """Same arguments as ``Gemini.__call__``; sampling controls are ignored."""
        del temperature, top_p, seed  # no Codex equivalent
        if image_paths is None:
            image_paths = []
        elif isinstance(image_paths, (str, os.PathLike)):
            image_paths = [image_paths]
        image_paths = [str(Path(p).resolve()) for p in image_paths]

        last_error = None
        for attempt in range(n_retries):
            print(
                f"Querying Codex [{self.model}, {self.reasoning_effort}] in place of "
                f"{self.gemini_model}: attempt {attempt + 1} of {n_retries}...",
                flush=True,
            )
            try:
                result = self._call_once(prompt, image_paths)
            except CodexCallFailed as exc:
                last_error = exc
                print(f"Codex call failed: {exc}", flush=True)
                time.sleep(min(30, 5 * (attempt + 1)))
                continue
            if print_results:
                print(result.text)
            return result
        raise CodexCallFailed(
            f"Codex [{self.model}] failed after {n_retries} attempts: {last_error}"
        )

    def _call_once(self, prompt, image_paths):
        parent = os.environ.get("SIMFOUNDRY_CODEX_WORKDIR")
        if parent:
            os.makedirs(parent, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="codex_vlm_", dir=parent))
        try:
            last_message = workdir / "last_message.txt"
            output_path = workdir / "output.png"
            if self.image_output:
                width, height = self._first_image_size(image_paths)
                full_prompt = IMAGE_TEMPLATE.format(
                    prompt=prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                )
                sandbox = "workspace-write"
            else:
                full_prompt = TEXT_TEMPLATE.format(prompt=prompt)
                sandbox = "read-only"

            cmd = [
                self.codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-C",
                str(workdir),
                "-s",
                sandbox,
                "-m",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
            ]
            if self.image_output:
                cmd += ["--enable", "image_generation"]
            cmd += [f"--image={p}" for p in image_paths]
            cmd += ["-o", str(last_message), "-"]
            try:
                proc = subprocess.run(
                    cmd,
                    input=full_prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexCallFailed(f"timed out after {self.timeout_s:.0f}s") from exc
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout)[-2000:]
                raise CodexCallFailed(f"exit code {proc.returncode}: {tail}")
            text = last_message.read_text() if last_message.exists() else ""
            if self.image_output:
                if not output_path.exists():
                    raise CodexCallFailed(f"no image written; last message: {text!r}")
                with PILImage.open(output_path) as img:
                    image = img.convert("RGB")
                return CodexResult(text=text, images=[image])
            if not text.strip():
                raise CodexCallFailed("empty response")
            return CodexResult(text=text.strip())
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _first_image_size(image_paths):
        if not image_paths:
            return 1024, 1024
        with PILImage.open(image_paths[0]) as img:
            return img.size

    def get_result_text(self, result):
        """The response text."""
        return result.text

    def get_result_images(self, result):
        """The response images (PIL)."""
        return list(result.images)
