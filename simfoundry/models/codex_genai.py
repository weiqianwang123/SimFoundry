# SPDX-License-Identifier: Apache-2.0
"""A ``google.genai.Client`` stand-in that answers through the Codex CLI.

articulate-anything's agents (stage 9's joint actor/critic) call
``client.models.generate_content(model=..., contents=..., config=...)`` on a raw
genai client. With ``SIMFOUNDRY_VLM_BACKEND=codex`` they get this object instead:

- text parts become the prompt, and the config's system instruction is put first;
- image parts (PIL images or inline image bytes) are attached, and their place in
  the prompt is marked ``[image k]``;
- video parts (e.g. the critic's joint-motion renders) are attached as a few evenly
  spaced frames, since Codex takes images only.

Only ``response.text`` is provided, which is all the agents read.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

from simfoundry.models.codex_vlm import CodexVLM

VIDEO_FRAMES = int(os.environ.get("SIMFOUNDRY_CODEX_VIDEO_FRAMES", "6"))


@dataclass
class GenAIResponse:
    """The part of ``GenerateContentResponse`` the agents read."""

    text: str


class _Prompt:
    """Collects text and images from genai-style content, in order."""

    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.texts: list[str] = []
        self.image_paths: list[str] = []

    def _image(self, image: PILImage.Image) -> None:
        path = self.tmp_dir / f"image_{len(self.image_paths) + 1}.png"
        image.convert("RGB").save(path)
        self.image_paths.append(str(path))
        self.texts.append(f"[image {len(self.image_paths)}]")

    def _video(self, data: bytes, mime_type: str) -> None:
        import cv2  # pylint: disable=import-outside-toplevel

        suffix = ".mp4" if "mp4" in mime_type else ".video"
        path = self.tmp_dir / f"video_{len(self.image_paths)}{suffix}"
        path.write_bytes(data)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        picks = sorted({round(i * (n - 1) / max(VIDEO_FRAMES - 1, 1)) for i in range(VIDEO_FRAMES)})
        self.texts.append(f"[video: {len(picks)} evenly spaced frames follow]")
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                self._image(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

    def add(self, obj) -> None:
        """Add a string, image, genai Part/Content/Blob, or a list of those."""
        if obj is None:
            return
        if isinstance(obj, str):
            self.texts.append(obj)
        elif isinstance(obj, PILImage.Image):
            self._image(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self.add(item)
        elif getattr(obj, "parts", None) is not None:  # Content
            self.add(list(obj.parts))
        elif getattr(obj, "text", None):  # Part with text
            self.texts.append(obj.text)
        elif getattr(obj, "inline_data", None) is not None:  # Part with bytes
            self.add(obj.inline_data)
        elif getattr(obj, "data", None) is not None:  # Blob
            mime = getattr(obj, "mime_type", "") or ""
            if mime.startswith("video/"):
                self._video(obj.data, mime)
            else:
                from io import BytesIO  # pylint: disable=import-outside-toplevel

                self._image(PILImage.open(BytesIO(obj.data)))
        elif not hasattr(obj, "model_dump"):  # skip empty/unknown genai objects
            self.texts.append(str(obj))


class _Models:
    def __init__(self, vlm: CodexVLM) -> None:
        self.vlm = vlm

    def generate_content(self, model=None, contents=None, config=None):
        """``client.models.generate_content``; sampling settings are ignored."""
        del model  # the Codex model is fixed by the environment
        with tempfile.TemporaryDirectory(prefix="codex_genai_") as tmp:
            prompt = _Prompt(Path(tmp))
            system = getattr(config, "system_instruction", None)
            if system is not None:
                prompt.texts.append("<system_instruction>")
                prompt.add(system)
                prompt.texts.append("</system_instruction>")
            prompt.add(contents)
            result = self.vlm("\n\n".join(prompt.texts), image_paths=prompt.image_paths)
        return GenAIResponse(text=result.text)


class CodexGenAIClient:
    """What ``genai.Client(...)`` returns, for text generation through Codex."""

    def __init__(self, model_name: str) -> None:
        self.models = _Models(
            CodexVLM(
                gemini_model=model_name,
                image_output=False,
                image_shapes=set(),
                resolutions={},
            )
        )
