"""Tool implementations: BaseTool, FunctionTool, RemoteTool, MultipartContent.

Public import path: ``from pyagent.tools import FunctionTool``.
Implementation lives in ``pyagent.tools.base`` to avoid a module/package
name collision (Python can't have both ``tools.py`` and ``tools/``).

``Permission`` is a ``Literal["read", "mutate", "dangerous"]`` type alias.
``TextPart``/``ImagePart``/``ResourcePart``/``StructuredPart``/``PartSpec``
are the building blocks for ``MultipartContent``.
"""

from pyagent.tools.base import (
    BaseTool,
    FunctionTool,
    ImagePart,
    MultipartContent,
    PartSpec,
    Permission,
    RemoteTool,
    ResourcePart,
    StructuredPart,
    TextPart,
)

__all__ = [
    "BaseTool",
    "FunctionTool",
    "ImagePart",
    "MultipartContent",
    "PartSpec",
    "Permission",
    "RemoteTool",
    "ResourcePart",
    "StructuredPart",
    "TextPart",
]
