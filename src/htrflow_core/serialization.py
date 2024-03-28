from __future__ import annotations

import datetime
import json
import os
from typing import TYPE_CHECKING, Iterable, Sequence, Union

import xmlschema
from jinja2 import Environment, FileSystemLoader

import htrflow_core


if TYPE_CHECKING:
    from htrflow_core.volume import PageNode, RegionNode, Volume


_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "templates/schema")


class Serializer:
    """Serializer base class.

    Each output format is implemented as a subclass to this class.

    Attributes:
        extension: The file extension assigned with this format, for
            example ".txt" or ".xml"
        format_name: The name of this format, for example "alto"
    """

    extension: str
    format_name: str

    def serialize(self, page: PageNode) -> str:
        """Serialize page

        Arguments:
            page: Input page

        Returns:
            A string"""

    def serialize_volume(self, volume: Volume) -> Sequence[tuple[str, str]]:
        outputs = []
        for page in volume:
            doc = self.serialize(page)
            filename = os.path.join(volume.label, page.label + self.extension)
            outputs.append((doc, filename))
        return outputs

    def validate(self, doc: str):
        """Validate document"""


class AltoXML(Serializer):
    """Alto XML serializer"""

    extension = ".xml"
    format_name = "alto"

    def __init__(self):
        env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, "."]))
        self.template = env.get_template("alto")
        self.schema = os.path.join(_SCHEMA_DIR, "alto-4-4.xsd")

    def serialize(self, page: PageNode) -> str:
        if page.is_leaf():
            raise ValueError("Cannot serialize unsegmented page to Alto XML")

        # ALTO doesn't support nesting of regions ("TextBlock" elements)
        # This function is called from within the jinja template to tell
        # if a node corresponds to a TextBlock element, i.e. if its
        # children contains text and not other regions.
        def is_text_block(node):
            return bool(node.children) and all(child.is_line() for child in node.children)

        return self.template.render(
            page=page, metadata=metadata(page), labels=label_nodes(page), is_text_block=is_text_block
        )

    def validate(self, doc: str):
        xmlschema.validate(doc, self.schema)


class PageXML(Serializer):
    extension = ".xml"
    format_name = "page"

    def __init__(self):
        env = Environment(loader=FileSystemLoader([_TEMPLATES_DIR, "."]))
        self.template = env.get_template("page")
        self.schema = os.path.join(_SCHEMA_DIR, "pagecontent.xsd")

    def serialize(self, page: PageNode):
        if page.is_leaf():
            raise ValueError("Cannot serialize unsegmented page to Page XML")

        return self.template.render(
            page=page,
            metadata=metadata(page),
            labels=label_nodes(page),
        )

    def validate(self, doc: str):
        xmlschema.validate(doc, self.schema)


class Json(Serializer):
    """Simple JSON serializer"""

    extension = ".json"
    format_name = "json"

    def __init__(self, one_file=False):
        """Initialize JSON serializer

        Args:
            one_file: Export all pages of the volume to the same file.
                Defaults to False.
        """
        self.one_file = one_file

    def serialize(self, page: PageNode):
        def _serialize(obj):
            return {k: v for k, v in obj.__dict__.items() if k not in ["mask", "_image", "parent"]}
        return json.dumps(page.asdict(), default=_serialize, indent=4)

    def serialize_volume(self, volume: Volume):
        if self.one_file:
            filename = volume.label + self.extension
            doc = self.serialize(volume)
            return [(doc, filename)]
        return super().serialize_volume(volume)


class PlainText(Serializer):
    extension = ".txt"
    format_name = "txt"

    def serialize(self, page: PageNode) -> str:
        lines = page.traverse(lambda node: node.is_leaf())
        return "\n".join(line.text for line in lines)


def metadata(page: PageNode) -> dict[str, Union[str, list[dict[str, str]]]]:
    """Generate metadata for `page`

    Args:
        page: input page

    Returns:
        A dictionary with metadata
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    return {
        "creator": f"{htrflow_core.meta['Author']}",
        "software_name": f"{htrflow_core.meta['Name']}",
        "software_version": f"{htrflow_core.meta['Version']}",
        "application_description": f"{htrflow_core.meta['Summary']}",
        "created": timestamp,
        "last_change": timestamp,
        "processing_steps": [{"description": "", "settings": ""}],
    }


def supported_formats():
    """The supported formats"""
    return [cls.format_name for cls in Serializer.__subclasses__()]


def _get_serializer(format_name):
    for cls in Serializer.__subclasses__():
        if cls.format_name.lower() == format_name.lower():
            return cls()
    msg = f"Format '{format_name}' is not among the supported formats: {supported_formats()}"
    raise ValueError(msg)


def save_volume(volume: Volume, serializer: str | Serializer, dest: str) -> Iterable[tuple[str, str]]:
    """Serialize and save volume

    Arguments:
        volume: Input volume
        serializer: What serializer to use. Takes a Serializer instance
            or the name of the serializer as a string, see
            serialization.supported_formats() for supported formats.
        dest: Output directory
    """

    if isinstance(serializer, str):
        serializer = _get_serializer(serializer)

    for doc, filename in serializer.serialize_volume(volume):
        filename = os.path.join(dest, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(doc)


def label_nodes(node: PageNode | RegionNode, template="%s") -> dict[PageNode | RegionNode, str]:
    """Assign labels to node and its descendants

    Arguments:
        node: Start node
        template: Label template
    """
    labels = {}
    labels[node] = template % node.label
    for i, child in enumerate(node.children):
        labels |= label_nodes(child, f"{labels[node]}_%s{i}")
    return labels
