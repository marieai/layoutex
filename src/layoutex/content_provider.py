"""
class representing a content provider
"""
from typing import Union

from layoutex.content import (
    Content,
    TableContent,
    ImageContent,
    ParagraphContent,
    ListContent,
)


class ContentProvider(object):
    """A object that represents a content provider"""

    def __init__(self):
        pass

    def get_content(
        self,
    ) -> Union[TableContent, ImageContent, ParagraphContent, ListContent]:
        """Get content"""
        pass


class TextContentProvider(ContentProvider):
    """A object that represents a text content provider"""

    def __init__(self):
        super().__init__()
        self.text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "

    def get_content(self):
        """Get content"""
        return ParagraphContent(self.text)
