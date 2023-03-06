"""

"""
from enum import auto, Enum
from typing import Any


class ContentType(Enum):
    PARAGRAPH = auto()
    TITLE = auto()
    COMPOSITE = auto()
    TABLE = auto()
    FIGURE = auto()
    LIST = auto()
    LIST_ITEM = auto()


class Content:
    """
    Base class for all content types
    """

    def __init__(self):
        self.content_type = None
        self.iterable = True

        self.image = None
        self.mask = None

    def set_content_type(self, content_type):
        if type(content_type) != ContentType:
            raise TypeError(
                f"Invalid content type: {content_type}, valid types are {list(ContentType)}"
            )
        self.content_type = content_type

    def validate_content(self):
        """
        Validate content type
        Args:
            content: Content to validate

        Returns:
            None
        """
        NotImplementedError

    def __str__(self):
        return self._content.__str__()

    def __iter__(self):
        return self._content.__iter__()

    def __getitem__(self, key):
        return self._content.__getitem__(key)


class ParagraphContent(Content):
    """
    Paragraph content type
    """

    def __init__(self, content):
        super().__init__()
        self.set_content_type(ContentType.PARAGRAPH)
        self._content = content

    def validate_content(self):
        if type(self._content) != str:
            raise TypeError(
                f"Invalid content type for paragraph: {type(self._content)}"
            )


class TitleContent(Content):
    """
    Title content type
    """

    def __init__(self, content):
        super().__init__()
        self.set_content_type(ContentType.TITLE)
        self._content = content

    def validate_content(self):
        if type(self._content) != str:
            raise TypeError(f"Invalid content type for title: {type(self._content)}")


class CompositeContent(Content):
    def __init__(self, content_list, content_type_list):
        self.set_content_type(ContentType.COMPOSITE)
        self.validate_content(content_list)
        self.construct_content(content_list, content_type_list)
        self.iterable = True

    def validate_content(self, content_list):
        if not isinstance(content_list, list):
            raise TypeError(f"Expect a list of content, but got {type(content_list)}")

    def construct_content(self, content_list, content_type_list):
        self._content = []
        for content, content_type in zip(content_list, content_type_list):
            if content_type == ContentType.TITLE:
                self._content.append(TitleContent(content))
            elif content_type == ContentType.PARAGRAPH:
                self._content.append(ParagraphContent(content))
            else:
                raise NotImplementedError(f"{content_type} is not currently supported")

    def insert_content(self, new_content, index):
        NotImplementedError

    def delete_content(self, index):
        NotImplementedError

    def __repr__(self):
        return "CompositeContent(" + self._content.__repr__() + ")"

    def __str__(self) -> str:
        """get a string transparent of the nested object types"""
        transparent_str = "["
        for content in self._content:
            transparent_str += '"' + content.__str__() + '", '
        return transparent_str + "]"


class FigureContent(Content):
    """
    Image content type
    """

    def __init__(self, content):
        super().__init__()
        self.set_content_type(ContentType.FIGURE)
        self._content = content

    def validate_content(self):
        if type(self._content) != str:
            raise TypeError(f"Invalid content type for image: {type(self._content)}")


# TableContent, TextContent, ImageContent


class TableContent(Content):
    """
    Table content type
    """

    def __init__(self, content):
        super().__init__()
        self.set_content_type(ContentType.TABLE)
        self._content = content

    def validate_content(self):
        if type(self._content) != list:
            raise TypeError(f"Invalid content type for table: {type(self._content)}")


class ListContent(Content):
    """
    List content type
    """

    def __init__(self, content):
        super().__init__()
        self.set_content_type(ContentType.LIST)
        self._content = content

    def validate_content(self):
        if type(self._content) != list:
            raise TypeError(f"Invalid content type for list: {type(self._content)}")
