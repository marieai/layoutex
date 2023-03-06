from layoutex.content_provider import ContentProvider
from layoutex.layout_provider import LayoutProvider


class DocumentGenerator(object):
    """A object that represents a document generator"""

    def __init__(
        self, layout_provider: LayoutProvider, content_provider: ContentProvider
    ):
        self.layout_provider = layout_provider
        self.content_provider = content_provider

    def generate(self):
        """Generate the document"""

        pass
