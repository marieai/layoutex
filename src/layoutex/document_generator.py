class DocumentGenerator(object):
    """A object that represents a document generator"""

    def __init__(self, layout_provider, content_provider):
        self.layout_provider = layout_provider
        self.content_provider = content_provider

    def generate(self):
        """Generate the document"""

        pass
