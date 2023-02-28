"""
Document class
"""


class Document(object):
    """A composite object that represents a document"""

    def __init__(self, content, template):
        self.content = content
        self.template = template

    def render(self, resolution=300):
        """Render the document"""

        return self.template.render(self.content, resolution)
