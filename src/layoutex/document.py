"""
Document class
"""


class Document(object):
    """A composite object that represents a document"""

    def __init__(self, task_id, image, mask, layout):
        self.task_id = task_id
        self.image = image
        self.mask = mask
        self.layout = layout

    def __repr__(self):
        return f"Document(image={self.image}, mask={self.mask}, layout={self.layout})"

    def is_valid(self):
        return self.image is not None and self.mask is not None
