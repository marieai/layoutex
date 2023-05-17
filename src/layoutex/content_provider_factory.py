from layoutex.content_provider import (
    ContentProvider,
    TextContentProvider,
    TableContentProvider,
    FigureContentProvider,
    TitleContentProvider,
)


class ContentProviderFactory:
    """
    A static factory for producing subclasses of ContentProvider.
    """
    @staticmethod
    def get(content_type: str, assets_dir: str) -> ContentProvider:
        """Get a content provider"""
        if content_type in ["paragraph", "list"]:
            return TextContentProvider(assets_dir=assets_dir)

        if content_type in ["table"]:
            return TableContentProvider(assets_dir=assets_dir)

        if content_type == "figure":
            return FigureContentProvider(assets_dir=assets_dir)

        if content_type == "title":
            return TitleContentProvider(assets_dir=assets_dir)

        raise ValueError(f"Unknown content type {content_type}")