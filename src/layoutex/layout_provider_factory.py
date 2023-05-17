from layoutex.layout_provider import (
    GeneratedLayoutProvider,
    LayoutProvider,
    FixedLayoutProvider, 
    LayoutProviderConfig
)


class LayoutProviderFactory:
    """
    A static factory for producing subclasses of LayoutProvider.
    """
    @staticmethod
    def get(layout_provider_config: LayoutProviderConfig) -> LayoutProvider:
        if layout_provider_config.type == "fixed":
            return FixedLayoutProvider(layout_provider_config)
        
        if layout_provider_config.type == "generated":
            return GeneratedLayoutProvider(layout_provider_config)
        
        raise ValueError(f"Unknown layout provider: {layout_provider_config.type}")
    
    def get_basic(type_name: str, max_objects: int, max_length: int):
        config = LayoutProviderConfig(type_name, max_objects, max_length)
        return LayoutProviderFactory.get(config)