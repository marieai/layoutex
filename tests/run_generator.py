from layoutex.document_generator import DocumentGenerator
from layoutex.layout_provider import LayoutProvider, get_layout_provider

import asyncio
from codetiming import Timer


async def main():
    layout_provider = get_layout_provider("fixed", 10, 100)
    generator = DocumentGenerator(
        layout_provider=layout_provider,
        target_size=1024 * 1,
        solidity=0.5,
        expected_components=["figure", "table"],
    )

    # Run the tasks
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        async for document in generator:
            print(document)

    if False:
        render_documents = generator.render_documents(100)
        async for document in render_documents:
            print(document)
            mask = document.mask
            image = document.image

    # for i in range(10):
    #     document = generator.render(i)
    #     print(document)
    #


# create main with asyncio
if __name__ == "__main__":

    asyncio.run(main())
