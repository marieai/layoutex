# layoutex

### Requirements

Download and decompress the PubLayNet `labels.tar.gz` here: https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz

You can find all of the PubLayNet datasets here: https://developer.ibm.com/exchanges/data/all/publaynet/

Move the publaynet directory into `~/datasets/publaynet`, make a directory called `annotations` and copy the `val.json` into it.

### Running layoutex

To run, do the following:

```bash
# create the virtual environment
python3 -m venv layoutex_env

# activate it
source layoutex_env/bin/activate

# install requirements
pip install -e .
pip install -r requirements.txt

# run pytest
pytest -s tests/test_document_generator.py

# deactivate the venv when finished
deactivate
```