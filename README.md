# torchdemon

[![PyPI](https://img.shields.io/pypi/v/torchdemon?style=flat-square)](https://pypi.python.org/pypi/torchdemon/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchdemon?style=flat-square)](https://pypi.python.org/pypi/torchdemon/)
[![PyPI - License](https://img.shields.io/pypi/l/torchdemon?style=flat-square)](https://pypi.python.org/pypi/torchdemon/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://jacknurminen.github.io/torchdemon](https://jacknurminen.github.io/torchdemon)

**Source Code**: [https://github.com/jacknurminen/torchdemon](https://github.com/jacknurminen/torchdemon)

**PyPI**: [https://pypi.org/project/torchdemon/](https://pypi.org/project/torchdemon/)

---

# Inference Server for RL

__Inference Server__. Serve model on GPU to workers. Workers communicate with the inference server over
multiprocessing Pipe connections.

__Dynamic Batching__. Accumulate batches from workers for forward passes. Set maximum batch size or maximum wait time
for releasing batch for inference.


## Installation

```sh
pip install torchdemon
```

## Usage

Define a model
```python
import torch

class Model(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

model = Model(8, 4)
```

Create an inference server for the model

```python
import torchdemon

inference_server = torchdemon.InferenceServer(
    model, batch_size=8, max_wait_ns=1000000, device=torch.device("cuda:0")
)
```

Create an inference client per agent and run in parallel processes
```python
import multiprocessing

processes = []
for _ in range(multiprocessing.cpu_count()):
    inference_client = inference_server.create_client()
    agent = Agent(inference_client)
    process = multiprocessing.Process(target=play, args=(agent,))
    process.start()
    processes.append(process)
```

Run server
```python
inference_server.run()

for process in processes:
    process.join()
```

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.7+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/jacknurminen/torchdemon/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/jacknurminen/torchdemon/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/jacknurminen/torchdemon/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
