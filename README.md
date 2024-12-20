# Docling-eval


[![arXiv](https://img.shields.io/badge/arXiv-2408.09869-b31b1b.svg)](https://arxiv.org/abs/2408.09869)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ds4sd.github.io/docling/)
[![PyPI version](https://img.shields.io/pypi/v/docling)](https://pypi.org/project/docling/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling)](https://pypi.org/project/docling/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/DS4SD/docling)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling/month)](https://pepy.tech/projects/docling)

Evaluate [Docling](https://github.com/DS4SD/docling) on various datasets.

## Features

Evaluate docling on various datasets. You can use the cli

```sh
docling-eval % poetry run evaluate --help
2024-12-20 10:51:57,593 - INFO - PyTorch version 2.5.1 available.

 Usage: evaluate [OPTIONS]

╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --task        -t      [create|evaluate|visualize]                                                                Evaluation task [default: None] [required]                                                                              │
│ *  --modality    -m      [end-to-end|layout|tableformer|codeformer]                                                 Evaluation modality [default: None] [required]                                                                          │
│ *  --benchmark   -b      [DPBench|OmniDcoBench|WordScape|PubLayNet|DocLayNet|Pub1M|PubTabNet|FinTabNet|WikiTabNet]  Benchmark name [default: None] [required]                                                                               │
│ *  --input-dir   -i      PATH                                                                                       Input directory [default: None] [required]                                                                              │
│ *  --output-dir  -o      PATH                                                                                       Output directory [default: None] [required]                                                                             │
│    --help                                                                                                           Show this message and exit.                                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## End to End examples

<details>
<summary>**Layout evaluation for DP-Bench**</summary>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m layout -b DPBench -i <location-of-dpbench> -o ./benchmarks/dpbench-layout
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b DPBench -i ./benchmarks/dpbench-layout -o ./benchmarks/dpbench-layout
```

| id |          label | MaP[0.5:0.95] |
| -- | -------------- | ------------- |
|  0 |    page_header |         0.151 |
|  1 |           text |         0.678 |
|  2 | section_header |         0.443 |
|  3 |       footnote |         0.221 |
|  4 |        picture |         0.761 |
|  5 |        caption |         0.458 |
|  6 |    page_footer |         0.344 |
|  7 | document_index |         0.755 |
|  8 |        formula |         0.066 |
|  9 |          table |         0.891 |
</details>

<details>
<summary>**Table evaluations for DP-Bench**</summary>

👉 Create the dataset,

```sh
poetry run evaluate -t create -m tableformer -b DPBench -i <location-of-dpbench> -o ./benchmarks/dpbench-tableformer
```

👉 Evaluate the dataset,

```sh
poetry run evaluate -t evaluate -m tableformer -b DPBench -i ./benchmarks/dpbench-tableformer -o ./benchmarks/dpbench-tableformer
```

👉 Visualise the dataset,

```sh
poetry run evaluate -t visualize -m tableformer -b DPBench -i ./benchmarks/dpbench-tableformer -o ./benchmarks/dpbench-tableformer
```

The final result can be visualised as,

![DPBench_TEDS](./docs/evaluations/evaluation_DPBench_tableformer.png)
</details>

## Contributing

Please read [Contributing to Docling](https://github.com/DS4SD/docling/blob/main/CONTRIBUTING.md) for details.

## License

The Docling codebase is under MIT license.
For individual model usage, please refer to the model licenses found in the original packages.

## IBM ❤️ Open Source AI

Docling-eval has been brought to you by IBM.
