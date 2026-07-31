
## Package structure

```text
laplskel/
├── __init__.py       # Package metadata and __version__
├── _version.py       # Versioneer-generated version discovery
├── cli.py            # CLI parser and entry point: get_parser(), main()
├── components.py     # Component labeling: label_and_sort_by_size()
├── contraction.py    # Contraction loop: laplacian_graph_contraction_edt()
├── graph.py          # Adjacency, Laplacian, and edge-collapse graph operations
├── image_io.py       # NIfTI loading/export, output paths, and rasterization
├── pipeline.py       # Public workflow: laplacian_skeletonisation()
├── runtime.py        # Parallel workers and temporary shared-volume lifecycle
└── solvers.py        # Linear solvers: solve_lu(), solve_cg()
```

## Example usage

```bash
laplskel \
  --input vessel_mask.nii.gz \
  --output results/vessel_skeleton \
  --use_edt \
  --use_anisotropic \
  --separate_streams \
  --n_jobs 8
```

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-0-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
