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

## Code Structure

Start with `laplskel/cli/run_laplskel.py` for CLI options and `laplskel/workflows.py`
for input loading, component processing, and output assembly. Algorithm modules
live in `laplskel/`:

| Work area               | Main files                                     | Responsibility                                                                                                                                     |
| ----------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thinning and refinement | `refinement.py`                              | Topology-preserving voxel deletion, graph extraction, smoothing, and path simplification.                                                          |
| Contraction             | `contraction.py`, `triangle_decimation.py` | Laplacian contraction, linear solvers, convergence, and default-workflow graph decimation.                                                         |
| Runtime efficiency      | `parallelisation.py`                         | Component cropping, worker scheduling, and per-component algorithm dispatch. Solver and thinning optimizations also touch their algorithm modules. |
| Alternating algorithm   | `alternating.py`                             | Experimental contraction/thinning loop and branch fitting; integrates both algorithm areas.                                                        |
| Shared foundations      | `graph.py`, `medial.py`, `objects.py`    | Sparse adjacency/Laplacians, medial guidance, and union-find.                                                                                      |
| Data utilities          | `utils.py`                                   | Component labeling, voxel reconstruction, and GraphML export.                                                                                      |

Thinning and contraction can be developed separately within their modules.
Coordinate changes to shared helpers, `alternating.py`, and parameter forwarding
through the CLI, workflow, and component dispatcher. Runtime work spans scheduling
and algorithm internals, so agree on ownership before editing those internals.
Preserve coordinate conventions, graph topology, return values, and numerical
defaults across these boundaries.

