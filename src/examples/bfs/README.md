# Backward-facing-step meshes

`generate_meshes.py` defines and generates the initial mesh-convergence family
for the two-dimensional, expansion-ratio-two backward-facing-step benchmark.
The step height, peak inlet speed, and nominal Reynolds number are `h = 1`,
`U_inf = 1`, and `Re = 500`.

## Generate

From the FlowControl repository root, in an environment containing Gmsh and
meshio:

```bash
python src/examples/bfs/generate_meshes.py
```

Use `--levels medium fine` to regenerate selected levels, or `--output-dir` to
write elsewhere. The three standard levels scale all target sizes by `2`, `1`,
and `1/2` while keeping the domain and refinement regions fixed.

## Files

The numerical names follow the lid-cavity example convention:

| File stem | Resolution |
| --- | --- |
| `bfs_1` | coarse |
| `bfs_2` | medium |
| `bfs_3` | fine |

For each level, `data_input` contains:

- `bfs_<n>.msh`: Gmsh mesh with named physical groups;
- `bfs_<n>.xdmf` and `.h5`: triangle mesh read by `FlowSolver`;
- `bfs_<n>_facets.xdmf` and `.h5`: physical boundary markers stored under
  the attribute name `name_to_read`;
- `bfs_<n>_metadata.json`: geometry, target sizes, tags, and quality data.

`bfs_mesh_manifest.json` collects all level metadata. The current generated
family has 16,238, 62,672, and 241,230 triangles from coarse to fine.

## Physical tags

| Tag | Name | Expected length/area |
| ---: | --- | ---: |
| 1 | `fluid` | area 110 |
| 11 | `inlet` | 1 |
| 12 | `outlet` | 2 |
| 13 | `upper_wall` | 59.2 |
| 14 | `actuator` | 0.8 |
| 15 | `upstream_lower_wall` | 10 |
| 16 | `step_wall` | 1 |
| 17 | `downstream_lower_wall` | 50 |

The generator checks these measures, rejects zero-area triangles, and records
element-quality statistics. All generated volume and facet files have also been
read successfully with the project's legacy FEniCS 2019.1 environment.
