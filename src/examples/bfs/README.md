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
| `bfs_4` | targeted reattachment verification |

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
element-quality statistics. The three production candidates have 16,238,
62,672, and 241,230 triangles; the targeted `bfs_4` verification mesh has
392,715 triangles. All generated volume and facet files have also been read
successfully with the project's legacy FEniCS 2019.1 environment.

## Steady CFD and mesh convergence

`bfsflowsolver.py` follows the lid-driven-cavity example's split between
homogeneous perturbation boundary conditions and full-field steady boundary
conditions. The inlet is the exact peak-one Poiseuille profile, the outlet uses
the library's natural zero-traction condition, and all solid walls are no-slip
apart from the upper-wall actuator. Positive actuator input is blowing into the
domain and denotes volume flux per unit span; the finite-support Gaussian is
normalized so its integral is one.

Run the uncontrolled three-mesh study from the repository root with:

```bash
PYTHONPATH=src python src/examples/bfs/run_mesh_convergence.py \
  --output-root ../AdiabaticFlowControl/data/bfs
```

Add `--actuations -0.01 0.0 0.01` to verify the mesh over the intended control
range. The driver continues first in Reynolds number, then from the zero-input
state toward each nonzero input. It writes self-describing runs below
`mesh_convergence/<mesh_id>/<actuation>/`, including XDMF/HDF5 checkpoints,
raw velocity/pressure/mixed vectors, DOF coordinates and mappings, dense wall
shear, common-point samples, mass and nonlinear-residual diagnostics, and a
manifest. The campaign summary compares velocity, kinetic energy, and primary
reattachment location against the next finer mesh using the predeclared 0.5%
threshold.

`bfs_4` is not a globally finer production level. It retains the `bfs_3`
resolution away from the reattachment region, halves the lower-wall target
size over `7 <= x <= 15`, grades that refinement through the first `0.4h`
off the wall, and extends the fine shear-layer corridor down toward the wall.
Generate it alone with `--levels verification`; it is intended to decide
whether `bfs_3` satisfies the wall-shear tolerance without paying for uniform
refinement of the full `60h`-long domain.

The completed `bfs_3` to `bfs_4` comparison passes the `0.5%` criterion at
`a = -0.01, 0, 0.01`. The largest lower-wall-shear change is `0.106%`, and the
largest common-point velocity change is `0.021%`. Therefore `bfs_3` is the
selected production mesh and `bfs_4` is retained only as its verification
mesh. The complete numerical comparison is written to
`data/bfs/mesh_convergence/mesh_convergence_summary.json` in the research-data
repository.

After choosing a mesh, verify a checkpoint against the library's actual
perturbation time-stepper with:

```bash
PYTHONPATH=src python src/examples/bfs/check_steady_restart.py \
  --run-dir ../AdiabaticFlowControl/data/bfs/mesh_convergence/bfs_3/a_p0p00000
```

This writes `restart_invariance.json` beside the selected run and requires the
zero perturbation about the frozen equilibrium to remain below the configured
tolerance.
