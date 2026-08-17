# Backward-facing-step meshes

`generate_meshes.py` defines and generates the initial mesh-convergence family
for the two-dimensional, expansion-ratio-two backward-facing-step benchmark.
The step height, peak inlet speed, and nominal Reynolds number are `h = 1`,
`U_inf = 1`, and `Re = 500`.

The domain extends from `x = -10` to `x = 35`. Blackburn, Barkley & Sherwin
(2008) found that reducing the downstream length from `50h` to `35h` changed
the two-dimensional transient growth at `t = 60` by only `0.02%` at `Re = 500`.

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
family has 13,314, 50,038, and 193,665 triangles from coarse to fine.

## Physical tags

| Tag | Name | Expected length/area |
| ---: | --- | ---: |
| 1 | `fluid` | area 80 |
| 11 | `inlet` | 1 |
| 12 | `outlet` | 2 |
| 13 | `upper_wall` | 44.2 |
| 14 | `actuator` | 0.8 |
| 15 | `upstream_lower_wall` | 10 |
| 16 | `step_wall` | 1 |
| 17 | `downstream_lower_wall` | 35 |

The generator checks these measures, rejects zero-area triangles, and records
element-quality statistics. The three production candidates have 13,314,
50,038, and 193,665 triangles; the targeted `bfs_4` verification mesh has
308,521 triangles. All generated volume and facet files have also been read
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
refinement of the full `45h`-long domain.

## Complete example

First compute the steady state at `Re = 500` by continuation:

```bash
PYTHONPATH=src python src/examples/bfs/compute_steady_state_increasing_Re.py
```

Then run the short unactuated time-domain example:

```bash
PYTHONPATH=src python src/examples/bfs/run_bfs_example.py
```

## Fixed-point branch and eigenvalues

Continue the local `Re=500` checkpoint from zero actuation toward both ends of
`[-0.01, 0.01]`:

```bash
PYTHONPATH=src python src/examples/bfs/compute_steady_state_increasing_actuation.py
```

The XDMF/HDF5 fixed-point checkpoints are written to
`data_output/bfs_3/fixed_points`, so this workflow is self-contained. To also
export raw fields, wall observations, and manifests to a separate research
repository, set `ADIABATIC_BFS_DATA_ROOT` or pass `--output-root`.

Export the operators for a fixed point in the FEniCS environment, then compute
eigenvalues in the complex SLEPc environment:

```bash
PYTHONPATH=src python src/examples/bfs/eig_compute_operators_bfs.py --actuation 0
PYTHONPATH=src python src/examples/bfs/eig_compute_bfs.py --actuation 0
```

The operator exporter loads the matching local fixed point automatically.
Without an external output root, operators and eigenvalues are written below
`data_output/bfs_3/spectral_validation`; with one, they are written below its
`spectral_validation` directory instead.

After exporting `A` and `E` at `a = -0.01, 0, 0.01`, compute the first
adiabatic response in the FEniCS environment with:

```bash
PYTHONPATH=src python src/examples/bfs/compute_adiabatic_conditioning.py
```

The script differentiates the existing 21-point fixed branch and solves
`A b1 = E x_star,a`. It records the mass norms, response ratios,
finite-difference check, solve residuals, and response vectors under the local
`data_output/bfs_3/conditioning` directory. If `ADIABATIC_BFS_DATA_ROOT` or
`--output-root` is supplied, the results instead go to
`critical_manifold/bfs_3/conditioning` in that research-data root.

## Disturbance-memory time

Measure the input-specific response tail at the two endpoints and center of
the fixed branch with:

```bash
PYTHONPATH=src python src/examples/bfs/compute_disturbance_memory.py
```

The driver reuses the four-mode, zero-net-flux inlet packet defined by the
transient-convergence study. The default pilot uses `dt = 0.01`, runs to
`t = 250`, and defines `T_m(a)` by the last downward crossing of 5% of the
peak perturbation energy. If a response has not crossed the threshold by the
end, the summary marks it as horizon-censored and reports only a lower bound.
Completed summaries also provide the frequencies corresponding to
`omega_a,max T_m = 0.05, 0.1, 0.2`.

The default production run advances 25,000 steps for each of three fixed
points and is therefore intended as an overnight campaign. Individual cases
can be run first with, for example, `--actuations 0`.

Use `--smoke` for a 20-step local check. Results go below the local
`data_output/bfs_3/disturbance_memory` directory, or below
`disturbance_memory/bfs_3` when an external research-data root is supplied.

On the former `x = 50` domain, the completed `bfs_3` to `bfs_4` comparison
passed the `0.5%` criterion at `a = -0.01, 0, 0.01`; its largest
lower-wall-shear and common-point velocity changes were `0.106%` and `0.021%`.
The shortened-domain meshes must be recomputed before those numerical results
are treated as current. The comparison is written to
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
