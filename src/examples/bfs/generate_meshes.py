"""Generate convergence meshes for the backward-facing-step benchmark.

The nondimensional geometry follows Blackburn, Barkley & Sherwin (2008),
shifted vertically to the convention used by Boujo & Gallaire (2015):

* inlet: ``x = -10``, ``1 <= y <= 2``;
* step edge: ``(x, y) = (0, 1)``;
* outlet: ``x = 35``, ``0 <= y <= 2``.

The upper wall is split around the Gaussian actuator at ``x = -1``.  The
nominal actuator segment extends four standard widths on either side of the
centre, so truncating the Gaussian at the segment endpoints is negligible.

The script writes, for each requested level:

* a Gmsh ``.msh`` file retaining named physical groups;
* a triangle-only XDMF/HDF5 mesh for legacy FEniCS;
* a line-only XDMF/HDF5 facet-marker file;
* a JSON file containing parameters and mesh-quality diagnostics.

Run from the FlowControl repository root, for example::

    python src/examples/bfs/generate_meshes.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import gmsh
import meshio
import numpy as np


PHYSICAL_TAGS = {
    "fluid": 1,
    "inlet": 11,
    "outlet": 12,
    "upper_wall": 13,
    "actuator": 14,
    "upstream_lower_wall": 15,
    "step_wall": 16,
    "downstream_lower_wall": 17,
}


@dataclass(frozen=True)
class MeshConfig:
    """Geometry and local target sizes for one convergence level."""

    level: str
    scale: float
    x_in: float = -10.0
    x_out: float = 35.0
    y_bottom: float = 0.0
    y_step: float = 1.0
    y_top: float = 2.0
    actuator_x: float = -1.0
    actuator_sigma: float = 0.1
    actuator_half_widths: float = 4.0
    h_bulk_reference: float = 0.4
    h_shear_reference: float = 0.08
    h_wall_reference: float = 0.02
    h_corner_reference: float = 0.01
    target_reattachment: bool = False
    reattachment_x_min: float = 7.0
    reattachment_x_max: float = 15.0
    reattachment_normal_extent: float = 0.4
    h_reattachment_wall: float = 0.005
    h_reattachment_core: float = 0.02

    @property
    def actuator_left(self) -> float:
        return self.actuator_x - self.actuator_half_widths * self.actuator_sigma

    @property
    def actuator_right(self) -> float:
        return self.actuator_x + self.actuator_half_widths * self.actuator_sigma

    @property
    def h_bulk(self) -> float:
        return self.scale * self.h_bulk_reference

    @property
    def h_shear(self) -> float:
        return self.scale * self.h_shear_reference

    @property
    def h_wall(self) -> float:
        return self.scale * self.h_wall_reference

    @property
    def h_corner(self) -> float:
        return self.scale * self.h_corner_reference


LEVELS = {
    "coarse": 2.0,
    "medium": 1.0,
    "fine": 0.5,
    "verification": 0.5,
}

MESH_STEMS = {
    "coarse": "bfs_1",
    "medium": "bfs_2",
    "fine": "bfs_3",
    "verification": "bfs_4",
}


def _add_physical_group(dim: int, entities: Iterable[int], name: str) -> None:
    tag = PHYSICAL_TAGS[name]
    gmsh.model.addPhysicalGroup(dim, list(entities), tag)
    gmsh.model.setPhysicalName(dim, tag, name)


def _build_geometry(config: MeshConfig) -> dict[str, list[int]]:
    """Build the BFS polygon and return its named curve groups."""

    geo = gmsh.model.geo
    point = {
        "inlet_lower": geo.addPoint(config.x_in, config.y_step, 0.0),
        "step_edge": geo.addPoint(0.0, config.y_step, 0.0),
        "step_bottom": geo.addPoint(0.0, config.y_bottom, 0.0),
        "reattachment_left": geo.addPoint(
            config.reattachment_x_min, config.y_bottom, 0.0
        ),
        "reattachment_right": geo.addPoint(
            config.reattachment_x_max, config.y_bottom, 0.0
        ),
        "outlet_lower": geo.addPoint(config.x_out, config.y_bottom, 0.0),
        "outlet_upper": geo.addPoint(config.x_out, config.y_top, 0.0),
        "actuator_right": geo.addPoint(config.actuator_right, config.y_top, 0.0),
        "actuator_left": geo.addPoint(config.actuator_left, config.y_top, 0.0),
        "inlet_upper": geo.addPoint(config.x_in, config.y_top, 0.0),
    }

    curve = {
        "upstream_lower_wall": geo.addLine(point["inlet_lower"], point["step_edge"]),
        "step_wall": geo.addLine(point["step_edge"], point["step_bottom"]),
        "downstream_lower_wall_left": geo.addLine(
            point["step_bottom"], point["reattachment_left"]
        ),
        "reattachment_lower_wall": geo.addLine(
            point["reattachment_left"], point["reattachment_right"]
        ),
        "downstream_lower_wall_right": geo.addLine(
            point["reattachment_right"], point["outlet_lower"]
        ),
        "outlet": geo.addLine(point["outlet_lower"], point["outlet_upper"]),
        "upper_wall_right": geo.addLine(point["outlet_upper"], point["actuator_right"]),
        "actuator": geo.addLine(point["actuator_right"], point["actuator_left"]),
        "upper_wall_left": geo.addLine(point["actuator_left"], point["inlet_upper"]),
        "inlet": geo.addLine(point["inlet_upper"], point["inlet_lower"]),
    }
    loop = geo.addCurveLoop(list(curve.values()))
    surface = geo.addPlaneSurface([loop])
    geo.synchronize()

    groups = {
        "fluid": [surface],
        "inlet": [curve["inlet"]],
        "outlet": [curve["outlet"]],
        "upper_wall": [curve["upper_wall_left"], curve["upper_wall_right"]],
        "actuator": [curve["actuator"]],
        "upstream_lower_wall": [curve["upstream_lower_wall"]],
        "step_wall": [curve["step_wall"]],
        "downstream_lower_wall": [
            curve["downstream_lower_wall_left"],
            curve["reattachment_lower_wall"],
            curve["downstream_lower_wall_right"],
        ],
    }
    _add_physical_group(2, groups["fluid"], "fluid")
    for name in PHYSICAL_TAGS:
        if name != "fluid":
            _add_physical_group(1, groups[name], name)

    groups["step_edge_point"] = [point["step_edge"]]
    groups["reattachment_lower_wall"] = [curve["reattachment_lower_wall"]]
    return groups


def _add_mesh_fields(config: MeshConfig, groups: dict[str, list[int]]) -> None:
    """Refine walls, the step corner, and the convectively active shear layer."""

    wall_curves = []
    for name in (
        "upper_wall",
        "actuator",
        "upstream_lower_wall",
        "step_wall",
        "downstream_lower_wall",
    ):
        wall_curves.extend(groups[name])

    distance_wall = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_wall, "CurvesList", wall_curves)
    gmsh.model.mesh.field.setNumber(distance_wall, "Sampling", 1000)

    wall = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(wall, "InField", distance_wall)
    gmsh.model.mesh.field.setNumber(wall, "SizeMin", config.h_wall)
    gmsh.model.mesh.field.setNumber(wall, "SizeMax", config.h_bulk)
    gmsh.model.mesh.field.setNumber(wall, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(wall, "DistMax", 0.35)

    distance_corner = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        distance_corner, "PointsList", groups["step_edge_point"]
    )

    corner = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(corner, "InField", distance_corner)
    gmsh.model.mesh.field.setNumber(corner, "SizeMin", config.h_corner)
    gmsh.model.mesh.field.setNumber(corner, "SizeMax", config.h_bulk)
    gmsh.model.mesh.field.setNumber(corner, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(corner, "DistMax", 2.0)

    shear = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(shear, "VIn", config.h_shear)
    gmsh.model.mesh.field.setNumber(shear, "VOut", config.h_bulk)
    gmsh.model.mesh.field.setNumber(shear, "XMin", -0.5)
    gmsh.model.mesh.field.setNumber(shear, "XMax", 35.0)
    gmsh.model.mesh.field.setNumber(shear, "YMin", 0.70)
    gmsh.model.mesh.field.setNumber(shear, "YMax", 1.30)
    gmsh.model.mesh.field.setNumber(shear, "Thickness", 0.30)

    fields = [wall, corner, shear]
    if config.target_reattachment:
        distance_reattachment = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            distance_reattachment,
            "CurvesList",
            groups["reattachment_lower_wall"],
        )
        gmsh.model.mesh.field.setNumber(
            distance_reattachment, "Sampling", 1000
        )

        reattachment_wall = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(
            reattachment_wall, "InField", distance_reattachment
        )
        gmsh.model.mesh.field.setNumber(
            reattachment_wall, "SizeMin", config.h_reattachment_wall
        )
        gmsh.model.mesh.field.setNumber(
            reattachment_wall, "SizeMax", config.h_shear
        )
        gmsh.model.mesh.field.setNumber(reattachment_wall, "DistMin", 0.02)
        gmsh.model.mesh.field.setNumber(
            reattachment_wall,
            "DistMax",
            config.reattachment_normal_extent,
        )

        descending_shear = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(
            descending_shear, "VIn", config.h_reattachment_core
        )
        gmsh.model.mesh.field.setNumber(
            descending_shear, "VOut", config.h_bulk
        )
        gmsh.model.mesh.field.setNumber(
            descending_shear, "XMin", config.reattachment_x_min - 2.0
        )
        gmsh.model.mesh.field.setNumber(
            descending_shear, "XMax", config.reattachment_x_max
        )
        gmsh.model.mesh.field.setNumber(
            descending_shear, "YMin", config.y_bottom
        )
        gmsh.model.mesh.field.setNumber(descending_shear, "YMax", 0.75)
        gmsh.model.mesh.field.setNumber(descending_shear, "Thickness", 0.30)
        fields.extend([reattachment_wall, descending_shear])

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)


def _extract_cells(mesh: meshio.Mesh, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    cells = mesh.cells_dict.get(cell_type)
    if cells is None:
        raise RuntimeError(f"Generated mesh has no {cell_type!r} cells")
    tags = mesh.cell_data_dict.get("gmsh:physical", {}).get(cell_type)
    if tags is None:
        raise RuntimeError(f"Generated {cell_type!r} cells have no physical tags")
    return cells, tags


def _mesh_statistics(
    points: np.ndarray,
    triangles: np.ndarray,
    lines: np.ndarray,
    line_tags: np.ndarray,
) -> dict:
    xy = points[:, :2]
    vertices = xy[triangles]
    side_1 = vertices[:, 1] - vertices[:, 0]
    side_2 = vertices[:, 2] - vertices[:, 0]
    twice_signed_area = side_1[:, 0] * side_2[:, 1] - side_1[:, 1] * side_2[:, 0]
    area = 0.5 * np.abs(twice_signed_area)

    edge_vectors = np.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 1],
            vertices[:, 0] - vertices[:, 2],
        ),
        axis=1,
    )
    edge_lengths = np.linalg.norm(edge_vectors, axis=2)
    quality = 4.0 * math.sqrt(3.0) * area / np.sum(edge_lengths**2, axis=1)

    boundary_lengths = {}
    line_lengths = np.linalg.norm(xy[lines[:, 1]] - xy[lines[:, 0]], axis=1)
    for name, tag in PHYSICAL_TAGS.items():
        if name == "fluid":
            continue
        boundary_lengths[name] = float(np.sum(line_lengths[line_tags == tag]))

    return {
        "points": int(len(points)),
        "triangles": int(len(triangles)),
        "boundary_edges": int(len(lines)),
        "domain_area": float(np.sum(area)),
        "triangle_area_min": float(np.min(area)),
        "triangle_area_max": float(np.max(area)),
        "edge_length_min": float(np.min(edge_lengths)),
        "edge_length_max": float(np.max(edge_lengths)),
        "quality_min": float(np.min(quality)),
        "quality_mean": float(np.mean(quality)),
        "quality_p01": float(np.quantile(quality, 0.01)),
        "boundary_lengths": boundary_lengths,
    }


def _validate_statistics(stats: dict) -> None:
    expected_lengths = {
        "inlet": 1.0,
        "outlet": 2.0,
        "upper_wall": 44.2,
        "actuator": 0.8,
        "upstream_lower_wall": 10.0,
        "step_wall": 1.0,
        "downstream_lower_wall": 35.0,
    }
    if not math.isclose(stats["domain_area"], 80.0, rel_tol=1e-10, abs_tol=1e-9):
        raise RuntimeError(f"Expected domain area 80, got {stats['domain_area']}")
    if stats["triangle_area_min"] <= 0.0:
        raise RuntimeError("Mesh contains a zero-area triangle")
    if stats["quality_min"] <= 0.25:
        raise RuntimeError(
            f"Minimum normalized triangle quality is too low: {stats['quality_min']}"
        )
    for name, expected in expected_lengths.items():
        actual = stats["boundary_lengths"][name]
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
            raise RuntimeError(
                f"Expected boundary {name!r} to have length {expected}, got {actual}"
            )


def _write_fenics_meshes(
    mesh: meshio.Mesh,
    stem: Path,
) -> dict:
    triangles, triangle_tags = _extract_cells(mesh, "triangle")
    lines, line_tags = _extract_cells(mesh, "line")
    points = mesh.points[:, :2]

    volume_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles)],
        cell_data={"name_to_read": [triangle_tags]},
    )
    facet_mesh = meshio.Mesh(
        points=points,
        cells=[("line", lines)],
        cell_data={"name_to_read": [line_tags]},
    )
    meshio.write(stem.with_suffix(".xdmf"), volume_mesh)
    meshio.write(stem.with_name(stem.name + "_facets").with_suffix(".xdmf"), facet_mesh)
    return _mesh_statistics(points, triangles, lines, line_tags)


def generate_one(config: MeshConfig, output_dir: Path) -> dict:
    """Generate, convert, validate, and summarize one mesh level."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / MESH_STEMS[config.level]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", config.h_corner)
        gmsh.option.setNumber("Mesh.MeshSizeMax", config.h_bulk)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.model.add(stem.name)
        groups = _build_geometry(config)
        _add_mesh_fields(config, groups)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Relocate2D")
        gmsh.write(str(stem.with_suffix(".msh")))
    finally:
        gmsh.finalize()

    mesh = meshio.read(stem.with_suffix(".msh"))
    stats = _write_fenics_meshes(mesh, stem)
    _validate_statistics(stats)
    summary = {
        "filename_stem": stem.name,
        "software": {
            "gmsh": gmsh.__version__,
            "meshio": meshio.__version__,
            "numpy": np.__version__,
        },
        "config": asdict(config),
        "derived_sizes": {
            "h_bulk": config.h_bulk,
            "h_shear": config.h_shear,
            "h_wall": config.h_wall,
            "h_corner": config.h_corner,
            "h_reattachment_wall": (
                config.h_reattachment_wall
                if config.target_reattachment
                else None
            ),
            "h_reattachment_core": (
                config.h_reattachment_core
                if config.target_reattachment
                else None
            ),
            "actuator_left": config.actuator_left,
            "actuator_right": config.actuator_right,
        },
        "physical_tags": PHYSICAL_TAGS,
        "statistics": stats,
    }
    stem.with_name(stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=tuple(LEVELS),
        default=list(LEVELS),
        help="Convergence levels to generate (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data_input",
        help="Output directory (default: the BFS example's data_input folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = args.output_dir / "bfs_mesh_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "description": "Backward-facing-step mesh-convergence family",
            "levels": {},
        }
    for level in args.levels:
        print(f"Generating {MESH_STEMS[level]} ({level})")
        config = MeshConfig(
            level=level,
            scale=LEVELS[level],
            target_reattachment=level == "verification",
        )
        manifest["levels"][level] = generate_one(config, args.output_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
