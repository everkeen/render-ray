"""Parse OBJ files"""

import logging
import dataclasses
import typing


class MTLDataItem(typing.TypedDict):
    """TypedDict for MTL material data"""

    name: str
    diffuse: tuple[float, float, float]
    ambient: tuple[float, float, float]
    specular: tuple[float, float, float]
    shininess: float


@dataclasses.dataclass
class MTLData:
    """Class to hold MTL data temporarily"""

    materials: dict[str, MTLDataItem] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class OBJData:
    """Class to hold OBJ data temporarily"""

    vertices: list[tuple[float, float, float]] = dataclasses.field(default_factory=list)
    faces: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    face_uvs: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    face_normals: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    normals: list[tuple[float, float, float]] = dataclasses.field(default_factory=list)
    uv: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    obj_names: list[str] = dataclasses.field(default_factory=list)
    smooth_lighting: bool = False
    materials: MTLData = MTLData()


class MTLParser:
    """Class to parse MTL files"""

    def __init__(self, mtl_data: str = "") -> None:
        self.mtl_data = mtl_data

    def load_file(self, file_path: str) -> None:
        """Load MTL data from a file"""
        with open(file_path, "r") as file:
            self.mtl_data = file.read()

    def parse(self) -> dict[str, MTLDataItem]:
        """Parse the MTL data and return a dictionary of materials."""
        materials: dict[str, MTLDataItem] = {}
        current_material: MTLDataItem = {
            "name": "",
            "diffuse": (0.0, 0.0, 0.0),
            "ambient": (0.0, 0.0, 0.0),
            "specular": (0.0, 0.0, 0.0),
            "shininess": 0.0,
        }
        current_name: str = ""
        lines = self.mtl_data.splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            match parts[0]:
                case "#":
                    continue  # Comment line
                case "newmtl":
                    if current_name:
                        materials[current_name] = current_material
                    current_name = parts[1]
                    current_material = {
                        "name": current_name,
                        "diffuse": (0.0, 0.0, 0.0),
                        "ambient": (0.0, 0.0, 0.0),
                        "specular": (0.0, 0.0, 0.0),
                        "shininess": 0.0,
                    }
                case "Kd":
                    current_material["diffuse"] = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                case "Ka":
                    current_material["ambient"] = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                case "Ks":
                    current_material["specular"] = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                case "Ns":
                    current_material["shininess"] = float(parts[1])
                case _:
                    logging.log(logging.INFO, "Ignoring line in MTL: %s", line)
        if current_name:
            materials[current_name] = current_material
        return materials


class OBJParser:
    """Class to parse OBJ files"""

    obj_data: str
    file_path: typing.Optional[str]

    def __init__(self, obj_data: str = "") -> None:
        self.obj_data = obj_data
        self.file_path = None

    def load_file(self, file_path: str) -> None:
        """Load OBJ data from a file"""
        with open(file_path, "r") as file:
            self.obj_data = file.read()

    def parse(self) -> OBJData:
        """Parse the OBJ data and return a OBJData object."""
        obj_data = OBJData()
        lines = self.obj_data.splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            match parts[0]:
                case "#":
                    continue  # Comment line
                case "v":
                    vertex: tuple[float, float, float] = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                    obj_data.vertices.append(vertex)
                case "vn":
                    normal: tuple[float, float, float] = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                    obj_data.normals.append(normal)
                case "f":
                    is_quad = len(parts) == 5
                    face_indices = []
                    if is_quad:
                        for part in parts[1:5]:
                            idx = int(part.split("/")[0]) - 1
                            face_indices.append(idx)
                        # Triangulate quad into two triangles
                        obj_data.faces.append(
                            (face_indices[0], face_indices[1], face_indices[2])
                        )
                        obj_data.faces.append(
                            (face_indices[0], face_indices[2], face_indices[3])
                        )
                        obj_data.face_uvs.append(
                            (face_indices[0], face_indices[1], face_indices[2])
                        )
                        obj_data.face_uvs.append(
                            (face_indices[0], face_indices[2], face_indices[3])
                        )
                        obj_data.face_normals.append(
                            (face_indices[0], face_indices[1], face_indices[2])
                        )
                        obj_data.face_normals.append(
                            (face_indices[0], face_indices[2], face_indices[3])
                        )
                    else:
                        for part in parts[1:4]:
                            idx = int(part.split("/")[0]) - 1
                            face_indices.append(idx)
                        obj_data.faces.append(tuple(face_indices))
                        obj_data.face_uvs.append(tuple(face_indices))
                        obj_data.face_normals.append(tuple(face_indices))
                case "s":
                    is_smooth = parts[1] == "1" or parts[1].lower() == "on"
                    obj_data.smooth_lighting = is_smooth
                case "vt":
                    uv_coord: tuple[float, float] = (
                        float(parts[1]),
                        float(parts[2]),
                    )
                    obj_data.uv.append(uv_coord)
                case "o":
                    obj_name = parts[1]
                    obj_data.obj_names.append(obj_name)
                case _:
                    logging.log(logging.INFO, "Ignoring line in OBJ: %s", line)
        return obj_data
