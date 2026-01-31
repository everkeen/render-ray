"""Scene representation for Render Ray"""

import enum
import typing
import numpy as np
from . import obj_parse
from .utilitys.classproperty import classproperty

Vec2: typing.TypeAlias = np.ndarray
Vec3: typing.TypeAlias = np.ndarray
Vec4: typing.TypeAlias = np.ndarray
VEC_DTYPE = np.float32


def is_vector_valid(vec: np.ndarray, expected_dim: int) -> bool:
    """Check if a vector has the expected dimension."""
    if not isinstance(expected_dim, int) or expected_dim <= 0:
        raise ValueError("expected_dim must be a positive integer")
    if not isinstance(vec, np.ndarray):
        return False
    if vec.ndim != 1:
        return False
    if vec.shape[0] != expected_dim:
        return False
    if vec.dtype != VEC_DTYPE:
        return False
    return True


def object_method(
    strict: bool = True, fallback: typing.Callable | None = None
) -> typing.Callable:
    """Object method decorator.

    DO NOT use on methods that do not require a scene
    DO NOT use on methods from other classes
    """

    def wrapper(func: typing.Callable) -> typing.Callable:
        def inner(
            self: "Object", *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            if self.scene is None:
                if strict:
                    raise RuntimeError("Object is not part of a scene")
                if fallback is not None:
                    return fallback(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return inner

    return wrapper


class Quarternion:
    """Class representing a quarternion for 3D rotations."""

    w: float
    x: float
    y: float
    z: float

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Quarternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    @classmethod
    def from_euler(cls, rotation: Vec3) -> "Quarternion":
        """Create a quarternion from Euler angles (in radians, XYZ Format)."""
        cy = np.cos(rotation[2] * 0.5)
        sy = np.sin(rotation[2] * 0.5)
        cp = np.cos(rotation[1] * 0.5)
        sp = np.sin(rotation[1] * 0.5)
        cr = np.cos(rotation[0] * 0.5)
        sr = np.sin(rotation[0] * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def from_euler_degrees(cls, rotation: Vec3) -> "Quarternion":
        """Create a quarternion from Euler angles (in degrees, XYZ Format)."""
        radians = rotation * (np.pi / 180.0)
        return cls.from_euler(radians)

    @classproperty
    def identity(cls: type[typing.Self]) -> typing.Self:  # type: ignore[misc] # pylint: disable=E0213
        """Get the identity quarternion."""
        return cls(  # pylint: disable=E1102 # You CAN actually call cls, pylint
            1.0, 0.0, 0.0, 0.0
        )

    def to_array(self) -> np.ndarray:
        """Convert the quarternion to a numpy array."""
        return np.array([self.w, self.x, self.y, self.z], dtype=VEC_DTYPE)


class Transform:
    """Class representing a transformation in 3D space."""

    position: Vec3
    rotation: Quarternion
    scale: Vec3
    parent: "Transform | None"

    def __init__(
        self,
        position: Vec3 = np.array([0.0, 0.0, 0.0], dtype=VEC_DTYPE),
        rotation: Quarternion = Quarternion(1.0, 0.0, 0.0, 0.0),
        scale: Vec3 = np.array([1.0, 1.0, 1.0], dtype=VEC_DTYPE),
        parent: "Transform | None" = None,
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.parent = parent

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get the transformation matrix."""
        # Translation matrix
        translation_matrix = np.eye(4, dtype=VEC_DTYPE)
        translation_matrix[0, 3] = self.position[0]
        translation_matrix[1, 3] = self.position[1]
        translation_matrix[2, 3] = self.position[2]
        translation_matrix[3, 3] = 1.0
        # Rotation matrix from quarternion
        w, x, y, z = (
            self.rotation.w,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        )
        rotation_matrix = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w), 0],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2), 0],
                [0, 0, 0, 1],
            ],
            dtype=VEC_DTYPE,
        )
        # Scale matrix
        scale_matrix = np.eye(4, dtype=VEC_DTYPE)
        scale_matrix[0, 0] = self.scale[0]
        scale_matrix[1, 1] = self.scale[1]
        scale_matrix[2, 2] = self.scale[2]
        scale_matrix[3, 3] = 1.0
        # Combined transformation matrix: T * R * S
        translation_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        for ancestor in self.get_ancestors():  # Apply parent transforms
            translation_matrix = ancestor.transform_matrix @ translation_matrix
        return translation_matrix

    def get_ancestors(self) -> list["Transform"]:
        """Get a list of ancestor transforms."""
        ancestors: list["Transform"] = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors[::-1]  # Reverse to get from root to immediate parent

    def transform_point(self, point: Vec3) -> Vec3:
        """Transform a point using the transformation matrix."""
        if not is_vector_valid(point, 3):
            raise ValueError(
                f"Point must be a 3-dimensional vector of type {VEC_DTYPE}"
            )
        point_homogeneous = np.array(
            [point[0], point[1], point[2], 1.0], dtype=VEC_DTYPE
        )
        transformed_point = self.transform_matrix @ point_homogeneous
        return transformed_point[:3]

    def get_transform_matrix_around(self, point: Vec3) -> np.ndarray:
        """Get the transformation matrix around a specific point."""
        if not is_vector_valid(point, 3):
            raise ValueError(
                f"Point must be a 3-dimensional vector of type {VEC_DTYPE}"
            )
        # Translate to origin
        to_origin = np.eye(4, dtype=VEC_DTYPE)
        to_origin[0, 3] = -point[0]
        to_origin[1, 3] = -point[1]
        to_origin[2, 3] = -point[2]
        to_origin[3, 3] = 1.0
        # Translate back
        from_origin = np.eye(4, dtype=VEC_DTYPE)
        from_origin[0, 3] = point[0]
        from_origin[1, 3] = point[1]
        from_origin[2, 3] = point[2]
        from_origin[3, 3] = 1.0
        # Combined transformation matrix: T_back * T * T_to_origin
        combined_matrix = from_origin @ self.transform_matrix @ to_origin
        return combined_matrix


# Random sidetrack art idk
# O  _____  O <(YIPPEEE)      ˅____˅ <(I got downscaled)
#   | |_| |               --> /____\
#    \___/
# goofy lil guy


class Mesh:
    """Class representing a 3D mesh."""

    vertices: list[Vec3]
    faces: list[tuple[int, int, int]]
    normals: list[Vec3]

    def __init__(self) -> None:
        self.vertices = []
        self.faces = []
        self.normals = []

    def add_vertex(self, vertex: Vec3) -> None:
        """Add a vertex to the mesh."""
        if not is_vector_valid(vertex, 3):
            raise ValueError(
                f"Vertex must be a 3-dimensional vector of type {VEC_DTYPE}"
            )
        self.vertices.append(vertex)

    def add_face(self, v1: int, v2: int, v3: int) -> None:
        """Add a face to the mesh."""
        self.faces.append((v1, v2, v3))

    def add_normal(self, normal: Vec3) -> None:
        """Add a normal vector to the mesh."""
        if not is_vector_valid(normal, 3):
            raise ValueError(
                f"Normal must be a 3-dimensional vector of type {VEC_DTYPE}"
            )
        self.normals.append(normal)

    @classmethod
    def from_obj(
        cls, obj_data: str, parser: obj_parse.OBJParser | None = None
    ) -> "Mesh":
        """Import a mesh from an OBJ formatted string."""
        parser = parser or obj_parse.OBJParser(obj_data)
        parsed_data = parser.parse()
        return cls.from_obj_data(parsed_data)

    @classmethod
    def from_obj_file(
        cls, file_path: str, parser: obj_parse.OBJParser | None = None
    ) -> "Mesh":
        """Import a mesh from an OBJ file."""
        parser = parser or obj_parse.OBJParser()
        parser.load_file(file_path)
        parsed_data = parser.parse()
        return cls.from_obj_data(parsed_data)

    @classmethod
    def from_obj_data(cls, obj_data: obj_parse.OBJData) -> "Mesh":
        """Import a mesh from an OBJData object."""
        mesh = cls()
        for vertex in obj_data.vertices:
            mesh.add_vertex(np.array(vertex, dtype=VEC_DTYPE))
        for face in obj_data.faces:
            mesh.add_face(*face)
        for normal in obj_data.normals:
            mesh.add_normal(np.array(normal, dtype=VEC_DTYPE))
        return mesh

    def transformed(
        self,
        transform: Transform,
        auto_recalculate_normals: bool = False,
        camera_transform: Transform | None = None,
    ) -> "Mesh":
        """Return a new mesh that is the transformed version of this mesh.

        recalculate_normals should be called after this if needed.

        Args:
            transform (Transform): The transformation to apply.
            auto_recalculate_normals (bool):
            Whether to automatically recalculate normals after transformation.
            Default is False.

        Returns:
            Mesh: A new transformed mesh.
        """
        new_mesh = Mesh()
        for vertex in self.vertices:
            transformed_vertex = transform.transform_point(vertex)
            if camera_transform is not None:
                transformed_vertex = camera_transform.transform_point(
                    transformed_vertex
                )
            new_mesh.add_vertex(transformed_vertex)
        new_mesh.faces = self.faces.copy()
        new_mesh.normals = (
            self.normals.copy()
        )  # Note: Normals should be transformed properly
        if auto_recalculate_normals:
            new_mesh.recalculate_normals()
        return new_mesh

    def recalculate_normals(self) -> None:
        """Recalculate normals for the mesh based on the current vertices and faces."""
        self.normals = [
            np.array([0.0, 0.0, 0.0], dtype=VEC_DTYPE) for _ in self.vertices
        ]
        counts = [0 for _ in self.vertices]

        for face in self.faces:
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]

            edge1 = v2 - v1
            edge2 = v3 - v1
            face_normal = np.cross(edge1, edge2)
            face_normal /= (
                np.linalg.norm(face_normal) + 1e-8
            )  # Normalize and avoid division by zero

            for idx in face:
                self.normals[idx] += face_normal
                counts[idx] += 1

        for i, normal in enumerate(self.normals):
            if counts[i] > 0:
                self.normals[i] /= counts[i]
                self.normals[i] /= np.linalg.norm(normal) + 1e-8  # Normalize


class ObjectType(enum.Enum):
    """Enum for object types."""

    MESH = enum.auto()
    LIGHT = enum.auto()
    CAMERA = enum.auto()
    DEFAULT = enum.auto()
    SCRIPT = enum.auto()


class Object:
    """Class representing a 3D object in the scene."""

    name: str
    mesh: Mesh
    transform: Transform
    scene: "Scene | None"
    obj_type: ObjectType
    obj_type_args: dict[str, typing.Any]  # What to provide to the object type
    _parent: "Object | None"

    def __init__(
        self,
        name: str,
        mesh: Mesh,
        scene: "Scene | None" = None,
        parent: "Object | None" = None,
        transform: Transform | None = None,
        obj_type: ObjectType = ObjectType.DEFAULT,
        obj_type_args: dict[str, typing.Any] | None = None,
    ) -> None:
        self.name = name
        self.mesh = mesh
        self.scene = scene
        self._parent = parent
        self.transform = transform or Transform()
        self.obj_type = obj_type
        self.obj_type_args = obj_type_args or {}

    @property
    def parent(self) -> "Object | None":
        """Get the parent object."""
        return self._parent

    @property
    def transform_parent(self) -> "Transform | None":
        """Get the parent transform."""
        return self.transform.parent

    @parent.setter
    def parent(self, value: "Object | None") -> None:
        """Set the parent object."""
        self._parent = value
        self.transform.parent = value.transform if value is not None else None

    def __repr__(self) -> str:
        return (
            f"Object(name={self.name}, mesh=Mesh(vertices={len(self.mesh.vertices)}, "
            f"faces={len(self.mesh.faces)}))"
        )

    def set_scene(self, scene: "Scene | None") -> None:
        """Set the scene for the object."""
        self.scene = scene

    @object_method(strict=False, fallback=lambda self: f"Object {self.name} (no scene)")
    def __str__(self) -> str:
        return f"Object {self.name}"

    @object_method()
    def get_full_path(self) -> str:
        """Get the full path of the object in the scene hierarchy."""
        if self.scene is None:
            raise RuntimeError(
                "Object is not part of a scene (THIS SHOULD BE HANDLED BY OBJECT_METHOD DECORATOR)"
            )
        return self.scene.get_full_path(self)


class Scene:
    """Class representing a 3D scene."""

    objects: list[Object]
    current_camera: Object | None

    def __init__(self) -> None:
        self.objects = []
        self.current_camera = None  # Will default to first camera in the scene if None

    def add_object(self, obj: Object) -> None:
        """Add an object to the scene."""
        obj.set_scene(self)
        self.objects.append(obj)

    def remove_object(self, obj: Object) -> None:
        """Remove an object from the scene."""
        obj.set_scene(None)
        self.objects.remove(obj)

    def get_full_path(self, obj: Object) -> str:
        """Get the full path of an object in the scene."""
        if obj not in self.objects:
            raise ValueError("Object not in scene")
        path_parts = []
        current_obj: Object | None = obj
        while current_obj is not None:
            path_parts.append(current_obj.name)
            current_obj = current_obj.parent
        path_parts.reverse()
        return "/".join(path_parts)

    def __repr__(self) -> str:
        return f"Scene(objects={len(self.objects)})"

    def __str__(self) -> str:
        return f"Scene with {len(self.objects)} objects"

    def list_objects(self) -> list[str]:
        """List the names of all objects in the scene."""
        return [obj.name for obj in self.objects]

    def find_object_by_name(self, name: str) -> Object | None:
        """Find an object in the scene by its name."""
        return next((obj for obj in self.objects if obj.name == name), None)

    def clear(self) -> None:
        """Clear all objects from the scene."""
        for obj in self.objects:
            obj.set_scene(None)
        self.objects.clear()

    def find_object_by_type(self, obj_type: ObjectType) -> list[Object]:
        """Find all objects in the scene by their type."""
        return [obj for obj in self.objects if obj.obj_type == obj_type]

    def find_first_object_by_type(self, obj_type: ObjectType) -> Object | None:
        """Find the first object in the scene by its type."""
        return next((obj for obj in self.objects if obj.obj_type == obj_type), None)

    def find_camera(self) -> Object | None:
        """Find the current camera object in the scene."""
        return self.current_camera or self.find_first_object_by_type(ObjectType.CAMERA)

    def count_objects(self) -> int:
        """Count the number of objects in the scene."""
        return len(self.objects)

    def get_meshes(self) -> list[Mesh]:
        """Get a list of all meshes in the scene untransformed."""
        return [obj.mesh for obj in self.objects if obj.obj_type == ObjectType.DEFAULT]

    def get_vertices(self) -> list[Vec3]:
        """Get a list of all vertices in the scene."""
        vertices: list[Vec3] = []
        camera = self.find_camera()
        for obj in self.objects:
            if obj.obj_type != ObjectType.DEFAULT:
                continue
            if camera is not None:
                vertices.extend(
                    obj.mesh.transformed(
                        obj.transform, camera_transform=camera.transform
                    ).vertices
                )
            else:
                vertices.extend(obj.mesh.transformed(obj.transform).vertices)
        return vertices

    def get_faces(self) -> list[tuple[int, int, int]]:
        """Get a list of all faces in the scene."""
        faces: list[tuple[int, int, int]] = []
        offset: int = 0
        for obj in self.objects:
            if obj.obj_type != ObjectType.DEFAULT:
                continue
            faces.extend(
                [
                    (v1 + offset, v2 + offset, v3 + offset)
                    for (v1, v2, v3) in obj.mesh.faces
                ]
            )
            offset += len(obj.mesh.vertices)
        return faces
