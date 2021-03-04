# This script is based on the DirectX format presented in
# http://paulbourke.net/dataformats/directx/
# (but may not 100% fully implement everything).

from abc import abstractmethod, ABC
from typing import List, Union


class Parsable(ABC):
    def __init__(self): self._value = None

    @abstractmethod
    def parse(self, s: Union[str, List[str]]) -> int: pass

    @property
    def value(self): return self._value


class ParsableField(Parsable):
    """Wrapper around parsable for parsing separators"""
    def __init__(self, parsable, sep=';'):
        super().__init__()
        self._parsable = parsable
        self._sep = sep

    def parse(self, s: Union[str, List[str]]) -> int:
        if isinstance(self._parsable, (ParsableTemplate, ParsableArray)):
            # print('ParsableField::parse  template/array', self._parsable, s)
            num_chars = self._parsable.parse(s)
            # print('                      -> ', num_chars)
            if len(s) > num_chars and s[num_chars] != self._sep:
                raise ValueError()
            num_chars += len(self._sep)

        else:  # assume we are directly parsing a simple type
            # print('ParsableField::parse  simple', self._parsable, s)
            sep_index = s.find(self._sep)
            num_chars = self._parsable.parse(s[:sep_index]) + len(self._sep)
            # print('                      -> ', num_chars)

        return num_chars

    @property
    def value(self):
        return self._parsable.value


class ParsableInt(Parsable):
    def parse(self, s):
        assert isinstance(s, str), 'trying to parse int from "{}"'.format(s)
        self._value = int(s)
        return len(s)


class ParsableFloat(Parsable):
    def parse(self, s):
        assert isinstance(s, str), 'trying to parse float from "{}"'.format(s)
        self._value = float(s)
        return len(s)


class ParsableTemplate(Parsable):
    def __init__(self, fields: List[Parsable]):
        super().__init__()
        self._field_sep = ';'
        self._fields = [ParsableField(parsable, self._field_sep) for parsable in fields]

    def parse(self, s):
        if isinstance(s, List):
            s = ''.join(s[:])
        # print()
        # print('ParsableTemplate::parse')

        total_chars = 0
        sub_s = s
        for field in self._fields:
            # print('  before parsing', type(field), total_chars, sub_s)
            num_chars = field.parse(sub_s)
            # print('   after parsing', num_chars)
            sub_s = sub_s[num_chars:]
            total_chars += num_chars

        # print('ParsableTemplate::parse complete')
        # print()

        return total_chars


class ParsableArray(Parsable):
    def __init__(self, type_, num_elements: Union[int, ParsableInt]):
        super().__init__()
        self._field_sep = ','
        self._fields = []  # created dynamically on the fly
        self._value_type = type_
        self.num_elements = num_elements

    def parse(self, s):
        if isinstance(s, List):
            s = ''.join(s[:])

        if isinstance(self.num_elements, ParsableInt):
            assert self.num_elements.value is not None, \
                'unable to create array: number of elements not set'
            self.num_elements = self.num_elements.value

        self._fields = [
            ParsableField(self._value_type(), self._field_sep)
            for _ in range(self.num_elements)]

        total_chars = 0
        sub_s = s
        for field in self._fields[:-1]:
            num_chars = field.parse(sub_s)
            sub_s = sub_s[num_chars:]
            total_chars += num_chars

        # need to parse the last field in a custom way
        field = self._fields[-1]
        try:
            # print('parsing last with ,')
            num_chars = field.parse(sub_s)
        except ValueError:
            # print('parsing last with ;')
            field._sep = ';'
            num_chars = field.parse(sub_s) - len(field._sep)
        total_chars += num_chars

        return total_chars

    @property
    def value(self):
        return [field.value for field in self._fields]


class ParsableVector(ParsableTemplate):
    def __init__(self):
        self.x = ParsableFloat()
        self.y = ParsableFloat()
        self.z = ParsableFloat()
        super().__init__(fields=[self.x, self.y, self.z])

    @property
    def value(self):
        return [self.x.value, self.y.value, self.z.value]


class ParsableMatrix4x4(ParsableTemplate):
    def __init__(self):
        self.matrix4x4 = ParsableArray(ParsableFloat, num_elements=16)
        super().__init__(fields=[self.matrix4x4])

    @property
    def value(self):
        return self.matrix4x4.value


class ParsableMeshFace(ParsableTemplate):
    def __init__(self):
        self.n_verts = ParsableInt()
        self.vertex_indexes = ParsableArray(ParsableInt, self.n_verts)
        super().__init__(fields=[self.n_verts, self.vertex_indexes])

    @property
    def value(self):
        return self.vertex_indexes


class ParsableMesh(ParsableTemplate):
    def __init__(self):
        self.num_verts = ParsableInt()
        self.vertices = ParsableArray(ParsableVector, num_elements=self.num_verts)
        self.num_faces = ParsableInt()
        self.faces = ParsableArray(ParsableMeshFace, num_elements=self.num_faces)
        super().__init__(fields=[
            self.num_verts,
            self.vertices,
            self.num_faces,
            self.faces,
        ])

    @property
    def value(self):
        return self.vertices.value, self.faces.value


class ParsableRVCalibration(ParsableTemplate):
    def __init__(self):
        self.rotation_matrix = ParsableMatrix4x4()
        self.focal_length = ParsableFloat()
        self.angles = ParsableArray(ParsableFloat, num_elements=3)
        self.translation = ParsableArray(ParsableFloat, num_elements=3)
        self.pixel_size_xy = ParsableArray(ParsableFloat, num_elements=2)
        self.center_xy = ParsableArray(ParsableFloat, num_elements=2)
        self.correction = ParsableArray(ParsableFloat, num_elements=6)
        super().__init__(fields=[
            self.rotation_matrix,
            self.focal_length,
            self.angles,
            self.translation,
            self.pixel_size_xy,
            self.center_xy,
            self.correction,
        ])
