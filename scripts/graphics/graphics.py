from copy import copy, deepcopy
import itertools
from sys import maxsize
import random

import numpy as np
import svgpathtools

from .utils import common as common_utils, parse
from .primitives import Bezier, Line, Primitive
from .utils.splitting import split_to_patches
from . import units


'''Arcs are ignored atm.'''

class VectorImage:
    def __init__(self, paths, view_size, origin=None, size=None, view_origin=None):
        self.paths = list(paths)

        if view_origin is not None:
            self.view_x, self.view_y = view_origin
        else:
            self.view_x, self.view_y = [units.Pixels(0), units.Pixels(0)]
        self.view_width, self.view_height = view_size

        if origin is not None:
            self.x, self.y = origin
        if size is not None:
            self.width, self.height = size


    @classmethod
    def from_svg(cls, file):
        # read svg
        paths, path_attribute_dicts, svg_attributes = svgpathtools.svg2paths2(file)

        # get canvas sizes
        x = y = view_x = view_y = width = height = view_width = view_height = None

        if 'x' in svg_attributes or 'y' in svg_attributes:
            origin = [units.Pixels(0), units.Pixels(0)]
            if 'x' in svg_attributes:
                origin[0] = units.fromrepr(svg_attributes['x'], units.Pixels)
            if 'y' in svg_attributes:
                origin[1] = units.fromrepr(svg_attributes['y'], units.Pixels)
        else:
            origin = None

        if 'width' in svg_attributes or 'height' in svg_attributes:
            size = [units.Pixels(0), units.Pixels(0)]
            if 'width' in svg_attributes:
                size[0] = units.fromrepr(svg_attributes['width'], units.Pixels)
            if 'height' in svg_attributes:
                size[1] = units.fromrepr(svg_attributes['height'], units.Pixels)
        else:
            size = None
        
        if 'viewBox' in svg_attributes:
            view_x, view_y, view_width, view_height = (units.fromrepr(coord, units.Pixels) for coord in svg_attributes['viewBox'].split())
            view_origin = (view_x, view_y)
            view_size = (view_width, view_height)
        else:
            view_origin = [units.Pixels(0), units.Pixels(0)]
            view_size = size.copy()
            
        # create image
        vector_image = cls(
                filter(None, (Path.create_visible(path, attributes) for (path, attributes) in zip(paths, path_attribute_dicts))),
                origin=origin, size=size, view_origin=view_origin, view_size=view_size)

        return vector_image


    def adjust_view(self, margin=.01):
        self.translate((0,0), adjust_view=True, margin=margin)


    def copy(self):
        return deepcopy(self)


    def leave_only_contours(self, default_width):
        [path.remove_fill(default_width) for path in self.paths]


    def mirror(self, *args, **kwargs):
        x = self.view_x.value + self.view_width.value / 2
        [path.mirror(x) for path in self.paths]


    def mirrored(self, *args, **kwargs):
        newim = self.copy()
        newim.mirror(*args, **kwargs)
        return newim


    def remove_filled(self):
        self.paths = [path for path in self.paths if not path.fill]


    def render(self, renderer):
        if (self.view_x == 0) and (self.view_y == 0):
            paths = self.paths
        else:
            paths = [path.translated((-self.view_x.as_pixels().value, -self.view_y.as_pixels().value)) for path in self.paths]
        return renderer(paths, (round(self.view_width.as_pixels()), round(self.view_height.as_pixels())))


    def rotate(self, rotation_deg, origin=None, adjust_view=False, margin=.01):
        if origin is None:
            origin = self.view_x.value + self.view_width.value / 2, self.view_y.value + self.view_height.value / 2
        [path.rotate(rotation_deg, origin) for path in self.paths]

        if adjust_view:
            self.adjust_view(margin=margin)


    def rotated(self, *args, **kwargs):
        newim = self.copy()
        newim.rotate(*args, **kwargs)
        return newim


    def save(self, file, **kwargs):
        if file[-3:].lower() == 'svg':
            return self._save2svg(file, **kwargs)
        else:
            raise NotImplementedError('Unknown file format {}'.format(file))


    def _save2svg(self, file, **kwargs):
        svg_attributes = {}
        if hasattr(self, 'x'):
            svg_attributes['x'] = str(self.x)
            svg_attributes['y'] = str(self.y)
        if hasattr(self, 'width'):
            svg_attributes['width'] = str(self.width)
            svg_attributes['height'] = str(self.height)
        svg_attributes['viewBox'] = '{} {} {} {}'.format(self.view_x, self.view_y, self.view_width, self.view_height)

        paths, attributes = zip(*(path.to_svgpathtools() for path in self.paths))
        return svgpathtools.wsvg(paths, filename=file, attributes=attributes, svg_attributes=svg_attributes)


    def scale(self, scale):
        self.view_x *= scale
        self.view_y *= scale
        self.view_width *= scale
        self.view_height *= scale

        [path.scale(scale) for path in self.paths]

    
    def scale_to_width(self, mode, new_value=None):
        widths = (path.width for path in self.paths if path.width is not None)

        if mode == 'max':
            max_width = max(widths)
            scale = new_value / max_width
        elif mode == 'min':
            min_width = min(widths)
            scale = new_value / min_width
        self.scale(scale)


    def simplify_segments(self, distinguishability_threshold):
        self.paths = list(filter(lambda path: path.nonempty, (path.with_simplified_segments(distinguishability_threshold) for path in self.paths)))


    def split_to_patches(self, patch_size, workers=0, paths_per_worker=10):
        origin = (float(self.view_x), float(self.view_y))

        patch_w, patch_h = patch_size
        patches_row_n = int(np.ceil(self.view_height.as_pixels() / patch_h))
        patches_col_n = int(np.ceil(self.view_width.as_pixels() / patch_w))
        patches_n = (patches_row_n, patches_col_n)

        vector_patch_origin_xs = np.array([patch_w * j for j in range(patches_col_n)])
        vector_patch_origin_ys = np.array([patch_h * i for i in range(patches_row_n)])
        vector_patch_origins = np.stack((vector_patch_origin_xs[None].repeat(patches_row_n, axis=0), vector_patch_origin_ys[:, None].repeat(patches_col_n, axis=1)), axis=-1)

        patch_size_pixels = units.Pixels(patch_w), units.Pixels(patch_h)
        vector_patches = np.array([[self.__class__([], origin=(units.Pixels(coord) for coord in vector_patch_origins[i,j]), size=patch_size_pixels, view_size=patch_size_pixels) for j in range(patches_col_n)] for i in range(patches_row_n)])

        split_path_to_patches = lambda path: path.split_to_patches(origin=origin, patch_size=patch_size, patches_n=patches_n)
        def distribute_path_in_patches(iS, jS, paths):
            for idx in range(len(iS)):
                i = iS[idx]
                j = jS[idx]
                path_in_patch = paths[idx]
                vector_patches[i, j].paths.append(path_in_patch.translated(-vector_patch_origins[i, j]))

        if isinstance(workers, int) and workers == 0:
            for path in self.paths:
                distribute_path_in_patches(*split_path_to_patches(path))
        else:
            if isinstance(workers, int):
                from pathos.multiprocessing import cpu_count, ProcessingPool as Pool
                if workers == -1:
                    batches_n = int(np.ceil(len(self.paths) / paths_per_worker))
                    optimal_workers = cpu_count() - 1
                    workers = min(optimal_workers, batches_n)
                workers = Pool(workers)
                close_workers = True
            else:
                close_workers = False

            for splits in workers.uimap(split_path_to_patches, self.paths, chunksize=paths_per_worker):
                distribute_path_in_patches(*splits)

            if close_workers:
                workers.close()
                workers.join()
                workers.clear()

        return vector_patches


    def translate(self, translation_vector, adjust_view=False, margin=.01):
        if adjust_view:
            minx, maxx, miny, maxy = common_utils.bbox(seg for path in self.paths for seg in path.segments)
            self.view_x.value = 0
            self.view_y.value = 0
            self.view_width.value = maxx - minx + translation_vector[0]
            self.view_height.value = maxy - miny + translation_vector[1]

            margin_w = self.view_width.value * margin
            margin_h = self.view_height.value * margin
            self.view_width.value *= (1 + margin * 2)
            self.view_height.value *= (1 + margin * 2)

            translation_vector = translation_vector[0] - minx + margin_w, translation_vector[1] - miny + margin_h
            [path.translate(translation_vector) for path in self.paths]
        else:
            self.view_x -= translation_vector[0]
            self.view_y -= translation_vector[1]


    def translated(self, *args, **kwargs):
        newim = self.copy()
        newim.translate(*args, **kwargs)
        return newim


    def vahe_representation(self, max_lines_n=maxsize, max_beziers_n=maxsize, random_sampling=False):
        if random_sampling:
            return self._vahe_representation_random(max_lines_n=max_lines_n, max_beziers_n=max_beziers_n)
        else:
            return self._vahe_representation_first(max_lines_n=max_lines_n, max_beziers_n=max_beziers_n)


    def _vahe_representation_first(self, max_lines_n, max_beziers_n):
        lines = list(itertools.islice((seg.vahe_representation(float(path.width.as_pixels()))\
                for path in self.paths if path.width is not None \
                    for seg in path if isinstance(seg, Line)),
            max_lines_n))
        beziers = list(itertools.islice((seg.vahe_representation(float(path.width.as_pixels()))\
                for path in self.paths if path.width is not None \
                    for seg in path if isinstance(seg, Bezier)),
            max_lines_n))
        return lines, beziers


    def _vahe_representation_random(self, max_lines_n, max_beziers_n):
        lines, beziers = self._vahe_representation_first(max_lines_n=maxsize, max_beziers_n=maxsize)
        if len(lines) > max_lines_n:
            ids = random.sample(range(len(lines)), max_lines_n)
            lines = [lines[idx] for idx in primitive_ids]
        if len(beziers) > max_beziers_n:
            ids = random.sample(range(len(beziers)), max_beziers_n)
            beziers = [beziers[idx] for idx in primitive_ids]
        return lines, beziers


class Path:
    def __init__(self, path, attributes, flatten_transforms=True, convert_to_pixels=True):
        # geometry
        if flatten_transforms:
            if 'transform' in attributes:
                matrix = svgpathtools.parser.parse_transform(attributes['transform'])
                path = svgpathtools.path.transform(path, matrix)
        
        self.segments = [Primitive.from_seg(seg) for seg in path if not isinstance(seg, svgpathtools.Arc)]
        
        # appearance
        self.fill = parse.fill(attributes)
        self.width = parse.stroke(attributes)

        self.convert_to_pixels = convert_to_pixels
        if convert_to_pixels:
            if self.width is not None:
                self.width = self.width.as_pixels()
            # TODO convert segments to pixels?


    @classmethod
    def create_visible(cls, *args, **kwargs):
        path = cls(*args, **kwargs)
        if path.nonempty and path.visible:
            return path
        else:
            return None


    def copy(self):
        return deepcopy(self)


    def copy_shallow(self, segments=None):
        new_path = copy(self)
        if segments is not None:
            new_path.segments = list(segments)
        return new_path


    def __getitem__(self, index): return self.segments[index]
    def __iter__(self): return self.segments.__iter__()
    def __len__(self): return len(self.segments)


    def mirror(self, x):
        [seg.mirror(x) for seg in self.segments]


    def remove_fill(self, default_width):
        if self.fill:
            self.fill = False
            if self.width is None:
                self.width = units.fromrepr(default_width, units.Pixels)
                if self.convert_to_pixels:
                    self.width = self.width.as_pixels()


    def __repr__(self):
        return "Path({})".format(
            ",\n     ".join(repr(x) for x in self.segments))


    def rotate(self, rotation_deg, origin):
        [seg.rotate(rotation_deg, origin) for seg in self.segments]


    def scale(self, scale):
        if self.width is not None:
            self.width *= scale
        [seg.scale(scale) for seg in self.segments]

    
    def scaled(self, scale):
        path = self.copy()
        path.scale(scale)
        return path


    def split_to_patches(self, *args, **kwargs):
        iS, jS, segments_in_patches = split_to_patches(self.segments, *args, **kwargs)
        return iS, jS, [self.copy_shallow(segments) for segments in segments_in_patches]


    def to_svgpathtools(self):
        path = svgpathtools.Path(*(seg.to_svgpathtools() for seg in self.segments))
        attributes = {}
        attributes['fill'] = 'black' if self.fill else 'none'
        if self.width is not None:
            attributes['stroke'] = 'black'
            attributes['stroke-width'] = str(self.width)
        return path, attributes


    def translate(self, t):
        [seg.translate(t) for seg in self.segments]


    def translated(self, t):
        path = self.copy()
        path.translate(t)
        return path


    def with_simplified_segments(self, distinguishability_threshold):
        self.segments = list(filter(None, (subseg for seg in self.segments for subseg in seg.simplified(distinguishability_threshold))))
        return self


    nonempty = property(lambda self: len(self.segments) > 0)
    visible = property(lambda self: self.width is not None or self.fill)
