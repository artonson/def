from io import BytesIO

import py7zlib


class AbstractABCDataProvider(ABC):
    pass


class ABC7zDataProvider:
    def __init__(self, filename):
        self.filename = filename
        try:
            self.modality = _extract_modality(filename)
        except:
            raise ValueError('cannot understand data modality for file "{}"'.format(self.filename))
        if self.modality not in ALL_ABC_MODALITIES:
            raise ValueError('unknown modality: "{}"'.format(self.modality))
        self._open()

    def _open(self):
        self._file_handle = open(self.filename, 'rb')
        self._archive_handle = py7zlib.Archive7z(self._file_handle)

    def get_names_list(self):
        return self._archive_handle.getnames()

    def get_item_by_name(self, name):
        return BytesIO(self._archive_handle.getmember(name).read())

    def close(self):
        self._file_handle.close()


class ABCUnarchivedDataProvider:
    def __init__(self, dirname):
        self.dirname = dirname

    def _open(self):
