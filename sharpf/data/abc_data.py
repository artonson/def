class ABCData(object):
    pass

import py7zlib
class SevenZFile(object):
    @classmethod
    def is_7zfile(cls, filepath):
        '''
        Class method: determine if file path points to a valid 7z archive.
        '''
        is7z = False
        fp = None
        try:
            fp = open(filepath, 'rb')
            archive = py7zlib.Archive7z(fp)
            n = len(archive.getnames())
            is7z = True
        finally:
            if fp:
                fp.close()
        return is7z

    def __init__(self, filepath):
        fp = open(filepath, 'rb')
        self.archive = py7zlib.Archive7z(fp)

    def extractall(self, path):
        for name in self.archive.getnames():
            outfilename = os.path.join(path, name)
            outdir = os.path.dirname(outfilename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = open(outfilename, 'wb')
            outfile.write(self.archive.getmember(name).read())
            outfile.close()


    f = open('abc_0099_meta_v00.7z', 'rb')
    a = py7zlib.Archive7z(f)
    a.getnames()


    from io import BytesIO
    b = BytesIO()

