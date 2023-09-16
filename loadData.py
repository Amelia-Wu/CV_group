import os
import glob

class ImageFinder:
    def __init__(self, path):
        self.path = path

    def get_jpg_filenames(self):
        """返回指定路径下所有的.jpg文件的文件名列表"""
        return glob.glob(os.path.join(self.path, "*.jpg"))