
# General tools for the lensing module

class classonly(classmethod):
    def __get__(self, obj, type):
        if obj: raise AttributeError
        return super(classonly, self).__get__(obj, type)