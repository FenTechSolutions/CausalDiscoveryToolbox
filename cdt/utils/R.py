class DefaultRPackages(object):
    __slots__ = ("pcalg",
                 "minet")

    def __init__(self):  # Define here the default values of the parameters
        self.pcalg = None
        self.minet = None

RPackages = DefaultRPackages()
