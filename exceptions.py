class Error(Exception):
    pass


class UnphysicalError(Error):
    '''Class of errors leading to unphysical results.
       Right now only used in (energy) fractured propagation without a seed.
    '''

    def __init__(self):
        self.message = "You were doing something unphysical here. Stop that!"
        super().__init__(self.message)
