class Coordinates:
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __str__(self):
        return 'xmin: ' + str(self.xmin) + ' ymin: ' + str(self.ymin) + \
               ' xmax: ' + str(self.xmax) + ' ymax: ' + str(self.ymax)