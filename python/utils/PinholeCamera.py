class PinholeCamera(object):
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

    # str function
    def __str__(self):
        return "PinholeCameraModel: fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f, width: %d, height: %d" % (
            self.fx, self.fy, self.cx, self.cy, self.width, self.height)
