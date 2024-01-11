import numpy as np
import matplotlib.patches as mpatches
import matplotlib.path as mpath


"""
An amusingly overcomplicated and unnecessary legend handler that
looks like a gaussian.
"""

class Handler():
    def __init__(self, theme='blue', skew=1):
        if theme == 'black':
            self.facecolor = '0.5'
            self.edgecolor = 'black'
        elif theme == 'blue':
            self.facecolor = 'cornflowerblue'
            self.edgecolor = 'mediumblue'
        elif theme == 'green':
            self.facecolor = 'limegreen'
            self.edgecolor = 'darkgreen'
        elif theme == 'red':
            self.facecolor = 'red'
            self.edgecolor = 'darkred'
        elif theme == 'orange':
            self.facecolor = 'gold'
            self.edgecolor = 'darkorange'
        elif isinstance(theme, list):
            self.facecolor = theme[0]
            self.edgecolor = theme[1]
        Path = mpath.Path

        if skew==0:
            self.path_data = [
                (Path.MOVETO, [0.0, 0.]),
                (Path.CURVE4, [0.25, 0.15]),
                (Path.CURVE4, [0.35, 0.99]),
                (Path.CURVE4, [0.5, 1.0]),

                (Path.CURVE4, [0.65, 0.99]),
                (Path.CURVE4, [0.75, 0.15]),
                (Path.CURVE4, [1.0, 0.0]),
                (Path.CLOSEPOLY, [0.0, 0.0])]
        elif skew==1:
            self.path_data = [
                (Path.MOVETO, [0.0, 0.]),  # Left edge
                (Path.CURVE4, [0.2, 0.2]),
                (Path.CURVE4, [0.3, 0.99]),
                (Path.CURVE4, [0.44, 1.0]),  # Peak

                (Path.CURVE4, [0.6, 0.99]),
                (Path.CURVE4, [0.65, 0.3]),
                (Path.CURVE4, [1.0, 0.0]),  # Right edge
                (Path.CLOSEPOLY, [0.0, 0.0])]
        elif skew==2:
            self.path_data = [
                (Path.MOVETO, [0.0, 0.]),  # Left edge
                (Path.CURVE4, [0.2, 0.2]),
                (Path.CURVE4, [0.3, 0.99]),
                (Path.CURVE4, [0.55, 1.0]),  # Peak

                (Path.CURVE4, [0.65, 0.9]),
                (Path.CURVE4, [0.75, 0.3]),
                (Path.CURVE4, [1.0, 0.0]),  # Right edge
                (Path.CLOSEPOLY, [0.0, 0.0])]

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        codes, verts = zip(*self.path_data)
        xmin, ymin = np.amin(verts, axis=0)
        xmax, ymax = np.amax(verts, axis=0)
        verts = np.array(verts)
        verts[:,0] = (verts[:,0]-xmin)/(xmax-xmin)*width
        verts[:,1] = (verts[:,1]-ymin)/(ymax-ymin)*height
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(
            path,
            facecolor=self.facecolor,
            edgecolor=self.edgecolor,
            lw=1.0,
        )
        handlebox.add_artist(patch)
        return patch

