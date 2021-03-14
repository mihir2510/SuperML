from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import mpld3

import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import StringIO


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

xpos = [1,2,3,1,2,3,1,2,3]
ypos = [1,2,3,4,1,2,3,4,1]
num_elements = len(xpos)
zpos = [0,0,0,0,0,0,0,0,0]
dx = np.ones(9)
dy = np.ones(9)
dz = [1,2,3,4,5,6,7,8,9]

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=('red', 'blue', 'purple','red', 'blue', 'purple','red', 'blue', 'purple'))
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

ticks = [1.5, 2.5, 3.5]
labels = ["a", "b", "c"]
ticks1 = [1.5, 2.5, 3.5,4.5]
labels1 = ["a", "b", "c",'d']
plt.xticks(ticks, labels)
plt.yticks(ticks1, labels1)
plt.show()

# html_str = mpld3.fig_to_html(fig)
# Html_file= open("index.html","w")
# Html_file.write(html_str)
# Html_file.close()

tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

html =  '<iframe src=\'data:image/png;base64,{}\'>'.format(encoded) 

with open('index.html','w') as f:
    f.write(html)
