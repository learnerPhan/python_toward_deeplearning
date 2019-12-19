import random
import matplotlib.pyplot as plt
import numpy as np

# plt.title('Cross-validation on k')
# plt.xlabel('k')
# plt.ylabel('Cross-validation accuracy')
# plt.show()
"""
source : https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
"""
"""
Errorbar
"""
x = np.array([1,2,3])
dy = 0.5
y = x + 1

print(x)
print(y)

plt.errorbar(x, y, yerr=dy, fmt='o', color='green',
             ecolor='lightgreen', elinewidth=0, capsize=3);
plt.show()

"""
Continuous errorbar
"""

xdata = np.array([1, 2, 3, 4])
ydata = xdata + 1

plt.plot(xdata, ydata, 'or')
plt.plot(xdata, ydata, '-')

xfit = xdata + 0.5
yfit = ydata - 0.5

plt.plot(xfit, yfit, '-', color='gray')

dyfit = 0.5

plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
# plt.xlim(0, 10);
plt.show()

"""
https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks
"""
x = np.linspace(0, 10, 6)
#generate 6 points in [0, 10]

y = x

# plt.plot(x, y, 'o')
# plt.plot(x, y, '-')

#fig may be saved later
fig = plt.figure()

plt.plot(x, y, '--')

plt.show()

#save the figure into a file
fig.savefig("fig1.png")