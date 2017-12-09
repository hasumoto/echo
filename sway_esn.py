import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq

f = open('sway.csv', 'r')

reader = csv.reader(f)
header = next(reader)

x=[]
y=[]

for row in reader:
	x.append(float(row[1]))
	y.append(float(row[2]))
f.close()

import esn

din = np.array(y[0:300],np.float32)
dout = np.array([0.0]*300,np.float32)
yf = fft(din)

plt.plot(y, label="close")

echo_state = esn.ESN(1,300,1)
echo_state.train(yf,dout)
out = echo_state.prop_sequence(yf)[1]

plt.plot(out, label="esn")

plt.legend()
plt.show()