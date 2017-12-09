import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq

f = open('indices_I101.csv', 'r')

reader = csv.reader(f)
header = next(reader)

date=[]
opening=[]
high=[]
low=[]
closing=[]

for row in reader:
	date.append(str(row[0]))
	opening.append(float(row[1]))
	high.append(float(row[2]))
	low.append(float(row[3]))
	closing.append(float(row[4]))
f.close()

import esn

din = np.array(closing[0:300],np.float32)
dout = np.array([0]*300,np.float32)
yf = fft(din)

print dout

plt.plot(closing, label="close")
plt.plot(opening, label="open")

echo = esn.ESN(1,300,1)
echo.train(yf,dout)
out = esn.prop_sequence(yf)[1]

plt.plot(out, label="esn")

plt.legend()
plt.show()