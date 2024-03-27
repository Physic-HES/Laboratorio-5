import matplotlib.pyplot as plt
import numpy as np

plt.figure()
time0,picos0=np.loadtxt('Practica 1\\137Cs_1_resultados.txt', skiprows=1, unpack=True, delimiter=',')
spectrum_137Cs=plt.hist(picos0, bins=500, histtype='step',label = '137Cs')
time0,picos0=np.loadtxt('Practica 1\\60Co_1_resultados.txt', skiprows=1, unpack=True, delimiter=',')
spectrum_60Co=plt.hist(picos0, bins=500, histtype='step',label = '60Co')
plt.title('Amplitud de los picos', fontsize = 14)
plt.xlabel('Tensión [V]', fontsize = 14)
plt.ylabel('Cuentas', fontsize = 14)
plt.yscale("log")
plt.legend(fontsize = 14)
plt.show()