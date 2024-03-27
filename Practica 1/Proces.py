import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.grid()
time0,picos0=np.loadtxt('Practica 1\\137Cs_1_resultados.txt', skiprows=1, unpack=True, delimiter=',')
spectrum_137Cs=plt.hist(picos0*662/3.4, bins=1000, histtype='step',label = '137Cs')
time0,picos0=np.loadtxt('Practica 1\\60Co_1_resultados.txt', skiprows=1, unpack=True, delimiter=',')
spectrum_60Co=plt.hist(picos0*662/3.4, bins=1000, histtype='step',label = '60Co')
plt.title(f'Amplitud de los picos para {len(picos0)} eventos', fontsize = 14)
plt.xlabel('Energía [keV]', fontsize = 14)
plt.ylabel('Cuentas', fontsize = 14)
plt.yscale("log")
plt.legend(fontsize = 14)
plt.show()