import matplotlib.pyplot as plt
import numpy as np

time0,picos0=np.loadtxt('Practica 1\\137Cs_1_resultados.txt', skiprows=1, unpack=True, delimiter=',')
plt.figure()
plt.title('Amplitud de los picos', fontsize = 14)
plt.hist(picos0, bins=500, label = 'data')
plt.xlabel('Tensión [V]', fontsize = 14)
plt.ylabel('Cuentas', fontsize = 14)
plt.yscale("log")
plt.legend(fontsize = 14)
plt.show()