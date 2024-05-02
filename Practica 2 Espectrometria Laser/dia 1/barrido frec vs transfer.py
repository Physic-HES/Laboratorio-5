# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:52:54 2022

@author: Publico
"""


import pyvisa as visa
import time
import numpy as np
from matplotlib import pyplot as plt


rm = visa.ResourceManager()

instrumentos = rm.list_resources()  
print('instrumentos:',instrumentos)

osc=rm.open_resource(instrumentos[1])

print(osc.query('*IDN?'))
gen=rm.open_resource(instrumentos[0])

print(gen.query('*IDN?'))

osc.write('CH1:COUPling DC')    # El canal queda acoplado en DC indepen-
osc.write('CH2:COUPling DC')  

#osc.write('HORizontal:MAIn:SCAle 0.01')   #escala horizontal del tiempo
osc.write('CH1:SCAle 0.2')    #escala vertical de voltaje                             
time.sleep(1)
osc.write('MEASU:MEAS1:SOURCE CH1') #define la fuente de la medicion nro 1
osc.write('MEASU:MEAS1:TYPE PK2pk') # define el tipo de medicion, vpp, vmax, etc.
time.sleep(1)
osc.write('MEASU:MEAS2:SOURCE CH2') #define la fuente de la medicion nro 1
osc.write('MEASU:MEAS2:TYPE PK2pk') # define el tipo de medicion, vpp, vmax, etc.
time.sleep(1)


#frec = 5   #1/espera #float(input("Introduzca la frecuencia de muestreo en Hz: "))#fija el tiempo entre mediciones
frec1 = 10  #float(input("Introduzca la frecuencia minima en Hz: "))
frec2 = 2000  #float(input("Introduzca la frecuencia maxima en Hz: "))
#paso = float(input("Introduzca el paso en Hz: "))
N=100 #int(input("Introduzca el número de mediciones: "))

datos=[]

# Rampa lineal de frequencias
#frecuencias = np.linspace(frec1, frec2, int((frec2-frec1)/paso)+1)
frecuencias=np.geomspace(frec1,frec2,N) #como linspace pero espacioado logaritmico
#es similar al logspace pero no hace falta cargar las frecs 1 y 2 con valores logaritmicos


print(frecuencias)

X=[]
Y1=[]
Y2=[]
for freq in frecuencias:
    #sch=(1/freq)/10
    #sch=str('HORizontal:MAIn:SCAle '+str(sch))
    #osc.write(sch)   #escala horizontal del tiempo
    #time.sleep(1)
    time.sleep(0.5)
    X.append(freq)
    gen.write('FREQ %f' % freq)
    VPP1=osc.query_ascii_values('MEASU:MEAS1:VAL?') #da como resultado una lista con números float
    VPP2=osc.query_ascii_values('MEASU:MEAS2:VAL?') #da como resultado una lista con números float
    
    Y1.append(VPP1)
    Y2.append(VPP2)

import pandas as pd

mediciones=np.zeros([3,2500])
mediciones=[X,Y1, Y2]
mediciones=np.transpose(mediciones)
df=pd.DataFrame(mediciones)
#print(time.localtime())
df.to_csv('Mediciones'+str(time.localtime()[0])+'-'+str(time.localtime()[1])+'-'+str(time.localtime()[2])+'-'+str(time.localtime()[3])+'-'+str(time.localtime()[4])+'-'+str(time.localtime()[5])+'.csv')


try:
    TRANSF=Y2/Y1
except:
    TRANSF=[]
    for i in range(len(Y1)):
        TRANSF.append(Y2[i][0]/Y1[i][0])
plt.plot(X, TRANSF, 'ob')
plt.xscale('log')
plt.yscale('log')
plt.show()