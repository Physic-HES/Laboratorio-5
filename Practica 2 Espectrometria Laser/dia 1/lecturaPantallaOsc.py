# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:52:54 2022

@author: Publico
"""

import time
import numpy as np
from matplotlib import pyplot as plt

'''

import pyvisa as visa
rm = visa.ResourceManager()

instrumentos = rm.list_resources()  
print('instrumentos:',instrumentos)
#%%

osc=rm.open_resource(instrumentos[0])

print(osc.query('*IDN?'))
#%%
#osc.write('CH1:COUPling DC')    # El canal queda acoplado en DC indepen-
#osc.write('CH2:COUPling DC')  

#osc.write('HORizontal:MAIn:SCAle 0.01')   #escala horizontal del tiempo
#osc.write('CH1:SCAle 0.2')    #escala vertical de voltaje                             
time.sleep(1)
#osc.write('MEASU:MEAS1:SOURCE CH1') #define la fuente de la medicion nro 1
#osc.write('MEASU:MEAS1:TYPE PK2pk') # define el tipo de medicion, vpp, vmax, etc.
#time.sleep(1)


datos=[]


X=[]
Y1=[]
Y2=[]

#time.sleep(0.5) # <------------- Pide la configuración de ejes del osciloscopio.
xze,xin=osc.query_ascii_values('WFMPRE:XZE?;XIN?',separator=';') #conf. base de tiempo
yze1,ymu1,yoff1=osc.query_ascii_values('WFMPRE:CH1:YZE?;YMU?;YOFF?',separator=';') #conf. vertical canal 1

## Modo de transmision: Binario (osea, la onda digitalizada)
osc.write('DAT:ENC RPB') 
osc.write('DAT:WID 1') 

#leo las curvas como datos binarios
osc.write('DAT:SOU CH1' )
data1=osc.query_binary_values('CURV?', datatype='B',container=np.array)	


time.sleep(0.5)											
#transformo los datos binarios en tiempo y voltajes	
tiempo = xze + np.arange(len(data1)) * xin #tiempos en s
data1V=(data1-yoff1)*ymu1+yze1 #tensión canal 1 en V 
#%%
plt.plot(tiempo, data1V, 'ob')
#plt.xscale('log')
#plt.yscale('log')
plt.show()
#%%
fil=open('Captura_2.txt','w')

fil.write('tiempo, voltaje \n')
for i in range(len(tiempo)):
    fil.write(str(tiempo[i])+','+str(data1V[i])+'\n')
    
fil.close()

#%%

def Line(x,a,b):
    return a*x+b

yline=Line(tiempo, 6.985645,0.0668)
print(type(yline), len(data1V),len(yline))

datos2=data1V-yline
#%%
plt.plot(tiempo, datos2, '-b', ms=3)
#plt.xscale('log')
#plt.yscale('log')
plt.savefig('captura2_filtrado_line.png')
plt.show()
#%%

fil=open('Captura_2_filtrada.txt','w')

fil.write('tiempo, voltaje \n')
for i in range(len(tiempo)):
    fil.write(str(tiempo[i])+','+str(datos2[i])+'\n')
    
fil.close()
'''
#%%
tiempo, datos2=np.genfromtxt('Captura_2_filtrada.txt',delimiter=',',skip_header=1,unpack=True)
#%%
plt.plot(tiempo, datos2, '-b', ms=3)
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('captura2_filtrado_line.png')
plt.show()
#%%
h=6.626E-34
f=np.array([3.035732439E9,377.107385690E12-(1.264888516E9+210.923E6),361.58E6])
f87=np.array([6.834682610904290E9,377.107463380E12-(2.563005979089109E9+509.06E6),814.5E6])
E=h*f
E87=h*f87
Saltos85=np.array([E[0]+E[1],E[0]+E[1]+E[2],E[1],E[1]+E[2]])
Saltos87=np.array([E87[0]+E87[1],E87[0]+E87[1]+E87[2],E87[1],E87[1]+E87[2]])

t1=0.00115
t2=0.00195
t3=0.00285
t4=0.00330
Energia=(tiempo-t2)*(Saltos85[2]-Saltos85[1])/(t3-t2)+Saltos85[1]
Energia87=(tiempo-t1)*(Saltos87[2]-Saltos87[1])/(t4-t1)+Saltos87[1]

#%%
plt.plot(Energia, datos2, '-b', ms=3,label='calib con R85')
plt.plot(Energia87, datos2, '-m', ms=3,label='calib con R87')
#plt.xscale('log')
#plt.yscale('log')
#for i in range(len(Saltos85)):
plt.vlines(Saltos85,min(datos2)*0.8,max(datos2)*0.8, color='g', linestyle='dashdot',label='R85')
plt.vlines(Saltos87,min(datos2)*0.8,max(datos2)*0.8, color='r', linestyle='dashdot',label='R87')
#plt.vlines(Saltos,min(datos2),max(datos2))
#plt.vlines(Saltos,min(datos2),max(datos2))
#plt.vlines(Saltos,min(datos2),max(datos2))
#plt.xlim(min(Energia),1.1*max(Energia))
#plt.ylim(min(datos2),1.1*max(datos2))
plt.legend()
plt.savefig('captura2_calib_line.png')
plt.show()

