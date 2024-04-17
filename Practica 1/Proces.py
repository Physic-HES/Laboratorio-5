import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd


def Gaussian(x,a,b,c,d):
    return d+c*np.exp(-(((x-a)/b)**2)/2)

def lineal(x,m,b):
    return m*x+b

class calib:
    def __init__(self):
        self.pendiente=1
        self.offset=0
        self.popterr=0
        self.std=1
        self.R2=1
        self.tab=pd.DataFrame({'E_Fotopico [keV]' : [], 'Voltaje_Fotopico [V]' : [], 'Elemento' : []})
        self.calib_ajustes=[]
        self.calib_etiq_ajustes=[]
    
    def add_med(self,name):
        elementos=['137Cs','133Ba','22Na','207Bi','60Co','57Co','241Am','152Eu']
        energias=[661.66,383.86,383.86,569.15,1332.51,136.47,59.54,1457.63]
        if name[:name.find('_')] in elementos:
            E=energias[elementos.index(name[:name.find('_')])]
            cond=1
        else:
            print('No hay datos de energia para el elemento '+name)
            cond=0
        if cond==1:
            time0,picos0=np.loadtxt(name+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
            plt.figure()
            spec=plt.hist(picos0, bins=1000, histtype='step',label = name)
            ind_peaks, _ = find_peaks(spec[0], height = 200, width = 0, distance = 80, prominence=900)
            print('Picos econtrados [V]:')
            print(spec[1][ind_peaks])
            ind_entorno_fotopico=np.where(np.abs(spec[1]-spec[1][np.max(ind_peaks)])<0.025*np.max(spec[1]))[0]
            
            # Ajuste Gaussiano
            popt0=[np.mean([np.min(spec[1][ind_entorno_fotopico]),np.max(spec[1][ind_entorno_fotopico])]),
                   (np.max(spec[1][ind_entorno_fotopico])-np.min(spec[1][ind_entorno_fotopico]))**2,
                   np.max(spec[0][ind_entorno_fotopico])-np.min(spec[0][ind_entorno_fotopico]),
                   np.min(spec[0][ind_entorno_fotopico])]
            popt, pcov = curve_fit(Gaussian,spec[1][ind_entorno_fotopico],spec[0][ind_entorno_fotopico],p0=popt0)
            popterr = np.sqrt(np.diag(pcov))
            print('Centro de la gaussiana')
            print(f'a: {popt[0]:.3}+-{popterr[0]:.3}')
            xteo=np.linspace(np.min(spec[1][ind_entorno_fotopico]),np.max(spec[1][ind_entorno_fotopico]),200)
            yteo=Gaussian(xteo,popt[0],popt[1],popt[2],popt[3])
            
            plt.plot(xteo, yteo, 'r-',label='Ajuste Gaussiano')
            raw=pd.DataFrame({'E_Fotopico [keV]' : [E], 'Voltaje_Fotopico [V]' : [popt[0]], 'Elemento' : [name[:name.find('_')]]})
            self.tab=pd.concat([self.tab,raw], ignore_index=True)
            self.calib_ajustes.append(np.c_[xteo,yteo])
            self.calib_etiq_ajustes.append('Ajuste Gaussiano para '+name)
            self.calibrar()
            plt.title('Ajuste Distribución Gaussiana')
            plt.xlabel('Valor máximo de tensión [V]')
            plt.ylabel('Cuentas')
            plt.legend()
            plt.yscale("log")
            plt.show()

    def calibrar(self):
        med_cal=len(self.tab['Voltaje_Fotopico [V]'].values)
        if med_cal>2:
            p=np.polyfit(self.tab['Voltaje_Fotopico [V]'].values,self.tab['E_Fotopico [keV]'].values,1)
            popt,pcov=curve_fit(lineal,self.tab['Voltaje_Fotopico [V]'].values,self.tab['E_Fotopico [keV]'].values,p0=[p[0],p[1]])
            self.popterr=np.sqrt(np.diag(pcov))
            self.offset=popt[1]
            self.pendiente=popt[0]
            self.R2=np.corrcoef(self.tab['E_Fotopico [keV]'].values,self.offset+self.pendiente*self.tab['Voltaje_Fotopico [V]'].values)[0][1]**2
            self.std=np.std(self.tab['E_Fotopico [keV]'].values-(self.offset+self.pendiente*self.tab['Voltaje_Fotopico [V]'].values))
            print('RESULTADOS DE CALIBRACIÓN:')
            print(f'Pendiente = {self.pendiente:0.5}+-{self.popterr[0]:0.5} keV/V')
            print(f'Offset = {self.offset:0.5}+-{self.popterr[1]:0.5} keV/V')
            print(f'R^2 = {self.R2:1.5}')
            print(f'sigma = {self.std:0.5} keV')
        else:
            print(f'Se necesitan al menos {3-med_cal} mediciones más para poder calibrar')

    def result(self):
        print('RESULTADOS DE CALIBRACIÓN:')
        print(f'Pendiente = {self.pendiente:0.5}+-{self.popterr[0]:0.5} keV/V')
        print(f'Offset = {self.offset:0.5}+-{self.popterr[1]:0.5} keV/V')
        print(f'R^2 = {self.R2:1.5}')
        print(f'sigma = {self.std:0.5} keV')

    def plot_cal(self):
        if not self.pendiente==1:
            plt.figure()
            x=np.linspace(np.min(self.tab['Voltaje_Fotopico [V]'].values),np.max(self.tab['Voltaje_Fotopico [V]'].values),1000)
            mk=['^','s','*','d']
            for j in range(len(self.tab['Voltaje_Fotopico [V]'].values)):
                elem=self.tab['Elemento'].values[j]
                plt.plot(self.tab['Voltaje_Fotopico [V]'].values[j],self.tab['E_Fotopico [keV]'].values[j],marker=mk[j],label=f'{elem}')
            plt.plot(x,self.offset+self.pendiente*x,'-b',label=rf'Ajuste [$R^2={self.R2:0.4}$]')
            plt.fill_between(x,self.offset+self.pendiente*x+2*self.std,
                                self.offset+self.pendiente*x-2*self.std,alpha=0.25,label=f'$2\sigma$')
            plt.xlabel('Tensión de Fotopico [V]')
            plt.ylabel('Energia de Fotopico [keV]')
            plt.legend()
            plt.show()

class Med:
    def __init__(self,m,b):
        self.pendiente=m
        self.offset=b
        self.curvas=[]
        self.tam=[]
        self.etiquetas=[]
        self.peaks=[]

    def add_med(self, name):
        time0,picos0=np.loadtxt(name+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
        picos0=picos0*self.pendiente+self.offset
        spec=plt.hist(picos0, bins=1000, histtype='step',label = name)
        ind_peaks, _ = find_peaks(spec[0], height = 200, width = 0, distance = 25, prominence=np.max(spec[0])/10)
        self.curvas.append(spec)
        self.tam.append(len(picos0))
        self.etiquetas.append(name[:name.find('_')])
        self.peaks.append(ind_peaks)
        plt.close()

    def plot_med(self):
        plt.figure()
        for j in range(len(self.curvas)):
            plt.plot(self.curvas[j][1][:-1],self.curvas[j][0],label=self.etiquetas[j])
            plt.title('Espectro de emisiones $\gamma$')
        if not self.pendiente==1:
            plt.xlabel('Energia [keV]')
        else:
            plt.xlabel('Valores maximos de tensión [V]')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.yscale("log")
        plt.show()



'''DISTBUCIN DE POISSON
lambda_ = np.mean(picos0[ind_entorno_fotopico])*100 
k = np.arange(0, np.floor(np.max(picos0))*100)
pmf_poisson = poisson.pmf(k, lambda_)*len(ind_entorno_fotopico)
indexs_fotopico_k=np.where(np.abs(k/100-spec[1][np.max(ind_peaks)]+0.01)<0.04*np.max(spec[1]))[0]
'''

cal=calib()
cal.add_med('137Cs_2T')
cal.add_med('133Ba_2')
cal.add_med('207Bi_1')
cal.plot_cal()
medicion=Med(cal.pendiente,cal.offset)
medicion.add_med('137Cs_2T')
medicion.add_med('133Ba_2')
medicion.add_med('34Es_1')
medicion.add_med('207Bi_1')
medicion.plot_med()