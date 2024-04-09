import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from scipy.signal import find_peaks
import pandas as pd


class calib:
    def __init__(self):
        self.pendiente=1
        self.std=1
        self.R2=1
        self.tab=pd.DataFrame({'E_Fotopico [keV]' : [], 'Voltaje_Fotopico [V]' : []})
        self.calib_ajustes=[]
        self.calib_etiq_ajustes=[]

    def add_med(self,name):
        elementos=['137Cs','133Ba','22Na','207Bi','60Co','57Co','241Am','152Eu']
        energias=[661.66,383.86,1274.5,1770.22,1332.51,136.47,59.54,1457.63]
        if name[:name.find('_')] in elementos:
            E=energias[elementos.index(name[:name.find('_')])]
            cond=1
        else:
            print('No hay datos de energia para el elemento '+name)
            cond=0
        if cond==1:
            time0,picos0=np.loadtxt('Practica 1\\'+name+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
            plt.figure()
            spec=plt.hist(picos0, bins=1000, histtype='step',label = name)
            ind_peaks, _ = find_peaks(spec[0], height = 3000, width = 0, distance = len(spec[0])/10)
            print(ind_peaks)
            ind_entorno_fotopico=np.where(np.abs(picos0-spec[1][np.max(ind_peaks)])<0.125*np.max(spec[1]))[0]
            lambda_ = np.mean(picos0[ind_entorno_fotopico])*100 
            k = np.arange(0, np.floor(np.max(picos0))*100)
            pmf_poisson = poisson.pmf(k, lambda_)*len(ind_entorno_fotopico)
            indexs_fotopico_k=np.where(np.abs(k/100-spec[1][np.max(ind_peaks)])<0.125*np.max(spec[1]))[0]
            plt.plot(k[indexs_fotopico_k]/100, pmf_poisson[indexs_fotopico_k], 'b-',label='Ajuste de Poisson')
            print(k[np.argmax(pmf_poisson[indexs_fotopico_k])])
            self.tab=pd.concat(self.tab,pd.DataFrame({'E_Fotopico [keV]' : [E], 'Voltaje_Fotopico [V]' : [k[np.argmax(pmf_poisson[indexs_fotopico_k])]]}), ignore_index=True)
            self.calib_ajustes.append(np.c_[k[indexs_fotopico_k]/100,pmf_poisson[indexs_fotopico_k]])
            self.calib_etiq_ajustes.append('Ajuste de Poisson para '+name)
            self.calibrar()
            plt.title('Ajuste de la Distribución de Poisson')
            plt.xlabel('Valor máximo de tensión')
            plt.ylabel('Cuentas')
            plt.legend()
            plt.yscale("log")
            plt.show()

    def calibrar(self):
        p=np.polyfit(self.tab['E_Fotopico [keV]'].values,self.tab['Voltaje_Fotopico [V]'].values,1)
        self.pendiente=p[1]
        self.R2=np.corrcoef(self.tab['Voltaje_Fotopico [V]'].values,p[0]+p[1]*self.tab['E_Fotopico [keV]'].values)[0][1]**2
        self.std=np.std(self.tab['Voltaje_Fotopico [V]'].values-(p[0]+p[1]*self.tab['E_Fotopico [keV]'].values))

class Med(calib):
    def __init__(self):
        super().__init__()
        self.curvas=[]
        self.ajustes=[]
        self.etiquetas=[]
        self.etiq_ajustes=[]

    def add_med(self, name):
        time0,picos0=np.loadtxt('Practica 1\\'+name+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
        picos0/=self.pendiente
        plt.figure()
        spec=plt.hist(picos0, bins=1000, histtype='step',label = name)
        ind_peaks, _ = find_peaks(spec[0], height = np.max(spec[0])/2, width = 0, distance = len(spec[0])/10)
        ind_entorno_fotopico=np.where(np.abs(picos0-spec[1][np.max(ind_peaks)])<0.125*np.max(spec[1]))[0]
        lambda_ = np.mean(picos0[ind_entorno_fotopico])*100 
        k = np.arange(0, np.floor(np.max(picos0))*100)
        pmf_poisson = poisson.pmf(k, lambda_)*len(ind_entorno_fotopico)
        indexs_fotopico_k=np.where(np.abs(k/100-spec[1][np.max(ind_peaks)])<0.125*np.max(spec[1]))[0]
        self.ajustes.append(np.c_[k[indexs_fotopico_k]/100,pmf_poisson[indexs_fotopico_k]])
        self.etiq_ajustes.append('Ajuste de Poisson para '+name)
        self.curvas.append(np.c_[spec[1],spec[0]])
        self.etiquetas.append(name)
        plt.plot(k[indexs_fotopico_k]/100, pmf_poisson[indexs_fotopico_k], 'b-',label='Ajuste de Poisson')
        plt.title('Ajuste de la Distribución de Poisson')
        plt.xlabel('Valor máximo de tensión')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.yscale("log")
        plt.show()


calibracion=calib()
calibracion.add_med('137Cs_1')