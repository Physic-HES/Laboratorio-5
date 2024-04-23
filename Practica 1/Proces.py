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
        self.tab=pd.DataFrame({'E [keV]' : [], 'Voltaje [V]' : [], 'Elemento' : []})
        self.calib_ajustes=[]
        self.calib_etiq_ajustes=[]
    
    def add_med(self,name):
        elementos=['137Cs','133Ba','22Na','207Bi','60Co','57Co','241Am','152Eu']
        energias=[661.66,81,1272.5,569.15,1332.51,136.47,59.54,1457.63]
        if name[:name.find('_')] in elementos:
            E=energias[elementos.index(name[:name.find('_')])]
            cond=1
        else:
            print('No hay datos de energia para el elemento '+name)
            cond=0
        if cond==1:
            time0,picos0=np.loadtxt('Practica 1\\'+name+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
            plt.figure()
            spec=plt.hist(picos0, bins=900, histtype='stepfilled',alpha=0.25,label = f'$^{{{name[0:3]}}}{name[3:5]}$')
            plt.hist(picos0, bins=900, histtype='step',color='b')
            ind_peaks, _ = find_peaks(spec[0], height = 200, width = 0, distance = 25, prominence=300)
            print('Picos econtrados [V]:')
            print(spec[1][ind_peaks])
            if elementos.index(name[:name.find('_')])==1:
                ener=[81,383.86]
                for n in range(2):
                    E=ener[n]
                    if n==0:
                        ind_entorno=np.where(np.abs(spec[1]-spec[1][ind_peaks[2]])<0.15)[0]
                    else:
                        ind_entorno=np.where(np.abs(spec[1]-spec[1][ind_peaks[-1]])<0.05*spec[1][np.max(ind_peaks)])[0]
                    
                    # Ajuste Gaussiano
                    popt0=[np.mean([np.min(spec[1][ind_entorno]),np.max(spec[1][ind_entorno])]),
                        (np.max(spec[1][ind_entorno])-np.min(spec[1][ind_entorno]))**2,
                        np.max(spec[0][ind_entorno])-np.min(spec[0][ind_entorno]),
                        np.min(spec[0][ind_entorno])]
                    popt, pcov = curve_fit(Gaussian,spec[1][ind_entorno],spec[0][ind_entorno],p0=popt0)
                    popterr = np.sqrt(np.diag(pcov))
                    print('Centro de la gaussiana')
                    print(f'a: {popt[0]:.3}+-{popterr[0]:.3}')
                    print(f'R^2={np.corrcoef(spec[0][ind_entorno],Gaussian(spec[1][ind_entorno],popt[0],popt[1],popt[2],popt[3]))[0][1]**2:0.5}')
                    xteo=np.linspace(np.min(spec[1][ind_entorno]),np.max(spec[1][ind_entorno]),200)
                    yteo=Gaussian(xteo,popt[0],popt[1],popt[2],popt[3])
                    
                    plt.plot(xteo, yteo,label='Ajuste Gaussiano')
                    raw=pd.DataFrame({'E [keV]' : [E], 'Voltaje [V]' : [popt[0]], 'Elemento' : [name[:name.find('_')]]})
                    self.tab=pd.concat([self.tab,raw], ignore_index=True)
                    self.calib_ajustes.append(np.c_[xteo,yteo])
                    self.calib_etiq_ajustes.append('Ajuste Gaussiano para '+name)
                    self.calibrar(320E-6)
            else:
                ind_entorno=np.where(np.abs(spec[1]-spec[1][np.max(ind_peaks)])<0.05*spec[1][np.max(ind_peaks)])[0]
                
                # Ajuste Gaussiano
                popt0=[np.mean([np.min(spec[1][ind_entorno]),np.max(spec[1][ind_entorno])]),
                    (np.max(spec[1][ind_entorno])-np.min(spec[1][ind_entorno]))**2,
                    np.max(spec[0][ind_entorno])-np.min(spec[0][ind_entorno]),
                    np.min(spec[0][ind_entorno])]
                popt, pcov = curve_fit(Gaussian,spec[1][ind_entorno],spec[0][ind_entorno],p0=popt0)
                popterr = np.sqrt(np.diag(pcov))
                print('Centro de la gaussiana')
                print(f'a: {popt[0]:.3}+-{popterr[0]:.3}')
                print(f'R^2={np.corrcoef(spec[0][ind_entorno],Gaussian(spec[1][ind_entorno],popt[0],popt[1],popt[2],popt[3]))[0][1]**2:0.5}')
                xteo=np.linspace(np.min(spec[1][ind_entorno]),np.max(spec[1][ind_entorno]),200)
                yteo=Gaussian(xteo,popt[0],popt[1],popt[2],popt[3])
                
                plt.plot(xteo, yteo, 'r-',label='Ajuste Gaussiano')
                raw=pd.DataFrame({'E [keV]' : [E], 'Voltaje [V]' : [popt[0]], 'Elemento' : [name[:name.find('_')]]})
                self.tab=pd.concat([self.tab,raw], ignore_index=True)
                self.calib_ajustes.append(np.c_[xteo,yteo])
                self.calib_etiq_ajustes.append('Ajuste Gaussiano para '+name)
                self.calibrar(320E-6)
            plt.xlabel('Canales de energía [V]')
            plt.ylabel('Cuentas')
            plt.xlim([0,np.max(picos0)])
            plt.legend()
            plt.yscale("log")
            if name=='207Bi_1':
                plt.ylim([1E3,1E5])
            plt.show()

    def calibrar(self,error_V):
        med_cal=len(self.tab['Voltaje [V]'].values)
        if med_cal>2:
            p=np.polyfit(self.tab['Voltaje [V]'].values,self.tab['E [keV]'].values,1)
            popt,pcov=curve_fit(lineal,self.tab['Voltaje [V]'].values,self.tab['E [keV]'].values,p0=[p[0],p[1]])
            self.popterr=np.sqrt(np.diag(pcov))
            self.offset=popt[1]
            self.pendiente=popt[0]
            self.error_V=error_V
            self.R2=np.corrcoef(self.tab['E [keV]'].values,self.offset+self.pendiente*self.tab['Voltaje [V]'].values)[0][1]**2
            self.std=np.std(self.tab['E [keV]'].values-(self.offset+self.pendiente*self.tab['Voltaje [V]'].values))
            print('RESULTADOS DE CALIBRACIÓN:')
            print(f'Pendiente = {self.pendiente:0.5}+-{self.popterr[0]:0.5} keV/V')
            print(f'Offset = {self.offset:0.5}+-{self.popterr[1]:0.5} keV')
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
            x=np.linspace(np.min(self.tab['Voltaje [V]'].values),np.max(self.tab['Voltaje [V]'].values),1000)
            mk=['^','s','*','d']
            for j in range(len(self.tab['Voltaje [V]'].values)):
                elem=self.tab['Elemento'].values[j]
                en=self.tab['E [keV]'].values[j]
                plt.plot(self.tab['Voltaje [V]'].values[j],self.tab['E [keV]'].values[j],marker=mk[j],label=f'$^{{{elem[0:3]}}}{elem[3:5]}$ @{en:3.4} keV')
            plt.plot(x,self.offset+self.pendiente*x,'-b',label=f'$E_{{\gamma}}=${self.pendiente:2.3}$V_{{MCA}}$+{self.offset:2.3}\n$R^2=${self.R2:0.4}')
            plt.fill_between(x,self.offset+self.pendiente*x+2*self.std,
                                self.offset+self.pendiente*x-2*self.std,alpha=0.25,label=f'$2\sigma$')
            plt.xlabel('Tensión de pico $V_{MCA}$ [V]')
            plt.ylabel('Energia de pico $E_{\gamma}$ [keV]')
            plt.legend()
            plt.show()

class Med:
    def __init__(self,m,b,Dm,Db,error_V):
        self.pendiente=m
        self.Dm=Dm
        self.offset=b
        self.Db=Db
        self.error_V=error_V
        self.curvas=[]
        self.tam=[]
        self.etiquetas=[]
        self.peaks=[]
        self.Egamma_p=[]
        self.T=[]

    def add_med(self, name):
        plt.figure()
        time0,picos0=np.loadtxt('Practica 1\\'+name+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
        Delta_picos0=np.sqrt(picos0**2*self.Dm**2+self.pendiente**2*self.error_V**2+self.Db**2)
        picos0=picos0*self.pendiente+self.offset
        if name=='34Es_1':
            spec=plt.hist(picos0, bins=1000, histtype='stepfilled',alpha=0.25,label = f'$^{{{name[0:2]}}}{name[2:4]}$')
            plt.hist(picos0, bins=1000, histtype='step',color='b')
        else:
            spec=plt.hist(picos0, bins=1000, histtype='stepfilled',alpha=0.25,label = f'$^{{{name[0:3]}}}{name[3:5]}$')
            plt.hist(picos0, bins=1000, histtype='step',color='b')
        ind_peaks, _ = find_peaks(spec[0], height = 100, width = 0, distance = 30, prominence=200)
        self.curvas.append(spec)
        self.tam.append(len(picos0))
        self.etiquetas.append(name[:name.find('_')])
        self.peaks.append(ind_peaks)
        self.Egamma_p.append([spec[1][ind_peaks[-1]],Delta_picos0[np.argmin(np.abs(picos0-spec[1][ind_peaks[-1]]))]])
        self.T.append([spec[1][ind_peaks[-2]],Delta_picos0[np.argmin(np.abs(picos0-spec[1][ind_peaks[-2]]))]])
        if (name=='137Cs_2T' or name=='207Bi_1'):
            plt.text(spec[1][ind_peaks[-1]],spec[0][ind_peaks[-1]]+1000,'Fotopico',horizontalalignment='center')
            plt.text(spec[1][ind_peaks[-2]],spec[0][ind_peaks[-2]]+500,'Borde Compton')
        elif name=='133Ba_2':
            plt.text(spec[1][ind_peaks[-1]],spec[0][ind_peaks[-1]]+1000,'Fotopico')
        plt.xlim([0,np.max(picos0)])
        if not self.pendiente==1:
            plt.xlabel('Energia [keV]')
        else:
            plt.xlabel('Valores maximos de tensión [V]')
        plt.ylabel('Cuentas')
        plt.xlim([0,715])
        if name=='207Bi_1':
            plt.ylim([1E3,1E5])
        plt.legend()
        plt.yscale("log")

    def plot_med(self):
        plt.show()

class Stat_gamma:
    def __init__(self):
        self.hist=[]
        self.ajustes=[]
        self.media_sigma=[]
        self.etiquetas=[]

    def add_med(self,name):
        eventos=np.loadtxt('Practica 1\\'+name+'_resultados_eventos.txt', skiprows=1, unpack=True, delimiter=',')
        lambda_ = np.mean(eventos) 
        k = np.arange(0, np.max(eventos))
        pmf_poisson = poisson.pmf(k, lambda_)
        spec=plt.hist(eventos, bins=22, histtype='stepfilled',density=True,alpha=0.25,label = f'$\Delta t$={0.25+len(self.hist)*0.25:0.3}')
        ajuste=plt.plot(k,pmf_poisson)
        self.hist.append(spec)
        self.ajustes.append(ajuste)
        self.media_sigma.append([lambda_,np.std(eventos)**2])
        self.etiquetas.append([name])

    def plot_med(self):
        plt.xlabel('Eventos')
        plt.ylabel('Cuentas Normalizadas')
        plt.legend()
        plt.xlim([0,1600])
        if len(self.hist)>1:
            p=np.polyfit(np.array(self.media_sigma)[:,0],np.array(self.media_sigma)[:,1],1)
            popt,pcov=curve_fit(lineal,np.array(self.media_sigma)[:,0],np.array(self.media_sigma)[:,1],p0=[p[0],p[1]])
            self.popterr=np.sqrt(np.diag(pcov))
            self.offset=popt[1]
            self.pendiente=popt[0]
            self.R2=np.corrcoef(np.array(self.media_sigma)[:,1],self.offset+self.pendiente*np.array(self.media_sigma)[:,0])[0][1]**2
            self.std=np.std(np.array(self.media_sigma)[:,1]-(self.offset+self.pendiente*np.array(self.media_sigma)[:,0]))
            mk=['^','s','*','d','p']
            plt.figure()
            plt.plot(np.array(self.media_sigma)[:,0],self.offset+self.pendiente*np.array(self.media_sigma)[:,0],
                     label=f'$\sigma^2$={self.pendiente:1.4}$<n>${self.offset:2.4}\n$R^2$={self.R2:1.5}')
            print(self.pendiente)
            plt.fill_between(np.array(self.media_sigma)[:,0],self.offset+self.pendiente*np.array(self.media_sigma)[:,0]+2*self.std,
                             self.offset+self.pendiente*np.array(self.media_sigma)[:,0]-2*self.std,alpha=0.25,label='$2\sigma$')
            for j in range(len(self.hist)):
                plt.plot(self.media_sigma[j][0],self.media_sigma[j][1],marker=mk[j],label=f'$\Delta t$={0.25+j*0.25}')
            plt.xlabel('$<n>$')
            plt.ylabel('$\sigma^2$')
            plt.legend()
            print('RESULTADOS DE CALIBRACIÓN:')
            print(f'Pendiente = {self.pendiente:0.5}+-{self.popterr[0]:0.5}')
            print(f'Offset = {self.offset:0.5}+-{self.popterr[1]:0.5}')
            print(f'R^2 = {self.R2:1.5}')
            print(f'sigma = {self.std:0.5}')
        plt.show()


class compton:
    def __init__(self):
        self.Egamma=[]
        self.borde=[]
        self.backscater=[]

    def add_med(self,Egamma,T):
        self.Egamma.append(Egamma)
        self.borde.append(T)
        self.backscater.append(Egamma-T)
        if len(self.borde)>1:
            for j in range(len(self.borde)):
                plt.plot(self.borde[j],self.Egamma[j])
        print('listo')

    def masa_e(self,Egamma,T):
        T[0]=T[0]*1.09
        print(' ')
        print(f'E_gamma: {Egamma[0]/1000}+-{Egamma[1]/1000} [MeV]')
        print(f'T: {T[0]/1000}+-{T[1]/1000} [MeV]')
        deltaMNR=np.sqrt((2*(2*Egamma[0]-T[0])/T[0])**2*(Egamma[1])**2+((2*Egamma[0]-T[0])/T[0]-((2*Egamma[0]-T[0])/(2*T[0]))**2)**2*(T[1])**2)
        deltaMR=np.sqrt(((4*Egamma[0]-2*T[0])/T[0])**2*(Egamma[1])**2+(-2*Egamma[0]/T[0]-2*Egamma[0]*(Egamma[0]-T[0])/T[0]**2)**2*(T[1])**2)
        print(f'Masa del electrón No Relativista: {(2*Egamma[0]-T[0])**2/(2*T[0])/1000}+-{deltaMNR/1000/2} [MeV]')
        print(f'Masa del electrón Relativista: {2*Egamma[0]*(Egamma[0]-T[0])/T[0]/1000}+-{deltaMR/1000/2} [MeV]') 

cal=calib()
cal.add_med('137Cs_2T')
cal.add_med('133Ba_2')
cal.add_med('207Bi_1')
cal.plot_cal()

medicion=Med(cal.pendiente,cal.offset,cal.popterr[0],cal.popterr[1],cal.error_V)
medicion.add_med('137Cs_2T')
medicion.add_med('133Ba_2')
medicion.add_med('34Es_1')
medicion.add_med('207Bi_1')
medicion.plot_med()
'''
estad=Stat_gamma()
estad.add_med('137Cs_025s')
estad.add_med('137Cs_2T')
estad.add_med('137Cs_075sT')
estad.add_med('137Cs_1s')
estad.add_med('137Cs_125s')
estad.plot_med()
'''
comp=compton()
comp.masa_e(medicion.Egamma_p[0],medicion.T[0])
comp.masa_e(medicion.Egamma_p[3],medicion.T[3])
