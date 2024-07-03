import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

def smd(lista,fps,R):
    cant=[]
    for k in range(len(lista)):
        cant.append(len(lista[k][0]))
    r_s=np.zeros((np.max(cant),len(lista)))
    SMD=np.zeros(len(r_s[:,0]))
    for k in range(len(lista)):
        x,y=np.array(lista[k][0])*9.375*1E-8,np.array(lista[k][1])*9.375*1E-8
        r_s[:len(lista[k][0]),k]=np.cumsum((x-x[0])**2+(y-y[0])**2)/(np.arange(len(lista[k][0]))+1)
    for k in range(3,len(r_s[:,0])):
        #print(k,np.max(cant),r_s[k,:])
        SMD[k]=1/np.count_nonzero(r_s[k,:])*np.sum(r_s[k,:])
    tiempo=(np.arange(np.max(cant))+1)*1/fps
    lin_coef=np.polyfit(tiempo,SMD,1)
    eta=2*1.38E-23*(273.15+13)/(3*np.pi*lin_coef[0]*R)
    part_rsmax=np.argmax(r_s[-1,:])
    return tiempo, SMD, eta, part_rsmax

class brownian():
    def __init__(self,rho,T):
        self.rho=rho
        self.T=T
        self.cont=0
    
    def walk(self,part,R,eta,iter,dt,pl=1):
        plt.ion()
        lista=[]
        self.gamma=6*np.pi*eta*R
        #print(f'gamma={self.gamma}')
        Kb=1.380649*1E-23
        m=self.rho*4/3*np.pi*R**3
        #print(f'masa={m}')
        F_r=np.sqrt(2*Kb*self.T*self.gamma)
        #print(f'F_r/m={F_r/m}')
        X=np.zeros((iter,2,part))
        V=np.zeros_like(X)
        A=np.zeros_like(X)
        X[0,0,:],X[0,1,:]=1280*9.375*1E-8*np.random.random_sample((1,part)),1024*9.375*1E-8*np.random.random_sample((1,part))
        if pl==1:
            plt.figure(1)
            plt.scatter(X[0,0,:]*1E3,X[0,1,:]*1E3,s=R*np.ones_like(X[0,1,:])*1E6)
            plt.xlabel('X [mm]')
            plt.ylabel('Y [mm]')
            plt.title(f'Simulación {self.cont}')
            plt.pause(1/15)
            plt.cla()
            plt.draw()
        for j in range(1,iter):
            A[j,:,:]=-self.gamma/m*V[j-1,:,:]+F_r/m*np.random.standard_normal(size=(1,2,part))
            V[j,:,:]=V[j-1,:,:]+A[j,:,:]*dt
            X[j,:,:]=X[j-1,:,:]+V[j,:,:]*dt
            if pl==1:
                plt.scatter(X[j,0,:]*1E3,X[j,1,:]*1E3,s=R*np.ones_like(X[0,1,:])*1E6)
                plt.xlim([0,1280*9.375*1E-8*1E3])
                plt.ylim([0,1024*9.375*1E-8*1E3])
                plt.xlabel('X [mm]')
                plt.ylabel('Y [mm]')
                plt.title(f'Simulación {self.cont}')
                plt.pause(1/15)
                plt.cla()
                plt.draw()
        for h in range(part):
            med=[]
            med.append(X[:,0,h]/(9.375*1E-8))
            med.append(X[:,1,h]/(9.375*1E-8))
            lista.append(med)
        plt.ioff()
        return lista
    
    def contador(self):
        self.cont=+1

def fun_lin(tiempo,p0,p1):
    return p0*tiempo+p1*tiempo**2

B=brownian(3055,25+273.4)
ETA=[]
SMD_lista=[]
COEF=[]
simu=[]
PART=[]
iteraciones=2500
pbar=tqdm(total=iteraciones,desc=f'Integrando simulaciones')
for hh in range(iteraciones):
    lista_simu=B.walk(25,7.3/2*1E-5,1.5*1E-5,405,1/15,pl=0)
    tiempo, SMD, eta, part=smd(lista_simu,15,4.5*1E-6)
    #print(f'Viscosidad: eta={eta}')
    ETA.append(eta)
    SMD_lista.append(SMD)
    simu.append(lista_simu)
    PART.append(part)
    lin_coef=np.polyfit(tiempo,SMD,1)
    lin_coef,lin_coef_=curve_fit(fun_lin,tiempo,SMD,p0=[0.003,0])
    perr=np.sqrt(np.diag(lin_coef_))
    COEF.append(lin_coef)
    B.contador()
    pbar.update(1)

for k in range(len(SMD_lista)):
    plt.figure(1)
    #plt.plot(tiempo,(COEF[k][0]*tiempo+COEF[k][1]*tiempo**2)*1E12,'-k',linewidth=0.75)
    plt.plot(tiempo,SMD_lista[k]*1E12,label=f'Simulación {k}'+r' - $\eta_{eff}=$'+f'{ETA[k]:.2} $Ns/m^2$')
    plt.ylabel(r'SMD [$\mu m^2$]')
    plt.xlabel('Tiempo [s]')
    #plt.legend()
    plt.figure(2)
    plt.plot(np.array(simu[k][PART[k]][0])*9.375*1E-8*1E6,np.array(simu[k][PART[k]][1])*9.375*1E-8*1E6,'-',label=f'Particula {PART[k]} Simu {k}'+r' Máximo $r^2$')
    plt.plot(np.array(simu[k][PART[k]][0])*9.375*1E-8*1E6,np.array(simu[k][PART[k]][1])*9.375*1E-8*1E6,'.k',markersize=0.25)
    plt.ylabel(r'Desplazamiento Y [$\mu m$]')
    plt.xlabel(r'Desplazamiento X [$\mu m$]')
    plt.title('Simulación')
    plt.gca().set_aspect('equal')
    #plt.legend()

plt.figure(3)
plt.hist(ETA,bins=100)
plt.xlabel(r'$\eta_{eff} [Ns/m^2]$')
plt.show()
