import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class brownian():
    def __init__(self,rho,T):
        self.rho=rho
        self.T=T
    
    def walk(self,part,R,eta,iter,dt):
        lista=[]
        self.gamma=6*np.pi*eta*R
        print(f'gamma={self.gamma}')
        Kb=1.380649*1E-23
        m=self.rho*4/3*np.pi*R**3
        print(f'masa={m}')
        X=np.zeros((iter,2,part))
        V=np.zeros_like(X)
        A=np.zeros_like(X)
        X[0,:,:]=R*1E2*np.random.random_sample((1,2,part))
        plt.figure()
        plt.plot(X[0,0,:],X[0,1,:],'.')
        plt.pause(0.01)
        plt.cla()
        plt.draw()
        for j in range(1,iter):
            A[j,:,:]=-self.gamma/m*V[j-1,:,:]+np.sqrt(2*Kb*self.T*self.gamma)/m*np.random.standard_normal((1,2,part))
            V[j,:,:]=V[j-1,:,:]+A[j,:,:]*dt
            X[j,:,:]=X[j-1,:,:]+V[j,:,:]*dt
            plt.plot(X[j,0,:],X[j,1,:],'.')
            plt.xlim([0,R*1E2])
            plt.ylim([0,R*1E2])
            plt.pause(0.01)
            plt.cla()
            plt.draw()
        for h in range(part):
            lista.append(X[:,:,h])
        return lista

B=brownian(70,13)
lista_simu=B.walk(50,4.5*1E-6,0.4*1E-8,300,1/15)
