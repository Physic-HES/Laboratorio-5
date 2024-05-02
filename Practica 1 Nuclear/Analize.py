import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#estos datos junto con el nombre se deben aclarar de entrada para 
#que se recorran todos los archivos de interes con sus picos de energía
#declarados previamente en esta variable picosELin
#

names=['137Cs_1']
picosELin0=[[32.1,36.4,0,0,661.66]]
Vmaximos0=[[0.1668,0.4168,1.0722,2.1331,3.3993]]
lencortes0=[[7,10,30,50,25]]

linxV=[]
errlinxV=[]
linyE=[]
for NN in range(len(names)):
    print('Lectura archivo',names[NN])
    time0,picos0=np.loadtxt('Practica 1\\'+names[NN]+'_resultados.txt', skiprows=1, unpack=True, delimiter=',')
    time,picos=[],[]
    for i in range(len(picos0)):
        if picos0[i]>4.3:
            pass
        else:
            time.append(time0[i])
            picos.append(picos0[i])

    plt.figure(figsize=(10,6))
    plt.title('Amplitud de los picos', fontsize = 14)
    plt.hist(picos, bins=1000, label = 'data')
    plt.xlabel('Tensión [V]', fontsize = 14)
    plt.ylabel('Cuentas', fontsize = 14)
    plt.yscale("log")
    plt.legend(fontsize = 14)
    plt.savefig(names[NN]+' Crudo Histog.png')
    #plt.show()

    bindM=1000
    counts=np.zeros(bindM+1)
    bindStep=(max(picos)-min(picos))/bindM
    maxV=max(picos)
    minV=min(picos)
    picosH=np.linspace(minV,maxV,bindM)

    for i in range(len(picos)):
        counts[int((picos[i]-minV)/bindStep)]+=1

    counts=list(counts)
    counts.pop()

    picosELin=picosELin0[NN]
    Vmaximos=Vmaximos0[NN]
    lencortes=lencortes0[NN]
    colorS=['b','r','y','g','m','c']
    cortes={}
    for I in range(len(Vmaximos)):
        cortes[I]=[0,[],[]]
        maximo=Vmaximos[I]
        diff=1
        center=0
        icenter=0
        for i in range(len(counts)):
            if abs(maximo-picosH[i])<diff:
                center=picosH[i]
                icenter=i
                diff=abs(maximo-picosH[i])
        cortes[I][0]=icenter
        for n in range(2*lencortes[I]+1):
            cortes[I][1].append(picosH[icenter-lencortes[I]+n])
            cortes[I][2].append(counts[icenter-lencortes[I]+n])

    '''
    plt.figure()
    plt.title('Amplitud de los picos Crudo', fontsize = 14)
    plt.plot(picosH, counts,'-ob',ms=1, label = 'Data')
    plt.xlabel('Tensión [V]', fontsize = 14)
    plt.ylabel('Cuentas', fontsize = 14)
    plt.yscale("log")
    plt.legend(fontsize = 14)
    plt.show()
    '''

    plt.figure()
    plt.title('%s Amplitud de los picos Recortes'%names[NN], fontsize = 14)
    plt.plot(picosH, counts,'-oc',ms=1, label = 'Data')
    for I in range(len(Vmaximos)):
        plt.plot(cortes[I][1], cortes[I][2],'-o%s'%colorS[I],ms=1, label = 'Data recorte %i'%I)
    plt.xlabel('Tensión [V]', fontsize = 14)
    plt.ylabel('Cuentas', fontsize = 14)
    plt.yscale("log")
    plt.legend(fontsize = 14)
    #plt.show()

    def Gaussian(x,a,b,c,d):
        return d+c*np.exp(-(((x-a)/b)**2)/2)

    def Gaussian2(x,a,b):
        return (1/(b*np.sqrt(2*np.pi)))*np.exp(-((x-a)**2)/2*(b**2))

    ajustes={}
    LENTEOS=1000
    for I in range(len(Vmaximos)):
        ajustes[I]={}
        ajustes[I]['poptSuger']=[picosH[cortes[I][0]],(max(cortes[I][1])-min(cortes[I][1]))**2,max(cortes[I][2])-min(cortes[I][2]),min(cortes[I][2])]
        #print('Ajuste Suger corte %i'%I, '<x>',ajustes[I]['poptSuger'][0], 'ancho',ajustes[I]['poptSuger'][1], 'amplit',ajustes[I]['poptSuger'][2], 'Offset',ajustes[I]['poptSuger'][3])

        ajustes[I]['popt'], ajustes[I]['pcov']=curve_fit(Gaussian, cortes[I][1], cortes[I][2], p0=ajustes[I]['poptSuger'])
        ajustes[I]['errpopt']=np.sqrt(np.diag(ajustes[I]['pcov']))
        ajustes[I]['Xteo']=np.linspace(min(cortes[I][1]),max(cortes[I][1]),LENTEOS)
        ajustes[I]['Yteo']=[]
        ajustes[I]['YteoSuger']=[]
        for i in range(len(ajustes[I]['Xteo'])):
            ajustes[I]['Yteo'].append(Gaussian(ajustes[I]['Xteo'][i],ajustes[I]['popt'][0],ajustes[I]['popt'][1],ajustes[I]['popt'][2],ajustes[I]['popt'][3]))
            ajustes[I]['YteoSuger'].append(Gaussian(ajustes[I]['Xteo'][i],ajustes[I]['poptSuger'][0],ajustes[I]['poptSuger'][1],ajustes[I]['poptSuger'][2],ajustes[I]['poptSuger'][3]))
        print('Ajuste corte %i'%I, '<x>',round(ajustes[I]['popt'][0],4),'+-',round(ajustes[I]['errpopt'][0],4))#, 'ancho',round(ajustes[I]['popt'][1],3), 'amplit',round(ajustes[I]['popt'][2],3), 'Offset',round(ajustes[I]['popt'][3],3))

    plt.figure(figsize=(10,6))
    plt.title('%s Amplitud Ajustes y Recortes'%names[NN], fontsize = 14)
    plt.plot(picosH, counts,'-oc',ms=1, label = 'Data')
    for I in range(len(Vmaximos)):
        plt.plot(cortes[I][1], cortes[I][2],'o%s'%colorS[I],ms=2, label = 'Data recorte %i'%I)
        plt.plot(ajustes[I]['Xteo'], ajustes[I]['Yteo'],'-%s'%colorS[I], label = 'Ajuste recorte %i'%I)
    plt.xlabel('Tensión [V]', fontsize = 14)
    plt.ylabel('Cuentas', fontsize = 14)
    plt.yscale("log")
    plt.legend(fontsize = 9)
    plt.savefig(names[NN]+'_Recortes_ajustes.png')
    #plt.show()


    for i in range(len(picosELin)):
        if picosELin[i]==0:
            pass
        else:
            linxV.append(round(ajustes[i]['popt'][0],4))
            errlinxV.append(round(ajustes[i]['errpopt'][0],4))
            linyE.append(picosELin[i])


def Linear(x,a,b):
    return a*x+b

bsug=0
for i in range(len(linyE)):
    bsug=linyE[i]-(max(linyE)-min(linyE))/(max(linxV)-min(linxV))*linxV[i]
bsug=bsug/len(linyE)
poptSuger=[(max(linyE)-min(linyE))/(max(linxV)-min(linxV)),bsug]

popt, pcov=curve_fit(Linear, linxV, linyE, p0=poptSuger)
errpopt=np.sqrt(np.diag(pcov))
Xteo=np.linspace(min(linxV),max(linxV),LENTEOS)
Yteo=[]
for i in range(len(Xteo)):
    Yteo.append(Linear(Xteo[i],popt[0],popt[1]))

plt.figure(figsize=(10,6))
plt.title('Ajuste Calibración', fontsize = 14)
plt.plot(linxV, linyE,'ob',ms=2, label = 'Data')
plt.plot(Xteo, Yteo,'-r', label = 'Ajuste')
plt.xlabel('Tensión [V]', fontsize = 14)
plt.ylabel('Energía [MEv]', fontsize = 14)
plt.legend(fontsize = 9)
plt.savefig('Ajuste Calibración.png')
plt.show()

print('Ajuste Calibración', 'pendiente (',round(popt[0],4),'+-',round(errpopt[0],4), ') - Offset  (',round(popt[1],4), '+-',round(errpopt[1],4), ')')
