import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from typing import List

import scipy.cluster.hierarchy as cluster
from scipy.spatial.distance import pdist
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from moviepy.video.io.VideoFileClip import VideoFileClip

import seaborn as sb
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gs
import pandas as pd
import scipy.stats as st

import uncertainties as u


def mkdir(path, *args, **kwargs):
    try:
        os.makedirs(path, *args, **kwargs)
    except FileExistsError:
        pass


def get_first_frame(video_file):
    capture = cv.VideoCapture(video_file)
    _, frame = capture.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return frame


def cv_draw_text(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0,
               (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN,
               1.0, (255, 255, 255), lineType=cv.LINE_AA)


def query_video_properties(video_file):
    capture = cv.VideoCapture(video_file)
    result = {}
    result['n_frames'] = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    result['fps'] = capture.get(cv.CAP_PROP_FPS)
    result['width'] = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    result['height'] = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    return result


def track_points(video_file, quality_level=0.2, max_points=500, screen_region=None, tolerance=5):
    capture = cv.VideoCapture(video_file)
    n_frames = query_video_properties(video_file)['n_frames']
    print(n_frames)
    success, prev_frame = capture.read()
    print(prev_frame.shape)
    if not success:
        print("Error al leer el primer frame del video")
        return None
    prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    prev_points = cv.goodFeaturesToTrack(
        prev_frame_gray, maxCorners=500, minDistance=7, blockSize=7, qualityLevel=quality_level)
    n_points = len(prev_points)
    line_colors = np.random.randint(0, 255, (n_points, 3))
    mask = np.zeros_like(prev_frame)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    point_holder = [prev_points[:, 0]]
    result = []
    for i in range(n_frames):
        # Agarrar el proximo frame
        success, frame = capture.read()
        if not success:
            print(f"Error al leer el frame nro. {i} del video")
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        points, _, _ = cv.calcOpticalFlowPyrLK(
            prev_frame_gray, frame_gray, prev_points, None, **lk_params)
        points_reverse, _, _ = cv.calcOpticalFlowPyrLK(
            frame_gray, prev_frame_gray, points, None, **lk_params)
        status = abs(points-points_reverse).reshape(-1, 2).max(-1) < tolerance
        tracked_points = points[status]
        prev_tracked_points = prev_points[status]
        line_colors = line_colors[status]
        for j, (current, prev) in enumerate(zip(tracked_points, prev_tracked_points)):
            a, b = current.ravel()
            c, d = prev.ravel()
            mask = cv.line(mask, (int(a), int(b)),
                           (int(c), int(d)), line_colors[j].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)),
                              5, line_colors[j].tolist(), -1)
        image = cv.add(frame, mask)
        cv_draw_text(image, (20, 20),
                     f'tracked points: {len(tracked_points)}')
        #cv.rectangle(image, (int(1248*3/5-150-40),int(1024*2/3-100)), (int(1248*3/5+150),int(1024*2/3+50)), color=(255,0,0), thickness=15) 
        cv.imshow("track_points", image)
        cv.waitKey(1)
        prev_frame_gray = frame_gray.copy()
        prev_points = tracked_points.reshape(-1, 1, 2)
        if not all(status):
            temp = np.array(point_holder)
            temp = np.swapaxes(temp, 0, 1)
            temp = temp[status == False]
            temp = np.swapaxes(temp, 1, 2)
            result += temp.tolist()
            point_holder = np.array(point_holder)[:, status].tolist()
        else:
            point_holder.append(points[:, 0])
    cv.destroyAllWindows()
    point_holder = np.array(point_holder)
    point_holder = np.swapaxes(point_holder, 0, 1)
    point_holder = np.swapaxes(point_holder, 1, 2)
    result += point_holder.tolist()
    return result

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


def graficar(file,path,R):
    eta_eff=[]
    for k in range(len(file)):
        ruta = path+file[k]
        print(ruta)
        lista = track_points(ruta, quality_level=0.10, max_points=100, screen_region=None, tolerance=20)
        
        tiempo, SMD, eta, part=smd(lista,15,R)
        print(f'Viscosidad: eta={eta}')
        plt.figure(1)
        lin_coef=np.polyfit(tiempo,SMD,1)
        plt.plot(tiempo,(lin_coef[0]*tiempo+lin_coef[1])*1E12,'-k',linewidth=0.75)
        plt.plot(tiempo,SMD*1E12,label=file[k]+r' - $\eta_{eff}=$'+f'{eta:.2} Ns/m')
        plt.ylabel(r'SMD [$\mu m^2$]')
        plt.xlabel('Tiempo [s]')
        plt.legend()
        plt.figure(k+2)
        plt.plot(np.array(lista[part][0])*9.375*1E-2,np.array(lista[part][1])*9.375*1E-2,'-',label=f'Particula {part} '+r'Máximo $r^2$')
        plt.plot(np.array(lista[part][0])*9.375*1E-2,np.array(lista[part][1])*9.375*1E-2,'.k',markersize=1.5)
        plt.ylabel(r'Desplazamiento Y [$\mu m$]')
        plt.xlabel(r'Desplazamiento X [$\mu m$]')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title(file[k])
        eta_eff.append(eta)
    plt.show()
    return eta_eff


def fuerza(lista,fps,R,eta,tam_im):
    cant=[]
    for k in range(len(lista)):
        cant.append(len(lista[k][0]))
    v=np.zeros((np.max(cant)-1,len(lista)))
    vel=np.zeros(np.max(cant)-1)
    vel_pin=np.zeros_like(vel)
    ind_pin=[]
    ind_unpin=[]
    for k in range(len(lista)):
        x,y=np.array(lista[k][0])*9.375*1E-8,np.array(lista[k][1])*9.375*1E-8
        v[:len(lista[k][0])-1,k]=np.diff(np.sqrt((x-x[0])**2+(y-y[0])**2))*fps
        if (np.abs(np.array(lista[k][1])-(tam_im[0]*2/3-25)).max()<75 and np.abs(np.array(lista[k][0])-(tam_im[1]*3/5-20)).max()<170):
            ind_pin.append(k)
        else:
            ind_unpin.append(k)
    print(ind_pin)
    N1_=[]
    for m in ind_pin:
        N1_.append(np.count_nonzero(v[:,m]))
    N1=np.max(N1_)
    N2_=[]
    for m in ind_unpin:
        N2_.append(np.count_nonzero(v[:,m]))
    N2=np.max(N2_)
    for j in range(1,N1):
        vel[j-1] = 1/np.count_nonzero(v[j,ind_unpin])*np.sum(v[j,ind_unpin])
    for j in range(1,N2):
        vel_pin[j-1] = 1/np.count_nonzero(v[j,ind_pin])*np.sum(v[j,ind_pin])
    ind_acel=np.argmax(np.diff(vel_pin)*1/fps)
    fuer=6*np.pi*eta*R*vel[ind_acel]
    tiempo = (np.arange(len(lista[k][0])-1)+1)*1/fps
    plt.plot(tiempo,np.abs(vel)*1E6,label=r'$V_{prom}$ Particulas libres')
    plt.plot(tiempo,np.abs(vel_pin)*1E6,label=r'$V$ Particula sujeta')
    plt.plot(tiempo[ind_acel],np.abs(vel[ind_acel])*1E6,'.r',label=r'$V$ para calcular $F$')
    plt.legend()
    return fuer


def Fuerza_Stoks(file,path,R,eta):
    F=[]
    for k in range(len(file)):
        ruta = path+file[k]
        lista = track_points(ruta, quality_level=0.10, max_points=100, screen_region=None, tolerance=20)

        plt.figure(k+1)
        plt.ylabel(r'Velocidad [$\mu m/s$]')
        plt.xlabel(r'Tiempo [$s$]')
        plt.title(file[k])

        fuer = fuerza(lista,15,R,eta,[1024,1280])
        F.append(fuer)
        print(f'Fuerza={F[-1]*1E12:.3} pN')
    print(f'Fuerza Promedio = {np.mean(F)*1E12:.3} +- {np.std(F)*1E12:.2} pN')
    plt.show()
    return F
        
# Muestras con silica de 0.5 micrones
#path='/home/hugo_sosa/Documents/L5/Laboratorio-5/Practica 3 Pinzas opticas/mov browneano muestra2 05um/'
#file=['01_muestra2_0.avi','01_muestra2_1.avi','01_muestra2_2.avi','01_muestra2_3.avi']
#eta_05=graficar(file,path,0.5E-6)

# Muestras con latex de 4.5 micrones
path='/home/hugo_sosa/Documents/L5/Laboratorio-5/Practica 3 Pinzas opticas/mov_browneano_agua_45um/'
#file=['01_muestra0_0.avi','01_muestraB_0.avi','01_muestraC_0.avi','01_muestraD_0.avi','01_muestraE_0.avi']
file=['01_muestraE_0.avi']
eta_45=graficar(file,path,4.5E-6)

# Muestras con latex de 4.5 micrones al sujetar con pinza optica
path='/home/hugo_sosa/Documents/L5/Laboratorio-5/Practica 3 Pinzas opticas/'
file=['Arrastre_muestraE_1.avi','Arrastre_muestraE_2.avi','Arrastre_muestraE_3.avi','Arrastre_muestraE_4.avi']
F=Fuerza_Stoks(file,path,4.5E-6,eta_45[-1])

