import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import covariance
import itertools
plt.ion()

def stack_padding(l):
    return np.column_stack((itertools.zip_longest(*l, fillvalue=0)))

def pross(image):
    # preprocess the image 
    gray_img = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) 

    gray_img = np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(gray_img))*np.where((X-image.shape[1]/2)**2+(Y-image.shape[0]/2)**2<40**2,0,1))))**2
    gray_img = (gray_img/gray_img.max()*255).astype('uint8')
    
    # Applying 7x7 Gaussian Blur 
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    # Applying threshold 
    threshold = cv2.threshold(blurred, 2, 255, cv2.THRESH_OTSU)[1]  
    threshold = cv2.dilate(threshold,(7, 7),iterations = 4)
    
    # Apply the Component analysis function 
    analysis = cv2.connectedComponentsWithStats(threshold, 
                                                4, 
                                                cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    
    # Initialize a new image to store  
    # all the output components 
    output = np.zeros(gray_img.shape, dtype="uint8") 
    dots=[[] for _ in range(totalLabels-1)]
    # Loop through each component 
    for i in range(1, totalLabels): 
        
        # Area of the component 
        area = values[i, cv2.CC_STAT_AREA]  
        
        if (area >= 2 and area < 1500): 
            #componentMask = (label_ids == i).astype("uint8") * 255
            #output = cv2.bitwise_or(output, componentMask) 
            dots[i-1].append(centroid[i])
    return dots, gray_img

el = covariance.EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None, 
                                    contamination=0.075, random_state=0)

vidcap = cv2.VideoCapture('Practica 3 Pinzas opticas\\mov browneano muestra2 05um\\01_muestra2_0.avi')
success,image = vidcap.read()
plt.figure()
X,Y=np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
frame=0
bar=tqdm(total=428,desc='Frames analizados')
while success:
    puntos,gray_img=pross(image)
    plt.imshow(image)
    X_ptos,Y_ptos=[],[]
    for k in range(len(puntos)):
        X_ptos.append(puntos[k][0][0])
        Y_ptos.append(puntos[k][0][1])
    if frame==0:
        Mx,My=np.c_[np.array(X_ptos),frame*np.ones(len(X_ptos))],np.c_[np.array(Y_ptos),frame*np.ones(len(Y_ptos))]
    else:
        Mx,My=np.vstack((Mx,np.c_[np.array(X_ptos),frame*np.ones(len(X_ptos))])),np.vstack((My,np.c_[np.array(Y_ptos),frame*np.ones(len(Y_ptos))]))
    plt.plot(Mx[:,0],My[:,0],'.',markersize=0.5)
    plt.pause(0.003)
    plt.cla()
    plt.draw()
    success,image = vidcap.read()
    bar.update(1)
    frame+=1

datos=np.c_[Mx[:,0],My[:,0]]
datos_cE=np.c_[Mx[:,0],My[:,0],My[:,1]]
el.fit(datos)
pred=el.predict(datos)
for j in range(len(datos[:,0])):
    if pred[j]<-0.1:
        np.delete(datos_cE,j,0)


#plt.figure()
#plt.plot(X_ptos,Y_ptos,'-',markersize=0.1)
#plt.show()