#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:42:02 2019

@author: tramananya
"""

import numpy as np
from matplotlib.path import Path
import cv2
import matplotlib.pyplot as plt
import math 

def Gaussianfilter(sigma):
        
    filter_size = 2 * int(4 * sigma + 0.5) + 1
#    filter_size = int(4*sigma+0.5)
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    return gaussian_filter


def pad_3by3(pad_type,imh,imw,grey_img,pad):
    grey_img_padded=np.array([[]])
    if pad_type in ['zero']:
        
            grey_img_padded=np.zeros((imh+(2*pad),imw+(2*pad)),dtype="float32")
            grey_img_padded[pad:imh+pad,pad:imw+pad]=grey_img
            grey_img_padded[0:pad,:]=grey_img_padded[:,0:pad]=grey_img_padded[imh+pad:imh+pad+pad,:]=grey_img_padded[:,imw+pad:imh+pad+pad]=0
                
         
            
        
    if pad_type in ['wrap']:
        
        
            grey_img_padded=np.zeros((imh+(2*pad),imw+(2*pad)),dtype="float32")
            grey_img_padded[pad:imh+pad,pad:imw+pad]=grey_img
            
            grey_img_padded[0:pad,pad:imw+pad]=grey_img[imh-pad:imh,:]
            grey_img_padded[pad:imh+pad,0:pad]=grey_img[:,imw-pad:imw]
            grey_img_padded[imh+pad:imh+pad+pad,pad:imw+pad]=grey_img[0:pad,:]
            grey_img_padded[pad:imh+pad,imw+pad:imw+pad+pad]=grey_img[:,0:pad]
            
            
        
    if pad_type in ['copy edge']:
        
        
            grey_img_padded=np.zeros((imh+(2*pad),imw+(2*pad)),dtype="float32")
            grey_img_padded[pad:imh+pad,pad:imw+pad]=grey_img
            
            grey_img_padded[0:pad,pad:imw+pad]=grey_img[0:pad,:]
            grey_img_padded[pad:imh+pad,0:pad]=grey_img[:,0:pad]
            grey_img_padded[imh+pad:imh+pad+pad,pad:imw+pad]= grey_img[imh-pad:imh,:]
            grey_img_padded[pad:imh+pad,imw+pad:imw+pad+pad]= grey_img[:,imw-pad:imw]
            
            
        
    if pad_type in ['reflect across edge']:
        
        
            grey_img_padded=np.zeros((imh+(2*pad),imw+(2*pad)),dtype="float32")
            grey_img_padded[pad:imh+pad,pad:imw+pad]=grey_img
            
            grey_img_padded[0:pad,pad:imw+pad]=grey_img[pad-1:pad,:]
            grey_img_padded[pad:imh+pad,0:pad]=grey_img[:,pad-1:pad]
            grey_img_padded[imh+pad:imh+pad+pad,pad:imw+pad]= grey_img[imh-1:imh,:]
            grey_img_padded[pad:imh+pad,imw+pad:imw+pad+pad]= grey_img[:,imw-1:imw]
            
            
      
    return grey_img_padded




def conv2(grey_img,w,pad_type): 
    
    imh,imw=grey_img.shape[:2]
    
    
    
    kernel=w
    kerh,kerw=kernel.shape[:2]
    
    output = np.zeros((imh, imw), dtype="float32")
    
    pad = (kerw - 1) // 2

    
    grey_img_padded=pad_3by3(pad_type,imh,imw,grey_img,pad)
    
    
    
    for h in range(pad,imh+pad):
        for w in range(pad,imw+pad):
            temp=grey_img_padded[h-pad:h+pad+1,w-pad:w+pad+1]
            mul=(temp*kernel).sum()
            output[h-pad,w-pad]=mul      
                
        
    return output

def upscale(image,new_imh,new_imw):
    
    
    
    
    imh,imw=image.shape[:2]
    
    
    new_h=new_imh
    new_w=new_imw
    
    hratio=int(new_h/imh)
    wratio=int(new_w/imw)
    
    
    
    if (new_h/imh)>2:
        new_rows=(np.arange(0,(imh*hratio)+1))
        new_rows[new_h-1]=new_h-2
    else:
        new_rows=(np.arange(0,(imh*hratio)))
        
    new_rows=(np.floor(np.divide(new_rows,(hratio))))
    new_rows=[int(x) for x in new_rows]
    new_rows=np.array(new_rows)
    
    
    
    if (new_w/imw)>2:
        new_cols=(np.arange(0,(imw*wratio)+1))
        new_cols[new_w-1]=new_w-2
    else:
        new_cols=(np.arange(0,(imw*wratio)))
        
    new_cols=(np.floor(np.divide(new_cols,(wratio))))
    new_cols=[int(x) for x in new_cols]
    new_cols=np.array(new_cols)
    
    #print(new_rows)
    upscaled = np.zeros((new_w, new_h), dtype="uint8")
    temp=image
    upscaled=temp[new_rows,:]
    upscaled=upscaled[:,new_cols]
    return upscaled

def downscale(image,layer):
    imh,imw=image.shape[:2]
    new_w=int(imw/layer)
    new_h=int(imh/layer)
    downscaled = np.zeros((imh,imw), dtype="uint8")
    
    
    
    for i in range(imh):
        
        
        if i == 0:
            downscaled[i,:]=image[i,:]
            i=i+1
            
        else:
            
            downscaled[int(i/layer),:]=image[i,:]
            i=i+1
            
        
    for j in range(imw):
        if j== 0:
            downscaled[:,j]=downscaled[:,j]
            j=j+1
        else:
            
            downscaled[:,int(j/layer)]=downscaled[:,j]
            j=j+1
        
    for j in range(new_h,imw):
        downscaled[:,j]=0
    
    
#    plt.figure()
#    plt.imshow(downscaled,cmap='gray')
    
    

    return downscaled[0:new_h,0:new_w]
    
def build_gaussian(grey_img,gaussian,num_layers,w,pad_type):
    output=grey_img
    
    gaussian[0]=grey_img
    for x in range(1,(num_layers+1)):
        output=conv2(output,w,pad_type) # convolution function
        for j in range(output.shape[0]):
            for k in range(output.shape[1]):
                  if output[j,k] < 0:      
                      output[j,k]=0
                  if output[j,k] > 255:
                      output[j,k]=255           
        
        output=downscale(output,(2))
        gaussian[x]=output
    return gaussian
    
def build_laplacian(laplacian,gaussian,num_layers,w,pad_type):
     
    for x in range(0,(num_layers)):
        new_imh,new_imw=gaussian[x].shape[:2]
        temp_var=upscale(gaussian[x+1],new_imh,new_imw)
        
        
        
        temp_var=conv2(temp_var,w,pad_type)
        
        
        
        laplacian[x]=gaussian[x]-temp_var
    return laplacian
#def selectroi_rect():
#    r = cv2.selectROI(grey_img,False,False)
#    imCrop = grey_img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#    y=grey_img
#    y1=int(r[1])
#    x1=int(r[0])
#    y2=int(r[1]+r[3])
#    x2=int(r[0]+r[2])
#    for i in range(0,y1):
#        for j in range(0,imw):
#             y[i,j]=0
#            
#    for i in range(y2,imh):
#          for j in range(0,imw):
#              y[i,j]=0
#            
#    for i in range(0,imh):
#         for j in range(0,x1):
#             y[i,j]=0
#            
#    for i in range(0,imh):
#         for j in range(x1,imw):
#             y[i,j]=0
#                
#    for i in range(y1,y2):
#         for j in range(x1,x2):
#            y[i,j]=255
#    return y

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    global start
    global stop
    global img_formask
    
    if (event == cv2.EVENT_LBUTTONDOWN) and (start==0):
        cv2.circle(img_formask,(x,y),1,(255,0,0),-1)
        mouseX,mouseY = x,y
        coordinates.append([x,y])
        
        start=1
    elif (event == cv2.EVENT_MOUSEMOVE) and (start==1):
        cv2.circle(img_formask,(x,y),1,(255,0,0),-1)
        mouseX,mouseY = x,y
        coordinates.append([x,y])
      #  print("hellosir")
    
    elif (event == cv2.EVENT_LBUTTONDOWN) and (start==1):
        cv2.circle(img_formask,(x,y),1,(255,0,0),-1)
        mouseX,mouseY = x,y
        coordinates.append([x,y])
        
        stop=1
      #  print('hisir')
def selectroi(img2):
    global img_formask
    global start
    global coordinates 
    global stop
    global grayscale
    
        
    img_formask = img2
#    img_formask = cv2.imread('Eyee.png',-1)
    if grayscale == 1:
        img_formask=cv2.cvtColor(img_formask,cv2.COLOR_BGR2GRAY)
    
        
    coordinates=[]         
    
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    
    
    
    while(1):
        
        cv2.imshow('image',img_formask)
        k = cv2.waitKey(20) & 0xFF
        if (k == 27) or (stop==1) :
          #  print('i stopped sir')
            cv2.destroyAllWindows()
            break

    
    
                
    coordinates=np.asarray(coordinates)
    
    path = Path(coordinates)
    xmin, ymin, xmax, ymax = np.asarray(path.get_extents(), dtype=int).ravel()
    
    
    x, y = np.mgrid[:img.shape[1], :img.shape[0]]
    
    points = np.vstack((x.ravel(), y.ravel())).T
    
    
    mask = path.contains_points(points)
    path_points = points[np.where(mask)]
    
    
    img_mask = mask.reshape(x.shape).T
    
    img_mask1=img1 * img_mask[..., None]
    img_mask2=img_mask1
    if grayscale==1:
        limit=1
    else:
    
        limit=3
        
    for k in range(0,limit):
        
        for i in range(0,img_mask2.shape[0]):
            for j in range(0,img_mask2.shape[1]):
                if img_mask1[i,j,k] != 0:
                    img_mask2[i,j,k]=255
                else:
                    img_mask2[i,j,k]=0    
    

    return img_mask2
    
    
def blend_images(blended,gaussian_mask,laplacian,lalacian1,w,pad_type,num_layers):
    
    for x in range(0,(num_layers+1)):
        blended[x]=((gaussian_mask[x]/255)*laplacian1[x])+(((255-gaussian_mask[x])/255)*laplacian[x])
    
    for x in range(num_layers,0,-1):
        new_imh,new_imw=blended[x-1].shape[:2]
        tmp=conv2(upscale(blended[x],new_imh,new_imw),w,pad_type)
        tmp=tmp+blended[x-1]
        blended[x-1]=tmp
    blended[0][blended[0]<0]=0
    blended[0][blended[0]>255]=255
    
    return blended

def create_image(img,firstpixelx,firstpixely):
    new_h,new_w=img.shape[:2]
    new_img=np.zeros((imh, imw,3), dtype="uint8")
    
    new_img[firstpixelx:firstpixelx+new_h,firstpixely:firstpixely+new_w]=img[:,:,0:3]
    return new_img
            



start=0
stop=0
grayscale=1

run_ex=int(input('Do you want to run examples?')) #1 or 0
if run_ex == 1:
    ex_no=int(input('Which example you want?')) # 1 or 2 or 3
else:
    ex_no=0


if ex_no ==1:
    
    
    img1 = cv2.imread('face.png',-1) #foreground image for blending
    img2 = cv2.imread('face.png',-1) #foreground image dedicated for mask
    img_formask=img2  #dummy initilization
    img=cv2.imread('hand.png',-1) #backgroud image
    image_type=input('image type:') # 'color' or 'gray'
    pad_type=input('pad type:')  # 'zero' or 'copy edge' or 'reflect across edge' or  'wrap'
    num_layers=int(input('no of layers')) 
    
    
    imh,imw=img.shape[:2]
    imh1,imw1=img.shape[:2]
     

if ex_no ==2 :
    img1 = cv2.imread('emma.png',-1) #foreground image for blending
    img2 = cv2.imread('emma.png',-1) #foreground image dedicated for mask
    img_formask=img2  #dummy initilization
    img=cv2.imread('body.png',-1) #backgroud image
    image_type=input('image type:') # 'color' or 'gray'
    pad_type=input('pad type:')  # 'zero' or 'copy edge' or 'reflect across edge' or  'wrap'
    num_layers=int(input('no of layers')) 
    
    
    imh,imw=img.shape[:2]
    imh1,imw1=img.shape[:2]
    img1=create_image(img1,12 ,210)
    img2=create_image(img2,12,210) 

if ex_no == 3:
    img1 = cv2.imread('yoga.png',-1) #foreground image for blending
    img2 = cv2.imread('yoga.png',-1) #foreground image dedicated for mask
    img_formask=img2  #dummy initilization
    img=cv2.imread('water.png',-1) #backgroud image
    image_type=input('image type:') # 'color' or 'gray'
    pad_type=input('pad type:')  # 'zero' or 'copy edge' or 'reflect across edge' or  'wrap'
    num_layers=int(input('no of layers')) 
    
    
    imh,imw=img.shape[:2]
    imh1,imw1=img.shape[:2]
    img1=create_image(img1,10,10)
    img2=create_image(img2,10,10) 
    
if ex_no ==4:
    img1 = cv2.imread('beanface.png',-1) #foreground image for blending
    img2 = cv2.imread('beanface.png',-1) #foreground image dedicated for mask
    img_formask=img2  #dummy initilization
    img=cv2.imread('selfie.png',-1) #backgroud image
    image_type=input('image type:') # 'color' or 'gray'
    pad_type=input('pad type:')  # 'zero' or 'copy edge' or 'reflect across edge' or  'wrap'
    num_layers=int(input('no of layers')) 
    
    
    imh,imw=img.shape[:2]
    imh1,imw1=img.shape[:2]
    img1=create_image(img1,100,60)
    img2=create_image(img2,100,60) 

if run_ex ==0:
    
    img1 = cv2.imread('face.png',-1) #foreground image for blending
    img2 = cv2.imread('face.png',-1) #foreground image dedicated for mask
    img_formask=img2  #dummy initilization
    img=cv2.imread('hand.png',-1) #backgroud image
    image_type=input('image type:') # 'color' or 'gray'
    pad_type=input('pad type:')  # 'zero' or 'copy edge' or 'reflect across edge' or  'wrap'
    num_layers=int(input('no of layers')) 
    
    
    imh,imw=img.shape[:2]
    imh1,imw1=img.shape[:2]


sigma=3
w=Gaussianfilter(sigma) 

max_num_layers=int(np.floor(math.log(imh,2)))-2
if num_layers>max_num_layers:
    num_layers=max_num_layers
    
gaussian = [[] for i in range(num_layers+1)]
laplacian=[[] for i in range(num_layers+1)]


gaussian1 = [[] for i in range(num_layers+1)]
laplacian1=[[] for i in range(num_layers+1)]


gaussian_mask = [[] for i in range(num_layers+1)]
gaussian_maskr = [[] for i in range(num_layers+1)]
gaussian_maskg= [[] for i in range(num_layers+1)]
gaussian_maskb = [[] for i in range(num_layers+1)]
blended = [[] for i in range(num_layers+1)]





if image_type in ['gray']:
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)       

    gaussian=build_gaussian(grey_img,gaussian,num_layers,w,pad_type)
    gaussian1=build_gaussian(grey_img1,gaussian1,num_layers,w,pad_type)                 

    laplacian=build_laplacian(laplacian,gaussian,num_layers,w,pad_type)  
    
    laplacian1=build_laplacian(laplacian1,gaussian1,num_layers,w,pad_type)
#        plt.figure()
#        plt.imshow(laplacian[x],cmap='gray')  
     
        
    laplacian[num_layers]=gaussian[num_layers]    
    laplacian1[num_layers]=gaussian1[num_layers]    


#    y=selectroi_rect()
    y=selectroi(img2)
    
    
#    output_mask=y
    gaussian_mask[0]=y[:,:,0] 
    
    gaussian_mask=build_gaussian(y[:,:,0],gaussian_mask,num_layers,w,pad_type)       

    
    blended=blend_images(blended,gaussian_mask,laplacian,laplacian1,w,pad_type,num_layers)

#     
        
    plt.figure()
#    blended[0]= cv2.medianBlur(blended[0], 1)
    blended[0].astype(np.uint8)
    plt.imshow(blended[0],cmap='gray')    
    
 
if image_type in ['color']:
    grayscale=0
    y=selectroi(img2)
   
    blended_image_rgb = np.zeros((imh, imw,3), dtype="uint8")

    gaussian_maskb[0]=y[:,:,0]
    gaussian_maskg[0]=y[:,:,1]
    gaussian_maskr[0]=y[:,:,2]
    
#    gaussian_mask[0]=y[:,:,0]
#    gaussian_mask=build_gaussian(y[:,:,0],gaussian_mask,num_layers,w,pad_type) 
    gaussian_maskb=build_gaussian(y[:,:,0],gaussian_maskb,num_layers,w,pad_type)  
    gaussian_maskg=build_gaussian(y[:,:,1],gaussian_maskg,num_layers,w,pad_type)
    gaussian_maskr=build_gaussian(y[:,:,2],gaussian_maskr,num_layers,w,pad_type)
    for k in range(0,3):
        
        gaussian = [[] for i in range(10)]
        laplacian=[[] for i in range(10)]

        blended = [[] for i in range(10)]
        gaussian1 = [[] for i in range(10)]
        laplacian1=[[] for i in range(10)]
        
        
        grey_img=img[:,:,k]
        grey_img1=img1[:,:,k]
        gaussian=build_gaussian(grey_img,gaussian,num_layers,w,pad_type)
        gaussian1=build_gaussian(grey_img1,gaussian1,num_layers,w,pad_type)
        laplacian=build_laplacian(laplacian,gaussian,num_layers,w,pad_type)  
    
        laplacian1=build_laplacian(laplacian1,gaussian1,num_layers,w,pad_type)
        laplacian[num_layers]=gaussian[num_layers]    
        laplacian1[num_layers]=gaussian1[num_layers]
        
        if k==0:
            
            blended=blend_images(blended,gaussian_maskb,laplacian,laplacian1,w,pad_type,num_layers)
            blended_image_rgb[:,:,k]=blended[0]
        if k==1:
            blended=blend_images(blended,gaussian_maskg,laplacian,laplacian1,w,pad_type,num_layers)
            blended_image_rgb[:,:,k]=blended[0]
        if k==2:
            blended=blend_images(blended,gaussian_maskr,laplacian,laplacian1,w,pad_type,num_layers)
            blended_image_rgb[:,:,k]=blended[0]
    
    plt.figure()
    blended_image_rgb=np.asarray(blended_image_rgb,dtype='uint8')
    
    blended_image_rgb=cv2.cvtColor(blended_image_rgb, cv2.COLOR_BGR2RGB)
#    blended_image_rgb= cv2.medianBlur(blended_image_rgb, 3)
    plt.imshow(blended_image_rgb) 
    
    
    
    
    
    
    
    