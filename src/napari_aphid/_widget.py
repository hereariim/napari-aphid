from tkinter import Image
from tkinter.ttk import Progressbar
from tifffile import imsave
import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage import img_as_uint, measure
from skimage.transform import resize
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops_table, regionprops
from skimage.io import imsave
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import skimage.io

import os
from os import listdir,makedirs
from os.path import isfile, join
# import imagej

import cv2
import numpy as np
import subprocess

from sklearn.cluster import KMeans
from skimage import measure, color, io
from skimage.segmentation import clear_border


import pandas
from magicgui import magic_factory
from magicgui import magicgui
from magicgui.widgets import Table
from magicgui.tqdm import trange

from napari.utils.notifications import show_info
from napari.types import ImageData, LabelsData, NewType
import napari
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari import Viewer
from napari import layers
from napari.utils import progress
from napari.utils.colormaps import colormap_utils as cu

import pathlib
import tempfile
from zipfile import ZipFile

import re
import PIL
import h5py

import napari_aphid.path as paths
from collections import Counter
from pandas import DataFrame
import shutil
import matplotlib.cm as cm
from fileinput import filename
from glob import glob
from scipy import ndimage
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog, QHBoxLayout, QPushButton, QWidget, QListWidget
from qtpy.QtCore import Qt

import pandas as pd
from turtle import done
from skimage.morphology import closing, square, remove_small_objects
from magicgui.widgets import ComboBox, Container

import subprocess
import time
import threading
import queue


zip_dir = tempfile.TemporaryDirectory()

class MyProcess(threading.Thread):
    def __init__(self,nom_image,q):
        threading.Thread.__init__(self)
        self.nom_image = nom_image
        self.q = q
    
    def run(self):
        data = self.q.get()
        ilastik_path = 'C:/Program Files/ilastik-1.3.3post3/ilastik.exe'
        filename, file_extension = os.path.splitext(self.nom_image)
        donner = '--raw_data='+self.nom_image
        recevoir = '--output_filename_format='+filename+'_result_type'+file_extension
        projet_path = '--project=C:/Users/Metuarea Herearii/Desktop/yolo_detection_tools/segmentation_model.ilp'
        start_process = time.time()
        subprocess.run([ilastik_path,'--headless',projet_path,'--export_source=Simple Segmentation',donner,recevoir])
        end_process = time.time()
        file_name = os.path.basename(filename)
        print(f"IMG {file_name} = {np.round(end_process-start_process,2)} second")
        os.remove(filename+'_result_type'+file_extension)



def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def table_to_widget(table: dict) -> QWidget:
    """
    Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
    contains a QTableWidget with that data.
    """
    view = Table(value=table)

    copy_button = QPushButton("Copy to clipboard")

    @copy_button.clicked.connect
    def copy_trigger():
        view.to_dataframe().to_clipboard()

    save_button = QPushButton("Save as csv...")

    @save_button.clicked.connect
    def save_trigger():
        filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
        view.to_dataframe().to_csv(filename)

    widget = QWidget()
    widget.setWindowTitle("region properties")
    widget.setLayout(QGridLayout())
    widget.layout().addWidget(copy_button)
    widget.layout().addWidget(save_button)
    widget.layout().addWidget(view.native)

    return widget

def get_quantitative_data(image, napari_viewer):
    img=image
    seuil=25

    connidie=np.where(img==1)
    hyphe=np.where(img==2)
    img[connidie]=0

    labels_mask = measure.label(img, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=False) 

    print(">",len(hyphe[0]))
    print(">",len(connidie[0]))
    
    minus=0
    for rg in regions:
        if len(rg.coords[:,0])>seuil:
            print(">",len(regions)-minus)
            break
        else: 
            minus+=1    
    d = {'nombre dhyphes': [len(regions)-minus], 'hyphe': [len(hyphe[0])], 'connidie': [len(connidie[0])]}

    dock_widget = table_to_widget(d)
    napari_viewer.window.add_dock_widget(dock_widget, area='right')
    
def quantitative_data_for_all(dictionnaire,napari_viewer):
    A = [] #sous dossier
    B = [] #nom image
    C = []
    D = []
    E = []
    for ix in dictionnaire.keys():
        img=dictionnaire[ix]
        seuil=25

        connidie=np.where(img==1)
        hyphe=np.where(img==2)
        img[connidie]=0

        labels_mask = measure.label(img, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=False) 

        print(">",len(hyphe[0]))
        print(">",len(connidie[0]))
        
        minus=0
        for rg in regions:
            if len(rg.coords[:,0])>seuil:
                print(">",len(regions)-minus)
                name_xx = ix.split('xx')
                A.append(name_xx[0])
                B.append(name_xx[1])
                C.append(len(regions)-minus)
                D.append(len(hyphe[0]))
                E.append(len(connidie[0]))
                break
            else: 
                minus+=1    

    d = {'sous dossier':A,'nom image':B,'nombre dhyphes': C, 'hyphe': D, 'connidie': E}
    dock_widget = table_to_widget(d)
    napari_viewer.window.add_dock_widget(dock_widget, area='right')
    
    
    
def get_quantitative_data_all_for_csv(dossier_des_images,napari_viewer):
    A = [] 
    B = []
    C = []
    D = []
    E = []
    
    dictionnaire = {}
    
    for ix in os.listdir(dossier_des_images):
        chemin_dans_sousdossier = os.path.join(dossier_des_images,ix)
        if len(os.listdir(chemin_dans_sousdossier))!=0:
            for iy in os.listdir(chemin_dans_sousdossier):
                if iy.find("result")!=-1:
                    data_dico=imread(os.path.join(chemin_dans_sousdossier,iy))
                    print("chemin sous dossier",os.path.join(chemin_dans_sousdossier,iy))
                    dictionnaire[iy]=data_dico

    for ix in dictionnaire.keys():
        img=dictionnaire[ix]
        seuil=25

        connidie=np.where(img==1)
        hyphe=np.where(img==2)
        img[connidie]=0

        labels_mask = measure.label(img, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=False) 
        
        minus=0
        for rg in regions:
            if len(rg.coords[:,0])>seuil:
                name_xx = ix.split('xx')
                A.append(name_xx[0])
                B.append(name_xx[1][:-4])
                C.append(len(regions)-minus)
                D.append(len(hyphe[0]))
                E.append(len(connidie[0]))
                break
            else: 
                minus+=1    

    d = {'sous dossier':A,'nom image':B,'nombre dhyphes': C, 'hyphe': D, 'connidie': E}

    dock_widget = table_to_widget(d)
    napari_viewer.window.add_dock_widget(dock_widget, area='right',name="Save")
    
def define_marker(labels):
    n,m = labels.shape
    for ij in range(n):
        lit_temp=labels[ij]
        i=0
        while i<(len(lit_temp)-1):
            x = lit_temp[i]
            y = lit_temp[i+1]
            if x!=y:
                if x!=0:
                    if y!=0:
                        lit_temp[i]=0
                        lit_temp[i+1]=0
            i+=1
    return labels
        
output_dir = tempfile.TemporaryDirectory()
    
@magic_factory(call_button="Run segmentation",filename={"label": "Images:"},filename2={"label": "Ilastik Pixel classification:"})
def process_function_segmentation(napari_viewer : Viewer,filename=pathlib.Path.cwd(),filename2=pathlib.Path.cwd()): 
    
    zip_dir = tempfile.TemporaryDirectory()

    with ZipFile(filename,'r') as zipObject:
        listOfFileNames = zipObject.namelist()        
        for i in range(len(listOfFileNames)):            
            zipObject.extract(listOfFileNames[i],path=zip_dir.name)
            
    image_abs_path = []

    T1 = os.listdir(zip_dir.name) # = dossier1, dossier2, ...

    for ix in T1:
        dossier,_ = os.path.splitext(zip_dir.name+'\\'+ix)
        T2 = os.listdir(dossier)
        for iix in T2:
            sub_dossier = zip_dir.name+'\\'+ix+'\\'+iix
            for iiix in os.listdir(sub_dossier):
                image_abs_path.append(zip_dir.name+'\\'+ix+'\\'+iix+'\\'+iiix)

    abs_path_image_h5 = [ix.replace("\\","/") for ix in image_abs_path if ix.split('\\')[-1].endswith('h5')]
    abs_path_image_tif = [ix.replace("\\","/") for ix in image_abs_path if ix.split('\\')[-1].endswith('tif')]
    

    fpath = ""
    for ix in os.listdir(os.environ["ProgramFiles"]):
        if ix.startswith('ilastik'):
            fpath = os.path.join(os.environ["ProgramFiles"],ix)
            break

    try:
        ilastik_path = ""
        for ix in os.listdir(fpath):
            if ix.endswith('.exe') and ix.startswith("ilastik"):
                ilastik_path = os.path.join(fpath,ix)
                break
    except FileNotFoundError:
        print("Ilastik folder not found in :",os.environ["ProgramFiles"])
    except:
        print("Search task failed")

    if os.path.isfile(ilastik_path):
        print(f"{ix} found")
    else:
        print("ilastik.exe not found in :",fpath)


    print('output',output_dir.name)

    start_time = time.time()
    queueLock = threading.Lock()
    workQueue = queue.Queue(10)

    SEG = []
    threads_list = []
    for path_ix in trange(len(abs_path_image_h5)):
        im_h5 = abs_path_image_h5[path_ix]
        thread = MyProcess(im_h5,workQueue)
        thread.start()
        threads_list.append(thread)

        filename, file_extension = os.path.splitext(im_h5)
        image_name = os.path.basename(filename)
        SEG.append(os.path.join(output_dir.name,image_name+'_result_type.tif'))

    queueLock.acquire()
    for word in abs_path_image_h5:
        name_image = os.path.basename(word)
        workQueue.put(name_image)
    queueLock.release()

    while not workQueue.empty():
        pass

    for t in threads_list:
        t.join()

    print(f"Total process time : {np.round(time.time() - start_time,2)} seconds")


    # ICI, ON CREE DEUX CLASSES ET ON REMPLI LES TROUS
    for ix in range(len(SEG)):
        print("création deux classes :",ix,len(SEG))
        path_image = SEG[ix] 
        img = skimage.io.imread(path_image)


        data = np.array(img)
        fond_image=np.where(data==0)
        aphid=np.where(data!=0)
        data[fond_image]=0
        data[aphid]=255
        
        # data_fill = ndi.binary_fill_holes(data).astype(int)
        
        # data = np.array(data)
        # fond_image=np.where(data==0)
        # aphid=np.where(data!=0)
        # data[fond_image]=0
        # data[aphid]=255

        os.remove(path_image)
        #imsave(path_image,data)
        path_image_new = path_image[:-4]+'.png'
        plt.imsave(path_image_new, data, cmap = plt.cm.gray)
    print('done')
        
    ########## Enlever les petites régions avec moyenne

    SEG = [os.path.join(output_dir.name,ix) for ix in os.listdir(output_dir.name)]

    def PolyArea2D(pts):
        lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
        return area

    dico = {}
    for iw in range(len(SEG)):
        img = cv2.imread(SEG[iw])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        thresh_img = 255- thresh_img
        contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
        # create an empty black image
        drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)
        dico[SEG[iw]] = [contours,hull]
    
    SEG = [os.path.join(output_dir.name,ix) for ix in os.listdir(output_dir.name)]


    for ix in range(len(SEG)):
        path_image = SEG[ix]
        img = imread(path_image)
        img = np.squeeze(img[:,:,0])

        bigger = np.zeros_like(img)
        labels = label(img)

        A_temps = [R.area for R in regionprops(labels)]

        Y = np.sort(A_temps) 
        Y1 = np.array(Y)

        classe1 = Y1[Y1 <= np.mean(Y)]
        classe2 = Y1[Y1 > np.mean(Y)]

        for R in regionprops(labels):
            if R.area in classe2:
                for c in R.coords:  
                    bigger[c[0], c[1]] = 1
        
        os.remove(path_image)
        plt.imsave(path_image, bigger, cmap = plt.cm.gray)
        
    ################################ Ranger les images dans un seul dossier
    
    print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",len(SEG))
    #names =  []
    for ix in range(len(abs_path_image_tif)):
        ab1 = abs_path_image_tif[ix].split('/')[-2]
        ab2 = abs_path_image_tif[ix].split('/')[-1][:-4].replace('.','_')
        name_folder = ab1+'_'+ab2
        #names.append(name_folder)
        path_folder = os.path.join(output_dir.name,name_folder)

        os.mkdir(path_folder)

        old_image_tif_path = abs_path_image_tif[ix]
        old_image_h5_path = abs_path_image_h5[ix]        
        old_image_mask_tif_math = SEG[ix]

        new_image_tif_path = os.path.join(path_folder,ab1+'.'+abs_path_image_tif[ix].split('/')[-1])
        new_image_h5_path = os.path.join(path_folder,ab1+'.'+abs_path_image_h5[ix].split('/')[-1])
        new_image_mask_tif_math = os.path.join(path_folder,ab1+'.'+SEG[ix].split('\\')[-1])
            
        shutil.move(old_image_tif_path,new_image_tif_path)
        shutil.move(old_image_h5_path,new_image_h5_path)
        shutil.move(old_image_mask_tif_math,new_image_mask_tif_math)
        print(path_folder)
        print(">",new_image_tif_path)
        print(">",new_image_h5_path)
        print(">",new_image_mask_tif_math)

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    ######
    
    MASK = []
    for ix in os.listdir(output_dir.name):
        path_temp = os.path.join(output_dir.name,ix)
        for subfolder in os.listdir(path_temp):
            path_file = os.path.join(path_temp,subfolder)
            if path_file.endswith('result_type.png'):
            # if path_file.endswith('result_type.tif'):
                MASK.append(path_file)
                
    dico_dico = {}
    for iw in range(len(MASK)):
        one_image = imread(MASK[iw])
        
        data = np.array(one_image)
        #fond_image=np.where(data==85)
        #bord=np.where(data==170)
        #aphid=np.where(data==255)
        #data[fond_image]=0
        #data[bord]=255
        #data[aphid]=255

        #data_fill = ndi.binary_fill_holes(data).astype(int)

        #data = np.array(data_fill)
        #fond_image=np.where(data==0)
        #aphid=np.where(data==1)
        #data[fond_image]=0
        #data[aphid]=255
        
        name_image = MASK[iw].split('\\')[-1].replace('_type','')[:-3]+'png'
        
        image_with_two_classes = os.path.join('\\'.join(MASK[iw].split('\\')[:-1]),name_image) #new name for image with two classes
        dico_dico[image_with_two_classes]=data
        plt.imsave(image_with_two_classes, data, cmap = plt.cm.gray)
    
    L = list(dico_dico.keys())    
    
    dico = {}
    for iw in range(len(L)):
        img = cv2.imread(L[iw])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        thresh_img = 255- thresh_img
        contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))
        # create an empty black image
        drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)
        dico[L[iw]] = [contours,hull]
    
    dico_out = {}
    for iw in dico.keys():
        L = dico[iw]
        contours = L[0]
        hull = L[1]

        area_hull_list = []
        area_contour_list = []
        diff = []

        for i in range(len(contours)):
            new_list_hull = []
            for ix in range(len(hull[i])):
                new_list_hull.append(list(hull[i][ix][0]))

            new_list_contours = []
            for ix in range(len(contours[i])):
                new_list_contours.append(list(contours[i][ix][0]))

            area_of_hull = PolyArea2D(new_list_hull)
            area_of_contour = PolyArea2D(new_list_contours)
            area_hull_list.append(area_of_hull)
            area_contour_list.append(area_of_contour)
            diff.append(area_of_hull-area_of_contour)

        mean_area_aphid = np.mean(area_hull_list)

        OUT_value = []
        for i in range(len(diff)):
            if area_hull_list[i] > mean_area_aphid:
                if diff[i]/area_hull_list[i] > 0.2:
                    OUT_value.append([i,diff[i]])
 
        M = [np.mean(diff) for _ in range(len(diff))]
        limit_moyenne_value = np.mean(diff)*3

        OUT_value_y = [OUT_value[i][1] for i in range(len(OUT_value))]
        OUT_value_x = [OUT_value[i][0] for i in range(len(OUT_value))]

        M = [np.mean(diff) for _ in range(len(diff))]
        
        if OUT_value!=[]:
            dico_out[iw] = OUT_value
            
    L1 = list(dico_out.keys())
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",L1)
    names =  []

    for ix in range(len(L1)):
        ###########################

        path_image_for_wat = L1[ix]
        one_image = np.squeeze(imread(path_image_for_wat))[:,:,0]
    


        img1 = cv2.imread(path_image_for_wat)
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)    
        opening = clear_border(opening)
        sure_bg = cv2.dilate(opening,kernel,iterations=2)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        ret3, markers = cv2.connectedComponents(sure_fg)
        markers = markers+10
        markers[unknown==255] = 0
        markers = cv2.watershed(img1,markers)
        img1[markers == -1] = [0,255,255]  
        # img2 = color.label2rgb(markers, bg_label=0)
        # OUTPUT = img1

        #ENLEVER LES REGIONS    
        # INPUT = img1

        stud_image = np.squeeze(img1[:,:,0])
        shape_image = stud_image.shape
        kernel = np.ones((3, 3), np.uint8)
        stud_image_eros = cv2.erode(stud_image, kernel) 

        # data_fill = ndi.binary_fill_holes(stud_image_eros).astype(int)        
        # data_bigger_m = np.array(data_fill)
        
        data_bigger_m = np.array(stud_image_eros)
        fond_image=np.where(data_bigger_m==0)
        aphid=np.where(data_bigger_m!=0)
        data_bigger_m[fond_image]=0
        data_bigger_m[aphid]=255 
        
        # OUTPUT = data_bigger_m

        #FIXER LE SEUIL
        # INPUT = data_bigger_m

        img = data_bigger_m

        labels = label(img)

        A_temps = [R.area for R in regionprops(labels)]

        Y = np.sort(A_temps) 
        Y1 = np.array(Y)
        X = np.arange(len(Y))

        ### Application de la moyenne

        A_moy = []
        moy_m = np.mean(Y1)
        for i in range(len(Y1)):
            if Y1[i] > moy_m:
                A_moy.append(1)
            else:
                A_moy.append(0)

        A_moy_np = np.array(A_moy)

        classe2_m = Y1[A_moy_np == 1]

        # OUTPUT = classe2_m

        ### Enlever les petits morceaux
        # INPUT = img et classe2_m

        bigger_m = np.zeros_like(img)         
        for R in regionprops(labels):
            if R.area in classe2_m:
                for c in R.coords:  
                    bigger_m[c[0], c[1]] = 1  
        
        data_bigger_m_m = np.array(bigger_m)
        fond_image=np.where(data_bigger_m_m==0)
        aphid=np.where(data_bigger_m_m!=0)
        data_bigger_m_m[fond_image]=0
        data_bigger_m_m[aphid]=255                     
                    
        # OUTPUT = data_bigger_m_m
        
        # ML_Ml.append([nom_image,data_bigger_m_m])
        # names.append(L1[ix].split('\\')[-2])
        nom_image = L1[ix].split('\\')[-2]
        print("image :",nom_image)

        names.append(nom_image)
        skimage.io.imsave(L1[ix], data_bigger_m_m)

    def open_name(item):
        
        name = item.text()
        name_folder = name[:-4]

        
        print('Loading', name, '...')

        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
        fname = f'{output_dir.name}\{name}'
        for fname_i in os.listdir(fname):
            if fname_i.find('_result_type')==-1:
                if fname_i.find('_result')!=-1:
                    data_label = imread(f'{fname}\{fname_i}')
                    data_label1 = np.array(data_label)       
                    
                    fond_image=np.where(data_label1==0)
                    aphid=np.where(data_label1!=0)
                    data_label1[fond_image]=0
                    data_label1[aphid]=255     
                    
                    napari_viewer.add_labels(data_label1,name=f'{fname_i[:-4]}')
                elif fname_i.endswith('.tif'):
                    napari_viewer.add_image(imread(f'{fname}\{fname_i}'),name=f'{fname_i[:-4]}')

        print('... done.')


    list_widget = QListWidget()
    for n in names:
        list_widget.addItem(n)    
    list_widget.currentItemChanged.connect(open_name)
    napari_viewer.window.add_dock_widget([list_widget], area='right',name="Images")
    list_widget.setCurrentRow(0)
    
@magic_factory(call_button="save modification", layout="vertical")
def save_modification(image_seg : napari.layers.Labels, image_raw : ImageData, napari_viewer : Viewer):
    data_label = image_seg.data   
    sousdossier = image_seg.name.split('_result')[0].replace('.','_')
    nom_image = image_seg.name
    os.remove(f'{output_dir.name}\{sousdossier}\{image_seg}.png')
    imsave(f'{output_dir.name}\{sousdossier}\{image_seg}.png', img_as_uint(data_label))

@magic_factory(call_button="Run classification",filename={"label": "Ilastik Object classification:"})
def process_function_classification(napari_viewer : Viewer,filename=pathlib.Path.cwd()): 
    path_folder = output_dir.name
    L = os.listdir(path_folder)
    image_path = [os.path.join(path_folder,ix) for ix in L]
    print(">>>>>>>>>>>>", image_path)
    
    ####################################
    raw_data_set = []
    segmentation_image_set = []
    raw_data_tif_set = []
    for ix in range(len(image_path)):
        image_set = image_path[ix]
        for ix_file in os.listdir(image_set):
            if ix_file.endswith('.h5'):
                raw_data_set.append(os.path.join(image_set,ix_file))
            if ix_file.endswith('_result.png'):
                segmentation_image_set.append(os.path.join(image_set,ix_file))
            if ix_file.endswith('.tif') and ix_file.find('_result_type')==-1:
                raw_data_tif_set.append(os.path.join(image_set,ix_file))
    ####################################
    for ix in range(len(segmentation_image_set)):
        path_image = segmentation_image_set[ix] 
        img = skimage.io.imread(path_image)

        gxg = img.shape
        if len(gxg)==3:
            data_fill = ndi.binary_fill_holes(np.squeeze(img[:,:,0])).astype(int)
        else:
            data_fill = ndi.binary_fill_holes(img).astype(int)
        data = np.array(data_fill)
        fond_image=np.where(data==0)
        aphid=np.where(data!=0)
        data[fond_image]=0
        data[aphid]=255

        os.remove(path_image)
        #imsave(path_image,data)
        path_image_new = path_image[:-4]+'.png'
        plt.imsave(path_image_new, data, cmap = plt.cm.gray)
    ####################################
    # path_image = segmentation_image_set[5]
    # img = skimage.io.imread(path_image)
    # img1 = np.squeeze(img[:,:,0])
    # labels = label(img1)
    # A_temps = [R.area for R in regionprops(labels)]
    # print(A_temps)

    # Y = np.sort(A_temps)
    # Y1 = np.array(Y)
    # classe2 = Y1[Y1 > 10]

    # for R in regionprops(labels):
    #     if R.area in classe2:
    #         for c in R.coords:  
    #             print(R.area,c[0],c[1])
    # plt.imshow(img)
    ####################################
    #for i,j in trange(zip(raw_data_set,segmentation_image_set),total=len(raw_data_set)):
    
    # projet_path = '--project="C:/Users/User/sergio_plugin/classification_simpseg_datad.ilp"'
    projet_path = '--project='+filename

    fpath = ""
    for ix in os.listdir(os.environ["ProgramFiles"]):
        if ix.startswith('ilastik'):
            fpath = os.path.join(os.environ["ProgramFiles"],ix)
            break

    try:
        ilastik_path = ""
        for ix in os.listdir(fpath):
            if ix.endswith('.exe') and ix.startswith("ilastik"):
                ilastik_path = os.path.join(fpath,ix)
                break
    except FileNotFoundError:
        print("Ilastik folder not found in :",os.environ["ProgramFiles"])
    except:
        print("Search task failed")

    if os.path.isfile(ilastik_path):
        print(f"{ix} found")
    else:
        print("ilastik.exe not found in :",fpath)
    
    for i in trange(len(raw_data_set)):
        
        path_label = raw_data_set[i].split('\\')
        path_ = '\\'.join(path_label[:-1]).replace('\\','/')
        nom = path_label[-2]

        table_filename_path = '--table_filename="'+path_+'/exported_'+nom+'.csv"'
        raw_image = '--raw_data="'+raw_data_set[i].replace('\\','/')+'"'
        seg_image = '--segmentation_image="'+segmentation_image_set[i].replace('\\','/')+'"'

        subprocess.run([ilastik_path,
                                '--headless',
                                projet_path,
                                '--export_source=Object Predictions',
                                raw_image,
                                seg_image,
                                table_filename_path])

        # subprocess.run(["C:/Program Files/ilastik-1.3.3post3/ilastik.exe",
        #                         '--headless',
        #                         projet_path,
        #                         '--export_source=Object Predictions',
        #                         raw_image,
        #                         seg_image,
        #                         table_filename_path])
    ####################################
    #delete images output of classification
    print(raw_data_set)
    for i in range(len(image_path)):
        L_temp = os.listdir(image_path[i])
        L_h5 = [ix for ix in L_temp if ix.endswith('.h5')]
        for j in L_h5:
            un_lien = os.path.join(image_path[i],j)
            if un_lien not in raw_data_set:
                print(un_lien)
                os.remove(un_lien)
                
    # Ranger les tableaux dans une liste

    TABLE_PATH = []
    for i in trange(len(raw_data_set)):   
        path_label = raw_data_set[i].split('\\')
        path_ = '\\'.join(path_label[:-1]).replace('\\','/')
        nom = path_label[-2]
        table_name_path = path_+'/exported_'+nom+'_table.csv'
        print(table_name_path)
        TABLE_PATH.append(table_name_path)
    ####################################
    def make_bbox(bbox_extents):
        """Get the coordinates of the corners of a
        bounding box from the extents
        Parameters
        ----------
        bbox_extents : list (4xN)
            List of the extents of the bounding boxes for each of the N regions.
            Should be ordered: [min_row, min_column, max_row, max_column]
        Returns
        -------
        bbox_rect : np.ndarray
            The corners of the bounding box. Can be input directly into a
            napari Shapes layer.
        """
        minr = bbox_extents[0]
        minc = bbox_extents[1]
        maxr = bbox_extents[2]
        maxc = bbox_extents[3]

        bbox_rect = np.array(
            [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
        )
        bbox_rect = np.moveaxis(bbox_rect, 2, 0)

        return bbox_rect



    def table_to_widget(table: dict) -> QWidget:
        """
        Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
        contains a QTableWidget with that data.
        """
        view = Table(value=table)

        copy_button = QPushButton("Copy to clipboard")

        @copy_button.clicked.connect
        def copy_trigger():
            view.to_dataframe().to_clipboard()

        save_button = QPushButton("Save as csv...")

        @save_button.clicked.connect
        def save_trigger():
            filename, _ = QFileDialog.getSaveFileName(save_button, "Save as csv...", ".", "*.csv")
            view.to_dataframe().to_csv(filename)

        edit_button = QPushButton("Edit")
        @edit_button.clicked.connect
        def edit_trigger():
            pass
            
        widget = QWidget()
        widget.setWindowTitle("region properties")
        widget.setLayout(QGridLayout())
        widget.layout().addWidget(copy_button)
        widget.layout().addWidget(save_button)
        widget.layout().addWidget(view.native)
        widget.layout().addWidget(edit_button)

        return widget

    # set up the annotation values and text display properties
    box_annotations = ['Winged adult', 'Apterous adult', 'Nymph','Larvae','Larvae/Nymph small','Molt']
    text_property = 'box_label'
    text_color = 'green'

    # create the GUI for selecting the values
    def create_label_menu(shapes_layer, label_property, labels):
        """Create a label menu widget that can be added to the napari viewer dock

        Parameters:
        -----------
        shapes_layer : napari.layers.Shapes
            a napari shapes layer
        label_property : str
            the name of the shapes property to use the displayed text
        labels : List[str]
            list of the possible text labels values.

        Returns:
        --------
        label_widget : magicgui.widgets.Container
            the container widget with the label combobox
        """
        # Create the label selection menu
        label_menu = ComboBox(label='text label', choices=labels)
        label_widget = Container(widgets=[label_menu])

        def update_label_menu(event):
            """This is a callback function that updates the label menu when
            the current properties of the Shapes layer change
            """
            new_label = str(shapes_layer.current_properties[label_property][0])
            if new_label != label_menu.value:
                label_menu.value = new_label

        shapes_layer.events.current_properties.connect(update_label_menu)

        def label_changed(event):
            """This is acallback that update the current properties on the Shapes layer
            when the label menu selection changes
            """
            selected_label = event.value
            current_properties = shapes_layer.current_properties
            current_properties[label_property] = np.asarray([selected_label])
            shapes_layer.current_properties = current_properties

        label_menu.changed.connect(label_changed)

        return label_widget

    def visualiser_resultat_detection(iy):

        doneeee = pd.read_csv(TABLE_PATH[iy])

        class_predicted = list(doneeee['Predicted Class'])
        size_in_pixel = list(doneeee['Size in pixels'])
        bbx_min0 = list(doneeee['Bounding Box Minimum_0'])
        bbx_min1 = list(doneeee['Bounding Box Minimum_1'])
        bbx_max0 = list(doneeee['Bounding Box Maximum_0'])
        bbx_max1 = list(doneeee['Bounding Box Maximum_1'])


        original_image = imread(raw_data_tif_set[iy].replace('\\','/'));print(raw_data_tif_set[iy].replace('\\','/'))

        image_label_temp = imread(segmentation_image_set[iy].replace('\\','/'))
        if len(image_label_temp.shape)==2:
            label_image = imread(segmentation_image_set[iy].replace('\\','/'));print(segmentation_image_set[iy].replace('\\','/'))
        else:
            label_image = np.squeeze(imread(segmentation_image_set[iy].replace('\\','/'))[:,:,0]);print(segmentation_image_set[iy].replace('\\','/'))
            
        # create the features dictionary
        features = {
            'label': class_predicted,
            'size' : size_in_pixel,
            'bbx_0' : bbx_min1,
            'bbx_1' : bbx_min0,
            'bbx_2' : bbx_max1,
            'bbx_3' : bbx_max0,    
        }

        donnee_feature = pd.DataFrame(features)
        rslt_donnee_feature = donnee_feature.loc[donnee_feature['size'] >= 10] #supprimer les petits espaces
        
        features = {
            'label': list(rslt_donnee_feature['label']),
            'size' : list(rslt_donnee_feature['size']),
            'bbx_0' : list(rslt_donnee_feature['bbx_0']),
            'bbx_1' : list(rslt_donnee_feature['bbx_1']),
            'bbx_2' : list(rslt_donnee_feature['bbx_2']),
            'bbx_3' : list(rslt_donnee_feature['bbx_3']),    
        }
        
        
        
        """
        Annotate segmentation with text
        ===============================
        Perform a segmentation and annotate the results with
        bounding boxes and text
        .. tags:: analysis
        """
        
        # create the bounding box rectangles
        bbox_rects = make_bbox([features[f'bbx_{i}'] for i in range(4)])

        # specify the display parameters for the text
        text_parameters = {
            'string': '{label}\nsize (in pxl): {size}',
            'size': 11,
            'color': 'green',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        viewer = napari.view_image(original_image, name='aphid')
        label_layer = viewer.add_labels(label_image, name='segmentation')
        
        
        label_image_ = label(label_image)
        stats_label = regionprops(label_image_)
        points = [s.centroid for s in stats_label]
        label_layer = viewer.add_points(points, face_color='green', symbol='cross', size=10, features=features,text=text_parameters)
        
        # shapes_layer = viewer.add_shapes(
        #     bbox_rects,
        #     face_color='transparent',
        #     edge_color='green',
        #     features=features,
        #     text=text_parameters,
        #     name='bounding box',
        # )

        dock_widget = table_to_widget(rslt_donnee_feature[['label','size']])
        
        viewer.window.add_dock_widget(dock_widget, area='right')
               
        # if __name__ == '__main__':
        #     napari.run()
            
    # visualiser_resultat_detection(6)

    names = [ix.split('\\')[-1] for ix in image_path]
    print(names)
    
    def open_name(item):
        
        name = item.text()
        name_folder = name[:-4]

        
        print('Loading', name, '...')

        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
        fname = f'{output_dir.name}\{name}'
        print('donnee dans fname :',os.listdir(fname))
        
        AAAA = np.array(os.listdir(fname))
        
        original_image_path_tif = list(filter(None,list(np.where(np.char.endswith(AAAA,'.tif'),AAAA,''))))[0]
        original_image_path_h5 = list(filter(None,list(np.where(np.char.endswith(AAAA,'.h5'),AAAA,''))))[0]
        result_image_path_png = list(filter(None,list(np.where(np.char.endswith(AAAA,'_result.png'),AAAA,''))))[0]
        result_type_image_path_png = list(filter(None,list(np.where(np.char.endswith(AAAA,'_result_type.png'),AAAA,''))))[0]
        table_path_csv = list(filter(None,list(np.where(np.char.endswith(AAAA,'.csv'),AAAA,''))))[0]
        
        # result
        data_label = np.squeeze(imread(f'{fname}\{result_image_path_png}')[:,:,0])
        data_label1 = np.array(data_label)       
                    
        fond_image=np.where(data_label1==0)
        aphid=np.where(data_label1!=0)
        data_label1[fond_image]=0
        data_label1[aphid]=255     
                    
        napari_viewer.add_labels(data_label1,name=f'{result_image_path_png[:-4]}')
        
        # tif        
        napari_viewer.add_image(imread(f'{fname}\{original_image_path_tif}'),name=f'{original_image_path_tif[:-4]}')
        
        # donnee
        doneeee = pd.read_csv(f'{fname}\{table_path_csv}')
        
        class_predicted = list(doneeee['Predicted Class'])
        size_in_pixel = list(doneeee['Size in pixels'])
        bbx_min0 = list(doneeee['Bounding Box Minimum_0'])
        bbx_min1 = list(doneeee['Bounding Box Minimum_1'])
        bbx_max0 = list(doneeee['Bounding Box Maximum_0'])
        bbx_max1 = list(doneeee['Bounding Box Maximum_1'])
        
        original_image = imread(f'{fname}\{original_image_path_tif}')
        image_label_temp = imread(f'{fname}\{result_image_path_png}')
        if len(image_label_temp.shape)==2:
            label_image = imread(f'{fname}\{result_image_path_png}')
        else:
            label_image = np.squeeze(imread(f'{fname}\{result_image_path_png}')[:,:,0])
            
        features = {
            'label': class_predicted,
            'size' : size_in_pixel,
            'bbx_0' : bbx_min1,
            'bbx_1' : bbx_min0,
            'bbx_2' : bbx_max1,
            'bbx_3' : bbx_max0,    
        }

        donnee_feature = pd.DataFrame(features)
        
        rslt_donnee_feature = donnee_feature.loc[donnee_feature['size'] >= 10] #supprimer les petits espaces
        features = {
            'label': list(rslt_donnee_feature['label']),
            'size' : list(rslt_donnee_feature['size']),
            'bbx_0' : list(rslt_donnee_feature['bbx_0']),
            'bbx_1' : list(rslt_donnee_feature['bbx_1']),
            'bbx_2' : list(rslt_donnee_feature['bbx_2']),
            'bbx_3' : list(rslt_donnee_feature['bbx_3']),    
        }
        
        bbox_rects = make_bbox([features[f'bbx_{i}'] for i in range(4)])

        # specify the display parameters for the text
        text_parameters = {
            'string': '{label}\nsize (in pxl): {size}',
            'size': 11,
            'color': 'green',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        label_layer = napari_viewer.add_labels(label_image, name='segmentation')
        
        

        bbox_du_rectangle = [features[f'bbx_{i}'] for i in range(4)]
        minr = bbox_du_rectangle[0]
        minc = bbox_du_rectangle[1]
        maxr = bbox_du_rectangle[2]
        maxc = bbox_du_rectangle[3]
        n_len = len(bbox_du_rectangle[0])
        rectangle_center = [(int(minr[iixh] + maxr[iixh])/2, int(minc[iixh] + maxc[iixh])/2) for iixh in range(n_len)]
        label_layer = napari_viewer.add_points(rectangle_center, face_color='green', symbol='cross', size=10, features=features,text=text_parameters)        
        
        dock_widget = table_to_widget(rslt_donnee_feature[['label','size']])        
        napari_viewer.window.add_dock_widget(dock_widget, area='right')                   

        print('... done.')


    list_widget = QListWidget()
    for n in names:
        list_widget.addItem(n)    
    list_widget.currentItemChanged.connect(open_name)
    napari_viewer.window.add_dock_widget([list_widget], area='right',name="Images")
    list_widget.setCurrentRow(0)