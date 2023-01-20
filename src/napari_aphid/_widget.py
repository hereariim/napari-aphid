# from tkinter import Image
# from tkinter.ttk import Progressbar
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
from skimage.morphology import erosion
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_opening
from skimage.filters import threshold_otsu as gaussian, sobel
import skimage.io
from tqdm import tqdm

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
import h5py

import napari_aphid.path as paths
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
from skimage.morphology import closing, square, remove_small_objects
from magicgui.widgets import ComboBox, Container

import subprocess
import time
import threading
import queue

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


zip_dir = tempfile.TemporaryDirectory()

class MyProcess(threading.Thread):
    def __init__(self,nom_image,projet,path_to_ilastik,q):
        threading.Thread.__init__(self)
        self.nom_image = nom_image
        self.q = q
        self.projet = projet
        self.ilastik_path = path_to_ilastik
    
    def run(self):
        data = self.q.get()
        filename, file_extension = os.path.splitext(self.nom_image)
        print(f"{filename} IMPORTED")
        donner = '--raw_data='+self.nom_image
        recevoir = '--output_filename_format='+filename+'_result'+file_extension
        projet_path = '--project='+self.projet #C:/Users/Metuarea Herearii/Desktop/yolo_detection_tools/segmentation_model.ilp'
        start_process = time.time()
        subprocess.run([self.ilastik_path,'--headless',projet_path,'--export_source=Simple Segmentation',donner,recevoir])
        end_process = time.time()
        file_name = os.path.basename(filename)
        print(f"IMG {filename} = {np.round(end_process-start_process,2)} second")
        # os.remove(filename+'_result_type'+file_extension)
        
class MyProcess_classification(threading.Thread):
    def __init__(self,element_files,projet,path_to_ilastik,q):
        threading.Thread.__init__(self)
        self.elements_of_file = element_files
        self.q = q
        self.projet = projet
        self.ilastik_path = path_to_ilastik
    
    def run(self):
        data = self.q.get()
        
        table_filename_path = '--table_filename='+self.elements_of_file[2]
        raw_image = '--raw_data='+self.elements_of_file[0]
        seg_image = '--segmentation_image='+self.elements_of_file[1]
        
        nom_image = os.path.basename(self.elements_of_file[0])
        filename, file_extension = os.path.splitext(nom_image)
        print(f"{filename} IMPORTED")
        projet_path = '--project='+self.projet #C:/Users/Metuarea Herearii/Desktop/yolo_detection_tools/segmentation_model.ilp'
        start_process = time.time()
        
        subprocess.run([self.ilastik_path,'--headless',projet_path,'--export_source=Object Predictions',
                        raw_image,
                        seg_image,
                        table_filename_path])  
        
        end_process = time.time()
        print(f"IMG {filename} = {np.round(end_process-start_process,2)} second")
        # os.remove(filename+'_result_type'+file_extension)

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
       
output_dir = tempfile.TemporaryDirectory()
zip_dir = tempfile.TemporaryDirectory()
    
@magic_factory(call_button="Run segmentation",filename={"label": "Images:"},filename2={"label": "Ilastik Pixel classification:"})
def process_function_segmentation(napari_viewer : Viewer,filename=pathlib.Path.cwd(),filename2=pathlib.Path.cwd()): 
    
    filename2 = str(filename2)

    with ZipFile(filename,'r') as zipObject:
        listOfFileNames = zipObject.namelist()        
        for i in range(len(listOfFileNames)):            
            zipObject.extract(listOfFileNames[i],path=zip_dir.name)
            
    image_abs_path = []

    T1 = os.listdir(zip_dir.name)

    for ix in T1:
        dossier = os.path.join(zip_dir.name,ix)
        T2 = os.listdir(dossier)
        for iix in T2:
            sub_dossier = os.path.join(zip_dir.name,ix,iix)
            for iiix in os.listdir(sub_dossier):
                image_abs_path.append(os.path.join(zip_dir.name,ix,iix,iiix))

    abs_path_image_h5 = [ix for ix in image_abs_path if ix.endswith('h5')]
    abs_path_image_tif = [ix for ix in image_abs_path if ix.endswith('tif')]
    
    #CHECK ILASTIK
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
        root_pc = str(pathlib.Path.home()).split("\\")[0]+"\\Program Files"
        check_version = [ix for ix in os.listdir(root_pc) if ix.find('ilastik')!=-1][0]
        path_to_ilastik = os.path.join(root_pc,check_version,"ilastik.exe")
        print(f"{ix} found")
    else:
        print("ilastik.exe not found in :",fpath)

    # SEGMENTATION PROCESSING
    sub_list_h5 = []
    A_list_h5 = []
    ctp = 0
    for ix in abs_path_image_h5: 
        if len(A_list_h5)==5:
            sub_list_h5.append(A_list_h5)
            A_list_h5 = []
            A_list_h5.append(ix)
            ctp+=1
        else:
            A_list_h5.append(ix)
    s=0
    for ix in sub_list_h5:
        s+=len(ix)
    if s!=len(abs_path_image_h5):
        sub_list_h5.append(A_list_h5)
    
    #ilasitk path
    

    
    start_time = time.time()
    queueLock = threading.Lock()
    workQueue = queue.Queue(10)

    print("H5 file",abs_path_image_h5)

    SEG = []
    threads_list = []
    for iy in tqdm(range(len(sub_list_h5)), desc= 'PROCESSING'):
        list_to_work = sub_list_h5[iy]
        for path_ix in range(len(list_to_work)):
            im_h5 = list_to_work[path_ix]
            thread = MyProcess(im_h5,filename2,path_to_ilastik,workQueue)
            thread.start()
            threads_list.append(thread)

            # filename, file_extension = os.path.splitext(im_h5)
            # image_name = os.path.basename(filename)
            # SEG.append(os.path.join(output_dir.name,image_name+'_result_type.tif'))

        queueLock.acquire()
        for word in list_to_work:
            name_image = os.path.basename(word)
            workQueue.put(name_image)
        queueLock.release()

        while not workQueue.empty():
            pass

        for t in threads_list:
            t.join()

    print(f"Total process time : {np.round(time.time() - start_time,2)} seconds")

    # for ik in trange(len(abs_path_image_h5)):   
    #     donner = '--raw_data="'+abs_path_image_h5[ik]+'"'
    #     recevoir = '--output_filename_format="'+os.path.join(zip_dir.name,abs_path_image_h5[ik].split('/')[-1][:-3])+'_result.jpg"'
    #     projet_path = f'--project={filename2}'
        
    #     subprocess.run([ilastik_path,
    #                     '--headless',
    #                     projet_path,
    #                     '--export_source=Simple Segmentation',
    #                     donner,
    #                     recevoir])
     
    # Mettre dans une liste le chemin des images segmentées        
    SEG = []
    for ix in T1:
        dossier = os.path.join(zip_dir.name,ix)
        T2 = os.listdir(dossier)
        for iix in T2:
            sub_dossier = os.path.join(zip_dir.name,ix,iix)
            for iiix in os.listdir(sub_dossier):
                name_to_study = os.path.join(zip_dir.name,ix,iix,iiix)
                if name_to_study.find("result")!=-1:
                    SEG.append(name_to_study)
    print(SEG)

    # ICI, ON CREE DEUX CLASSES ET ON REMPLI LES TROUS
    from skimage.util import img_as_ubyte
    from skimage import io

    print("IMAGE BINARISATION")
    for ix in tqdm(range(len(SEG))):
        path_image = SEG[ix] 
        img = skimage.io.imread(path_image)

        data = np.array(img)
        fond_image=np.where(data==0)
        aphid=np.where(data!=0)
        data[fond_image]=0
        data[aphid]=255
                
        data = np.array(data)
        fond_image=np.where(data==0)
        aphid=np.where(data!=0)
        data[fond_image]=0
        data[aphid]=255

        head , _ = os.path.splitext(path_image)
        path_image_new = head+'.png'
        plt.imsave(path_image_new, data, cmap = plt.cm.gray)

    print("REMOVE tiny space")
    ########## Enlever les petites régions avec moyenne puis enregistrer
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
        
    ########## Ranger les images dans un seul dictionnaire
    
    print("Ranger dans un dictionnaire",len(SEG))

    dico_image_temp = {}
    names = []
    for ix in range(len(abs_path_image_tif)):
        path_image_temp = abs_path_image_tif[ix]
        image_temp = os.path.basename(path_image_temp)
        head, _ = os.path.splitext(path_image_temp)
        image_temp_name,_ =os.path.splitext(image_temp)
        if os.path.isfile(head+'.h5'):
            dico_image_temp[image_temp_name]=[path_image_temp,head+'.h5',head+'_result.png']
            names.append(image_temp_name)
    print('...done')

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    ########## Ranger les masques de segmentation dans une liste MASK et un dictionnaire Temp_L_M_dico
    MASK = []
    Temp_L_M_dico = {}
    for k in dico_image_temp:
        MASK.append(dico_image_temp[k][2])
        Temp_L_M_dico[k]=dico_image_temp[k][2]
    
    ########## Ranger dans un dictionnaire les masques de segmentation en np.array et enregistrer en _result
    dico_dico = {}
    for ix in range(len(MASK)):
        data = imread(MASK[ix])
        dico_dico[MASK[ix]]=np.squeeze(data[:,:,0])

        dico_dico[MASK[ix]]=data
        plt.imsave(MASK[ix], data, cmap = plt.cm.gray)
    
    L = list(dico_dico.keys())    
    M_dico = list(dico.keys())
    Temp_L_M_dico = {}
    for ix,iy in zip(M_dico,L):
        Temp_L_M_dico[ix]=iy
    
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
    dico_size = {}

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

        dico_size[iw]=area_contour_list            
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
    print("PROCESSING IMAGE OVERLAPPED")
    for ix in range(len(L1)):
        ###########################

        path_image_for_wat = L1[ix]
        one_image = np.squeeze(imread(path_image_for_wat))[:,:,0]
    


        img1 = cv2.imread(path_image_for_wat)
        one_image = np.squeeze(img1)[:,:,0]

        binary = np.asarray(one_image)
        # typical way of using scikit-image watershed
        distance = ndi.distance_transform_edt(binary)
        sigma = 15
        blurred_distance = gaussian_filter(distance,sigma=sigma)
        fp = np.ones((3,) * binary.ndim)
        coords = peak_local_max(blurred_distance, footprint=fp, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        labels = watershed(-blurred_distance, markers, mask=binary)
        # identify label-cutting edges
        edges = sobel(labels)
        edges2 = sobel(binary)
        almost = np.logical_not(np.logical_xor(edges != 0, edges2 != 0)) * binary
        img2 = binary_opening(almost)
        image = erosion(img2)

        diameter = 15
        radius = diameter // 2
        x = np.arange(-radius, radius+1)
        x, y = np.meshgrid(x, x)
        r = x**2 + y**2
        se = r < radius**2
        im3 = ndimage.binary_opening(image, se)
        
        img = im3

        labels = label(img)

        A_temps = [R.area for R in regionprops(labels)]

        Y = np.sort(A_temps) 
        Y1 = np.array(Y)
        X = np.arange(len(Y))

        ### Application de la moyenne
        A_moy = []
        moy_m = np.mean(Y1)
        for i in range(len(Y1)):
            if Y1[i] > 1000:
                A_moy.append(1)
            else:
                A_moy.append(0)

        A_moy_np = np.array(A_moy)

        classe2_m = Y1[A_moy_np == 1]
        # OUTPUT = classe2_m

        ### Enlever les petits morceaux
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

        skimage.io.imsave(L1[ix], data_bigger_m_m)
    
    for ix in MASK:
        if ix not in L1:    
            path_image_for_wat = ix
            one_image = np.squeeze(imread(path_image_for_wat))[:,:,0]
                


            img1 = cv2.imread(path_image_for_wat)
            one_image = np.squeeze(img1)[:,:,0]
            binary = np.asarray(one_image)

            diameter = 15
            radius = diameter // 2
            x = np.arange(-radius, radius+1)
            x, y = np.meshgrid(x, x)
            r = x**2 + y**2
            se = r < radius**2

            im3 = ndimage.binary_opening(binary,se)
            img = im3

            data_bigger_m_m = np.array(img)
            fond_image=np.where(data_bigger_m_m==0)
            aphid=np.where(data_bigger_m_m!=0)
            data_bigger_m_m[fond_image]=0
            data_bigger_m_m[aphid]=255                     

            skimage.io.imsave(path_image_for_wat, data_bigger_m_m)
            
    dico_image_temp_vis = {}
    names = []
    for ix in range(len(abs_path_image_tif)):
        path_image_temp = abs_path_image_tif[ix]
        image_temp = os.path.basename(path_image_temp)
        head, _ = os.path.splitext(path_image_temp)
        image_temp_name,_ =os.path.splitext(image_temp)
        if os.path.isfile(head+'.h5'):
            dico_image_temp_vis[image_temp_name]=[path_image_temp,head+'.h5',head+'_result.png']
            names.append(image_temp_name)

    def open_name(item):
            
        name = item.text()
        
        print('OPEN:',name)
        
        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
            
        RGB_name = os.path.basename(dico_image_temp[name][0])
        data_RGB = imread(dico_image_temp[name][0])
        label_name = os.path.basename(dico_image_temp[name][2])
        data_label = np.array(imread(dico_image_temp[name][2]))
        
        shape_matrix = data_label.shape
        napari_viewer.add_image(data_RGB,name=f'{RGB_name}')
        napari_viewer.add_labels(data_label,name=f'{label_name}')
        
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
    name_label = image_seg.name    
    directory_tmp = os.listdir(zip_dir.name)[0]
    lettre = name_label[0]    
    path_to_data_label = os.path.join(zip_dir.name,directory_tmp,lettre,name_label)
    os.remove(path_to_data_label)
    imsave(path_to_data_label, img_as_uint(data_label))
    
def find_imagej_exe(os_environment='USERPROFILE'):
    user_environment = os.environ[os_environment]
    for ix in os.listdir(user_environment):
        path_temp_ix = os.path.join(user_environment,ix)
        if os.path.isdir(path_temp_ix):
            try:
                for iy in os.listdir(path_temp_ix):
                    file_path = os.path.join(path_temp_ix,iy)
                    if os.path.isfile(file_path):
                        _,extension = os.path.splitext(iy)
                        string_with_lowercase = (iy.lower())
                        if string_with_lowercase.find('fiji')!=-1 or string_with_lowercase.find('imagej')!=-1:
                            if extension=='.exe':
                                return file_path
            except:
                pass
    return ''


@magic_factory(call_button="Run classification",filename={"label": "Ilastik Object classification:"})
def process_function_classification(napari_viewer : Viewer,filename=pathlib.Path.cwd()):
    temp_file = zip_dir.name
    folder_in_temp = os.listdir(temp_file)[0]
    path_to_folder = os.path.join(temp_file,folder_in_temp,'')
    
    run_batch_path = os.path.join(paths.get_models_dir(),'RunBatch.ijm')
    
    # Reminder : Install Fiji in your user space C:\Users\[your name]\ImageJ2.app
    path_to_image_exe = find_imagej_exe()
    if len(path_to_image_exe)!=0:
        subprocess.run([path_to_image_exe,'--headless','--console','-macro',run_batch_path,path_to_folder])
    else:
        print('ERROR : EXECUTABLE IMAGEJ NOT FOUND IN USER ENVIRONMENT')
    RAW_H5 = []
    SEG_H5 = []
    for ix in os.listdir(path_to_folder):
        path_sub = os.path.join(path_to_folder,ix)
        for iy in os.listdir(path_sub):
            if iy.endswith('.h5'):
                path_file = os.path.join(path_sub,iy)
                if path_file.find('_result')==-1:
                    RAW_H5.append(path_file)
                else:
                    SEG_H5.append(path_file)

    dico_obj_class = {}
    for rgb_h5,seg_h5 in zip(RAW_H5,SEG_H5):
        fichier_image = os.path.basename(rgb_h5)
        head,_ = os.path.splitext(fichier_image)
        prefx,_ = os.path.splitext(rgb_h5)
        dico_obj_class[head] = [rgb_h5,seg_h5,prefx+'.csv']
        
    sub_list_h5 = []
    A_list_h5 = []
    ctp = 0
    for ix in list(dico_obj_class.keys()): 
        if len(A_list_h5)==5:
            sub_list_h5.append(A_list_h5)
            A_list_h5 = []
            A_list_h5.append(ix)
            ctp+=1
        else:
            A_list_h5.append(ix)
    s=0
    for ix in sub_list_h5:
        s+=len(ix)
    if s!=len(list(dico_obj_class.keys())):
        sub_list_h5.append(A_list_h5)
        
    import time
    start_time = time.time()
    queueLock = threading.Lock()
    workQueue = queue.Queue(10)
    
    class_path_ilastik = str(filename)
    
    root_pc = str(pathlib.Path.home()).split("\\")[0]+"\\Program Files"
    check_version = [ix for ix in os.listdir(root_pc) if ix.find('ilastik')!=-1][0]
    path_to_ilastik = os.path.join(root_pc,check_version,"ilastik.exe")
    
    SEG = []
    threads_list = []
    for iy in tqdm(range(len(sub_list_h5)), desc= 'PROCESSING'):
        list_sub_h5_to_work = sub_list_h5[iy]
        for path_ix in list_sub_h5_to_work:
            elements_of_file = dico_obj_class[path_ix]
            thread = MyProcess_classification(elements_of_file,class_path_ilastik,path_to_ilastik,workQueue)
            thread.start()
            threads_list.append(thread)

            # filename, file_extension = os.path.splitext(im_h5)
            # image_name = os.path.basename(filename)
            # SEG.append(os.path.join(output_dir.name,image_name+'_result_type.tif'))

        queueLock.acquire()
        for word in list_sub_h5_to_work:
            name_image = os.path.basename(word)
            workQueue.put(name_image)
        queueLock.release()

        while not workQueue.empty():
            pass

        for t in threads_list:
            t.join()

    print(f"Total process time : {np.round(time.time() - start_time,2)} seconds")
        
    GET_PNG = []
    GET_CSV = []
    GET_TIF = []
    GET_OC = []
    for ix in os.listdir(path_to_folder):
        path_sub = os.path.join(path_to_folder,ix)
        for iy in os.listdir(path_sub):
            path_of_iy = os.path.join(path_sub,iy)
            if iy.endswith('.png') and iy.find('Object Predictions')==-1:
                GET_PNG.append(path_of_iy)
            if iy.endswith('.csv'):
                GET_CSV.append(path_of_iy)
            if iy.endswith('.tif'):
                GET_TIF.append(path_of_iy)
            if iy.find('Object Predictions')!=-1:
                GET_OC.append(path_of_iy)
    print(GET_OC)
    print("1 done")
    dico_for_vis = {}
    names = []
    for ix,iy,iz,i0 in zip(GET_PNG,GET_CSV,GET_TIF,GET_OC):
        name_file = os.path.basename(ix)
        head,_ = os.path.splitext(name_file)
        image_name = head.split('_result')[0]
        names.append(image_name)
        dico_for_vis[image_name]=[ix,iy,iz,i0]    

    import pandas as pd
    from skimage.io import imread
    from matplotlib.backends.backend_qt5agg import FigureCanvas
    from matplotlib.figure import Figure

    def table_to_widget(table: dict,path: str) -> QWidget:
        """
        Takes a table given as dictionary with strings as keys and numeric arrays as values and returns a QWidget which
        contains a QTableWidget with that data.
        """
        view = Table(value=table)
        print(path)
        export_button = QPushButton("Export")

        @export_button.clicked.connect
        def copy_trigger():
            DATA_FRAME = []
            for ix in os.listdir(temp_file):
                path_1 = os.path.join(temp_file,ix)
                for iy in os.listdir(path_1):
                    path_2 = os.path.join(path_1,iy)
                    for iz in os.listdir(path_2):
                        if iz.endswith('csv'):
                            head, _ = os.path.splitext(iz)
                            name_image = head.split("_table")[0]
                            path_3 = os.path.join(path_2,iz)

                            df = pd.read_csv(path_3)
                            if 'object_id'  in list(df.columns):
                                n = len(df['object_id'])
                                df['Country'] = [ix for iy in np.arange(n)]
                                df['ID'] = ['ID:'+str(im) for im in np.arange(n)]
                                df['Image'] = [name_image for im in np.arange(n)]
                                DATA_FRAME.append(df)
                            else:
                                print('NO PREDICTION:',name_image)
            pd_class = []
            cty = []
            id_obj = []
            id_img = []
            size_img = []
            for idx in range(len(DATA_FRAME)):
                pd_class = pd_class + DATA_FRAME[idx]['Predicted Class'].values.tolist()
                cty = cty + DATA_FRAME[idx]['Country'].values.tolist()
                id_obj = id_obj + DATA_FRAME[idx]['ID'].values.tolist()
                id_img = id_img + DATA_FRAME[idx]['Image'].values.tolist()
                size_img = size_img + DATA_FRAME[idx]['Size in pixels'].values.tolist()
            dico_final = {"Country":cty,"Image":id_img,"ID object":id_obj,"Class":pd_class}
            df_final = pd.DataFrame(dico_final)
            filename, _ = QFileDialog.getSaveFileName(save_button, "Export", ".", "*.csv")
            df_final.to_csv(filename, index=False)

        save_button = QPushButton("Save modification")
        @save_button.clicked.connect
        def save_trigger():
            un_tableau_csv_to_save = pd.read_csv(path)
            un_tableau_csv_to_save['Predicted Class'] = view.to_dataframe()['class'].values.tolist()
            un_tableau_csv_to_save.to_csv(path)


        widget = QWidget()
        widget.setWindowTitle("Prediction")
        layout_qgrid=QGridLayout()
        widget.setLayout(layout_qgrid)
        widget.layout().addWidget(export_button)
        widget.layout().addWidget(save_button)
        widget.layout().addWidget(view.native)
        return widget

    list_widget = QListWidget()

    for n in names:
        list_widget.addItem(n)    
        
    DOCK_widget_list=[]

    def open_name(item):
            
        name = item.text()
        
        print('OPEN:',name)
        
        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
            
        RGB_name = os.path.basename(dico_for_vis[name][2])
        data_RGB = imread(dico_for_vis[name][2])
        
        label_name = os.path.basename(dico_for_vis[name][3])
        data_label = np.array(imread(dico_for_vis[name][3]))
        
        napari_viewer.add_image(data_RGB,name=f'{RGB_name}')  
        label_layers = napari_viewer.add_labels(data_label,name=f'{label_name}')
        label_layers
        
        un_tableau_csv = dico_for_vis[name][1]
        df = pd.read_csv(un_tableau_csv)
        if 'Predicted Class'  in list(df.columns):
            class_predicted = df['Predicted Class']
            n= len(df['Predicted Class'])
            ID_list = ['ID:'+str(ix) for ix in np.arange(n)]
            d = {'ID':ID_list,'class':class_predicted}
            
            features = {
                        'name' : ID_list,
                        'label': list(df['Predicted Class']),
                        'size' : list(df['Size in pixels']),
                        'bbx_0' : list(df['Bounding Box Minimum_0']),
                        'bbx_1' : list(df['Bounding Box Minimum_1']),
                        'bbx_2' : list(df['Bounding Box Maximum_0']),
                        'bbx_3' : list(df['Bounding Box Maximum_1']),    
                    }
                    
            text_parameters = {
                        'string': '{name}',
                        'size': 11,
                        'color': 'red',
                        'anchor': 'upper_left',
                        'translation': [-3, 0],
                    }

            bbox_du_rectangle = [features['bbx_0'],features['bbx_1'],features['bbx_2'],features['bbx_3']]
            minr = bbox_du_rectangle[0]
            minc = bbox_du_rectangle[1]
            maxr = bbox_du_rectangle[2]
            maxc = bbox_du_rectangle[3]
            n_len = len(bbox_du_rectangle[0])
            rectangle_center = [(minc[iixh]+(maxc[iixh]-minc[iixh])/2,minr[iixh]+(maxr[iixh]-minr[iixh])/2) for iixh in range(n_len)]
            label_layer = napari_viewer.add_points(rectangle_center, face_color='red', symbol='cross', size=20, features=features,text=text_parameters)      
            
            table_dock_widget = table_to_widget(d,un_tableau_csv)        
            table_dock_widget.setFixedHeight(300)
            dock_widget = napari_viewer.window.add_dock_widget(table_dock_widget, area='right',name="Save")
            DOCK_widget_list.append(dock_widget)
            if len(DOCK_widget_list)!=1:
                napari_viewer.window.remove_dock_widget(DOCK_widget_list[-2])
            dock_widget
        else:
            show_info('NO PREDICTION:'+name)
        print('... done.')


    list_widget.currentItemChanged.connect(open_name)
    napari_viewer.window.add_dock_widget([list_widget], area='right',name="Images")
    list_widget.setCurrentRow(0)