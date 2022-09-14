from tkinter import Image
from tkinter.ttk import Progressbar
from tifffile import imsave
import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage import img_as_uint, measure
from skimage.transform import resize
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops_table
from skimage.io import imsave
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import skimage.io

import os
from os import listdir,makedirs
from os.path import isfile, join


import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy
import subprocess

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
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QListWidget
from qtpy.QtCore import Qt

zip_dir = tempfile.TemporaryDirectory()


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
    
@magic_factory(call_button="Run segmentation",filename={"label": "Pick a file:"})
def process_function_segmentation(napari_viewer : Viewer,filename=pathlib.Path.cwd()): 
    
    zip_dir = tempfile.TemporaryDirectory()

    with ZipFile(filename,'r') as zipObject:
        listOfFileNames = zipObject.namelist()        
        for i in trange(len(listOfFileNames)):            
            zipObject.extract(listOfFileNames[i],path=zip_dir.name)
            
    T1 = os.listdir(zip_dir.name)
    dossier = zip_dir.name+'\\'+T1[0]
    T2 = os.listdir(dossier)
    sub_dossier = zip_dir.name+'\\'+T1[0]+'\\'+T2[0]
    image_abs_path = [zip_dir.name+'\\'+T1[0]+'\\'+T2[0]+'\\'+ix for ix in os.listdir(sub_dossier)]
    abs_path_image_h5 = [ix.replace("\\","/") for ix in image_abs_path if ix.split('\\')[-1].endswith('h5')]
    abs_path_image_tif = [ix.replace("\\","/") for ix in image_abs_path if ix.split('\\')[-1].endswith('tif')]


    print('output',output_dir.name)

    SEG = []
    for path in abs_path_image_h5:
        
        donner = '--raw_data="'+path+'"'
        recevoir = '--output_filename_format="'+os.path.join(output_dir.name,path.split('/')[-1][:-3])+'_result_type.jpg"'
        projet_path = '--project="C:/Users/User/sergio_plugin/segmentation_model.ilp"'
        
        subprocess.run(["C:/Program Files/ilastik-1.3.3post3/ilastik.exe",
                        '--headless',
                        projet_path,
                        '--export_source=Simple Segmentation',
                        donner,
                        recevoir])

        print(os.path.join(output_dir.name,path.split('/')[-1][:-3])+'_result_type.tif')
        SEG.append(os.path.join(output_dir.name,path.split('/')[-1][:-3])+'_result_type.tif')
    
    #names =  []
    for ix in range(len(abs_path_image_tif)):
        ab1 = abs_path_image_tif[ix].split('/')[-2]
        ab2 = abs_path_image_tif[ix].split('/')[-1][:-4].replace('.','_')
        name_folder = ab1+'_'+ab2
        #names.append(name_folder)
        path_folder = os.path.join(output_dir.name,name_folder)

        os.mkdir(path_folder)

        old_image_tif_path = abs_path_image_tif[ix]
        old_image_mask_tif_math = SEG[ix]

        new_image_tif_path = os.path.join(path_folder,ab1+'.'+abs_path_image_tif[ix].split('/')[-1])
        new_image_mask_tif_math = os.path.join(path_folder,ab1+'.'+SEG[ix].split('\\')[-1])
            
        shutil.move(old_image_tif_path,new_image_tif_path)
        shutil.move(old_image_mask_tif_math,new_image_mask_tif_math)
        print(path_folder)
        print(">",new_image_tif_path)
        print(">",new_image_mask_tif_math)

    ######
    
    MASK = []
    for ix in os.listdir(output_dir.name):
        path_temp = os.path.join(output_dir.name,ix)
        for subfolder in os.listdir(path_temp):
            path_file = os.path.join(path_temp,subfolder)
            if path_file.endswith('result_type.tif'):
                MASK.append(path_file)
                
    dico_dico = {}
    for iw in range(len(MASK)):
        one_image = imread(MASK[iw])
        
        data = np.array(one_image)
        fond_image=np.where(data==85)
        bord=np.where(data==170)
        aphid=np.where(data==255)
        data[fond_image]=0
        data[bord]=255
        data[aphid]=255

        data_fill = ndi.binary_fill_holes(data).astype(int)

        data = np.array(data_fill)
        fond_image=np.where(data==0)
        aphid=np.where(data==1)
        data[fond_image]=0
        data[aphid]=255
        
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
    names =  []
    
    for ix in range(len(L1)):
        one_image = np.squeeze(imread(L1[ix])[:,:,0])
        data = np.array(one_image)

        distance = ndi.distance_transform_edt(one_image)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=one_image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=one_image)
        labels_data = define_marker(labels)
        
        data = np.array(labels_data)
        fond_image=np.where(data==0)
        aphid=np.where(data!=0)
        data[fond_image]=0
        data[aphid]=255
        names.append(L1[ix].split('\\')[-2])
        skimage.io.imsave(L1[ix], data)
        #plt.imsave(L1[ix], data, cmap = plt.cm.gray)

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
                else:
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
    print("image_seg.name :",image_seg.name)
    
    sousdossier = image_seg.name.split('_result')[0].replace('.','_')
    print("sousdossier (_result) :",image_seg.name.split('_result')[0].replace('.','_'))
    
    nom_image = image_seg.name
    print("nom_image (xx) :", image_seg.name)
    
    os.remove(f'{output_dir.name}\{sousdossier}\{image_seg}.png')
    imsave(f'{output_dir.name}\{sousdossier}\{image_seg}.png', img_as_uint(data_label))