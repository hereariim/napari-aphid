from tkinter import Image
from tkinter.ttk import Progressbar

import skimage
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage import img_as_uint, measure
from skimage.transform import resize
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops_table
from skimage.io import imsave
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

from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QListWidget
from qtpy.QtCore import Qt

zip_dir = tempfile.TemporaryDirectory()

def function_central(filepath):
    
    path_image = str(filepath).replace('\\','/')
  
    donner = '--raw_data="'+path_image+'"'
    output_dir = tempfile.TemporaryDirectory()
    recevoir = '--output_filename_format="'+os.path.join(output_dir.name,path_image.split('/')[-1][:-4])+'_result_type.jpg"'
    projet_path = '--project="'+os.path.join(paths.get_models_dir(),'segmentation_model.ilp')+'"'
    
    subprocess.run(["C:/Program Files/ilastik-1.3.3post3/ilastik.exe",
                    '--headless',
                    projet_path,
                    '--export_source=Simple Segmentation',
                    donner,
                    recevoir])
    
    f = os.path.join(output_dir.name,path_image.split('/')[-1][:-4])+'_result_type.png'
    # imag = np.squeeze(skimage.io.imread(path_image_n))
    
    ##################
    # Traitement Ante
    ##################
    print("Traitement Ante")
    sep=re.compile(r"\\")
    end=re.compile(r'.(jpg|png)$')

    # CUT = []
    # for f in output_image:
    
    lien=sep.split(f)[0]+r"/Otsu_centrage"
    name=sep.split(f)[len(sep.split(f))-1] # recuperation du nom du fichier de base
    name=end.sub(r'',name)
    img=PIL.Image.open(f)
    
    data = np.array(img)
    tache=np.where(data==255)
    condide=np.where(data==85)
    hyphe=np.where(data==170)
    data[tache]=0
    data[hyphe]=1
    data[condide]=1

    labels_mask = measure.label(data, background=0) # Solution venant de stackoverflow, Mesure les differents elements                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True) 
    if len(regions) > 1: #On mets toutes les regions qui ne sont la plus grande en back ground
        for rg in regions[1:]:          
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1 #Toutes les coordonnées du gros objet sont unifiées à 1
    data = labels_mask
    for j in range(len(hyphe[0])): # on remet les hyphes du plus gros element en label 2
        if data[hyphe[0][j],hyphe[1][j]] == 1 : 
            data[hyphe[0][j],hyphe[1][j]] = 2
    
    return data

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

    output_dir = tempfile.TemporaryDirectory()
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
        
    for ix in range(len(abs_path_image_tif)):
        ab1 = abs_path_image_tif[ix].split('/')[-2]
        ab2 = abs_path_image_tif[ix].split('/')[-1][:-4].replace('.','_')
        name_folder = ab1+'_'+ab2
        path_folder = os.path.join(output_dir.name,name_folder)

        os.mkdir(path_folder)

        old_image_tif_path = abs_path_image_tif[ix]
        old_image_mask_tif_math = SEG[ix]

        new_image_tif_path = os.path.join(path_folder,abs_path_image_tif[ix].split('/')[-1])
        new_image_mask_tif_math = os.path.join(path_folder,SEG[ix].split('\\')[-1])
            
        shutil.move(old_image_tif_path,new_image_tif_path)
        shutil.move(old_image_mask_tif_math,new_image_mask_tif_math)
        print(path_folder)
        print(">",new_image_tif_path)
        print(">",new_image_mask_tif_math)

    ######
    
    dico = {}
    with ZipFile(filename,'r') as zipObject:
    
        listOfFileNames = zipObject.namelist()
        
        for i in trange(len(listOfFileNames)):
            
            zipObject.extract(listOfFileNames[i],path=zip_dir.name)            
            temp_i = listOfFileNames[i].replace('/','xx').replace(" ","")       
            temp_i_jpg = listOfFileNames[i].replace('/','xx')[:-4].replace(" ","")
            os.mkdir(zip_dir.name+'\\'+temp_i_jpg)
            shutil.move(zip_dir.name+'\\'+listOfFileNames[i].replace('/','\\'),zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i)
            image_segm = function_central(zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i)
            imsave(zip_dir.name+'\\'+temp_i_jpg+'\\'+temp_i_jpg+'_result.png', img_as_uint(image_segm))
            dico[temp_i_jpg+'_result.png'] = image_segm
            
    print("Extraction done located into",zip_dir.name)
        
    names = []
    for ix in os.listdir(zip_dir.name):
        if len(os.listdir(os.path.join(zip_dir.name,ix)))!=0:
            names.append(ix)

    def open_name(item):
        
        name = item.text()
        name_folder = name[:-3]
        
        print('Loading', name, '...')

        napari_viewer.layers.select_all()
        napari_viewer.layers.remove_selected()    
        fname = f'{zip_dir.name}\{name}'
        for fname_i in os.listdir(fname):
            if fname_i.find('result')!=-1:
                data_label = imread(f'{fname}\{fname_i}')
                data_label1 = np.array(data_label)
                fond=np.where(data_label1==85)
                edge=np.where(data_label1==170)
                aphid=np.where(data_label1==255)
                data_label1[fond]=0
                data_label1[edge]=170
                data_label1[aphid]=255                
                napari_viewer.add_labels(data_label1,name=f'{fname_i[:-3]}')
            else:
                napari_viewer.add_image(imread(f'{fname}\{fname_i}'),name=f'{fname_i[:-3]}')

        print('... done.')


    list_widget = QListWidget()
    for n in names:
        list_widget.addItem(n)    
    list_widget.currentItemChanged.connect(open_name)
    napari_viewer.window.add_dock_widget([list_widget], area='right',name="Images")
    list_widget.setCurrentRow(0)