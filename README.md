# napari-aphid

[![License BSD-3](https://img.shields.io/pypi/l/napari-aphid.svg?color=green)](https://github.com/hereariim/napari-aphid/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-aphid.svg?color=green)](https://pypi.org/project/napari-aphid)
[![Downloads](https://static.pepy.tech/badge/napari-aphid)](https://pepy.tech/project/napari-aphid)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-aphid.svg?color=green)](https://python.org)
[![tests](https://github.com/hereariim/napari-aphid/workflows/tests/badge.svg)](https://github.com/hereariim/napari-aphid/actions)
[![codecov](https://codecov.io/gh/hereariim/napari-aphid/branch/main/graph/badge.svg)](https://codecov.io/gh/hereariim/napari-aphid)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-aphid)](https://napari-hub.org/plugins/napari-aphid)

A plugin to classify aphids by stage of development.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-aphid` via [pip]:

    pip install napari-aphid

To install latest development version :

    pip install git+https://github.com/hereariim/napari-aphid.git

Several commands are used in the script. These commands are exclusively executable on windows, for the moment. So, this plugin works only on windows.

## Description

This plugin is a tool to count the number of aphids from two models developed on ilastik. Implemented in napari, this tool allows the correction of pixels and labels that are not well 
predicted. 

In this plugin we find our two main parts of the aphid counting model presented in two widgets. A third widget allows to save the updates applied on the segmentation mask.

This plugin is an use cas, dedicated to private use of french laboratory.

## Plugin input

### Segmentation

The user must give two objects as input:

- Compressed file in .zip format
- Ilastik pixel classification model in .ilp format

In particular, compressed file must be organized as follows:

```
.
└── Country.zip
    └── Country
        ├── Area1
        │   ├── Area1.im_1.tif
        │   ├── Area1.im_1.h5
        │   ├── Area1.im_2.tif 
        │   ├── Area1.im_2.h5  
        │   ├── Area1.im_3.tif
        │   ├── Area1.im_3.h5
        │   ...
        │   ├── Area1.im_n.tif
        │   └── Area1.im_n.h5
        │
        ├── Area2
        │   ├── Area2.im_1.tif
        │   ├── Area2.im_1.h5
        │   ├── Area2.im_2.tif
        │   ├── Area2.im_2.h5
        │   ├── Area2.im_3.tif
        │   ├── Area2.im_3.h5
        │   ...
        │   ├── Area2.im_n.tif
        │   └── Area2.im_n.h5
        │
        ...
        │
        └── Arean
            ├── Arean.im_1.tif
            ├── Arean.im_1.h5
            ├── Arean.im_2.tif
            ├── Arean.im_2.h5
            ├── Arean.im_3.tif
            ├── Arean.im_3.h5
            ...
            ├── Arean.im_n.tif
            └── Arean.im_n.h5
```

In each folder Area1, Area2, ..., Arean, we notice that **each tif image is accompanied by its h5 version**. The images in h5 format were generated by the Export h5 widget of the Ilastik plugin in the ImageJ software.

### Classification

The user must give the Ilastik object classification model in .ilp format.

## Widget: Image segmentation

This widget is a tool to segment a set of images. It takes as input a compressed file of images and an ilastik segmentation model. A Run button is used to start the image segmentation process. In the background, the console presents the progress status. This widget returns a menu which is a list of processed images. This list allows an RGB image and its segmentation mask to be displayed in the napari window.

![segmentation_cpe](https://user-images.githubusercontent.com/93375163/212323051-bc84d597-a9ff-46ca-b897-cb18a0e77b4c.png)

**User conduct :** In this widget, the user corrects the image with the annotation tools (brush and eraser only). With the brush, he/she has to add the same colour presented in the image. To obtain this colour, the user can take the color with the color picker tool. With the eraser, he/she erase colour not well predicted. Tous les annotations appliquées dans l'image doit être sauvegarder avec le bouton *Save* du widget **Save modification**

## Widget: Save modification

This is the backup of the segmentation mask. It saves updates applied to the mask.

## Widget: Object classification

This widget is a tool to classify segmented images. It takes as input an ilastik object classification model. A Run button is used to start the classification process. In the background, the console shows the progress of the image processing. This widget returns a menu that lists the processed images. This list provides two elements. The first is the display of the selected image in the window. The second is the display of a table that shows the predicted classes for each object.

![classification_cpe](https://user-images.githubusercontent.com/93375163/212323369-32423622-4f41-4dcb-800b-39ff66be67f9.png)

**User conduct :** In this widget, the user corrects labels not well predicted in the table at the bottom right. He must not forget to save his correction with the Save button.
When the user has finished with all his images, he uses the Export button to import a quantitative table. This table contains for each image, the name of the aphid type and its size in pixels.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-aphid" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/hereariim/napari-aphid/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
