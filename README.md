# Deep Learning For Metrology (Dante Zegveld)
Deep Learning For Metrology Thesis Project.
The objective of this project is to utilizing deep learning techniques for the quality control of sieves. 
To achieve this objective, two U-Net Modules are employed for the task of semantic image segmentation. The task of the U-Net modules is to predict the segments of individual holes, and the microscope's lens. 
By combining the predictions from both modules during a post-processing procedure, all relevant individual segments are extracted and a report is generated containing the dimensions of each individual hole. 
The report can be utilized to determine the quality of the sieve as a metrological tool. Additionally, a visual representation of the resulting segments is generated to facilitate visual inspection.

In this repository, all files used for training and evaluating the U-Net modules can be found. 
Moreover, the IPYNB's used for training on Google Collab are available. 
In the same Notebooks folder, you can also find the: DL_Metrology_Inference_Dante_Zegveld.ipynb notebook which let's you play around with the inference pipeline, 
generating an output csv and visual representation utilizing already trained models.

The trained models and training dataset are not available on github but can be found here: 
[Trained Models](https://drive.google.com/drive/folders/1TC0VTyAUSO5cr5OUHkAPEOcbzYE4BsJV?usp=sharing), [Training Data](https://drive.google.com/drive/folders/1KEtpBw4GdnH3jrnlx5rCna8K3uNWRQVe?usp=sharing)

The inference notebook can also be found here: [Inference Notebook](https://colab.research.google.com/drive/1duikg-xjS74IYG0-5cCrBl2TxZUjJ3C7?usp=sharing)
Follow the instructions in this notebook which refer you to get the following files to run it: [Required Files](https://drive.google.com/drive/folders/1kNN6Yu0MOQ6yzHv4rivvOhWIil8lxdtT?usp=sharing)