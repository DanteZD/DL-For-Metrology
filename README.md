# Deep Learning For Metrology (Dante Zegveld)
Deep Learning For Metrology Bachelor Thesis Project by Dante Zegveld.
The objective of this project is to utilizing deep learning techniques for the quality control of sieves. 
To achieve this objective, two U-Net Modules are employed for the task of semantic image segmentation. The task of the U-Net modules is to predict the segments of individual holes, and the microscope's lens. 
By combining the predictions from both modules during a post-processing procedure, all relevant individual segments are extracted and a report is generated containing the dimensions of each individual hole. 
The report can be utilized to determine the quality of the sieve as a metrological tool. Additionally, a visual representation of the resulting segments is generated to facilitate visual inspection.

In this repository, all files used for training and evaluating the U-Net modules can be found. 

The models are modified versions of the U-Net architecture as described by:  <br>
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for
biomedical image segmentation. In Medical image computing and computer-assisted
intervention–miccai 2015: 18th international conference, munich, germany, october
5-9, 2015, proceedings, part iii 18 (pp. 234–241).
