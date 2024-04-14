# CFZ-Net-training

Deep Learning for Capillary Free Zones (CFZ) Segmentation using OCTA Images. The implementation uses Python and PyTorch.


Overview
------------
This GitHub repository hosts the implementation of CFZ-Net, a deep learning model developed in PyTorch for segmenting capillary free zones (CFZ), arteries, and veins in optical coherence tomography angiography (OCTA) images. The model is trained on high-resolution 6x6mm OCTA images, aiming to accurately delineate vascular structures for medical research and diagnosis.

Key Features:

    Utilizes PyTorch framework for efficient training and inference
    Designed specifically for segmenting CFZ, arteries, and veins in OCTA images
    Preprocessing scripts for data augmentation and preparation
    Evaluation metrics for assessing segmentation performance


Images were acquired using the AngioVue SD-OCT device (Optovue, Fremont, CA, USA). The OCT system had a 70,000 Hz A-scan rate with ~5 μm axial and ~15 μm lateral resolutions. All OCTA images used for this study were 6 mm × 6 mm scans; only superficial OCTA images were used.

These figures show two representative OCTA images and corresponding manually generated ground truths and predicted images:
![The CFZ-Net](https://github.com/mansour2002/CFZ-Net-training/blob/main/Figures/CFZ%20Segmentation%201.png?raw=true)


![The CFZ-Net](https://github.com/mansour2002/CFZ-Net-training/blob/main/Figures/CFZ%20Segmentation%202.png?raw=true)


Dependencies
------------
- PyTorch >= 2.2.1+cu118
- CUDA >= 11.8
- Python >= 3.9


