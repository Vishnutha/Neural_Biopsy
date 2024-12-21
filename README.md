# Myelinn Neural Biopsy

This project aims to detect myelin sheaths in images, specifically focusing on the identification and measurement of the axon diameter, inner radius, outer radius, and thickness of the myelin sheaths. It uses image processing techniques like adaptive thresholding, contour detection, and morphological operations to segment and measure the axon and myelin sheath structures. This process is vital in neurobiology research for understanding the health and structure of nerve cells.

The project utilizes various tools such as OpenCV for image processing, Matplotlib for visualization, and Tkinter for creating an interactive graphical interface to view the results.

# How to Run the Code
Prerequisites
Ensure you have Python 3.x installed. You also need to install the following dependencies:

- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow
- Tkinter (usually comes pre-installed with Python)

## Clone the repository
```
https://github.com/Vishnutha/Neural_Biopsy.git
cd Neural_Biopsy
```
## Run Patch Generation.py
```
python3 patch_generation.py
```

## Run Neural Biopsy Processing to detect contours and generate results
```
python3 neural_biopsy.py
```

## Open the Jupyter Notebook for further analysis and results visualization
```
jupyter notebook neural_biopsy.ipynb
```
