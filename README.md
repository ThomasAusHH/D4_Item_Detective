# Readme

## Item Checker

This python program provides a graphical user interface (GUI) for selecting item classes and affixes, and scans your screen for items of the selected class that have the chosen affixes using OpenCV and Pytesseract. It's a helpful tool for games where you need to keep track of specific types of items and their characteristics (affixes).

## Requirements

To run the program, you need the following Python libraries:

- `tkinter` for the graphical user interface
- `cv2` for image processing
- `numpy` for numerical operations
- `pytesseract` for Optical Character Recognition (OCR)
- `pandas` for data processing
- `re` for regular expressions
- `pyautogui` for taking screenshots
- `threading` for running multiple processes at the same time
- `matplotlib.pyplot` for plotting (currently unused)

Furthermore, you need to install [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) on your system. The path to the `tesseract.exe` should be defined as `pytesseract.pytesseract.tesseract_cmd`.

## Setup

1. Install all required Python libraries:

```sh
pip install tkinter cv2 numpy pytesseract pandas re pyautogui threading matplotlib
```

2. Download and install Tesseract-OCR. Make sure to add it to your system's PATH or define the path to the `tesseract.exe` in the script (replace `r"C:\Program Files\Tesseract-OCR\tesseract.exe"` with the correct path on your system).

3. Clone or download this repository.

4. Make sure the `D4_affixes_eng.csv` file is in the same directory as the script. This CSV file should contain the data of the item affixes with the columns 'Class', 'Type', 'Affixes'. The 'Class' and 'Type' columns should contain lists of strings, where each string is separated by a "|".

## Usage

To start the program, run the python script using the command:

```sh
python <script-name.py>
```

The GUI provides a dropdown menu for selecting the item class and buttons for each item type, where you can select the affixes you want to search for. After setting up your selection, you can start the screenshot scan by clicking the 'Start/Stop' button. The program will start taking screenshots of your screen and searching for items that match your selection using OCR. The results are printed to the console.

To clear your current selection, click the 'Clear Selection' button. To stop the screenshot scan, click the 'Start/Stop' button again.

## Notes

This program uses simple image processing and OCR techniques and may not work perfectly for all scenarios. It might need some adjustments according to your specific needs and system configuration. If the program does not detect your items or affixes correctly, you may need to adjust the image processing parameters (e.g., blur_value, adaptive_block_size, adaptive_C, contour_area_threshold, contour_approximation_accuracy) in the script.
