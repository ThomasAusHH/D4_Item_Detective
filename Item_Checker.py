import tkinter as tk
from tkinter import ttk, filedialog, messagebox, IntVar
from tkinter.scrolledtext import ScrolledText
import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import pyautogui
import threading
import matplotlib.pyplot as plt

class AffixSelectionWindow(tk.Toplevel):
    def __init__(self, parent, item_type, affixes_df, selected_class, selected_affixes_callback):
        super().__init__(parent)
        self.title(f'{item_type} Affix Selection')
        self.geometry('400x400')
        self.item_type = item_type
        self.affixes_df = affixes_df
        self.selected_class = selected_class
        self.selected_affixes_callback = selected_affixes_callback
        self.create_widgets()
        

    def create_widgets(self):
        self.listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, height=10, width=30)
        self.listbox.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        item_affixes = self.affixes_df[
            (self.affixes_df['Class'].apply(lambda x: self.selected_class in x)) &
            (self.affixes_df['Type'].apply(lambda x: self.item_type in x))
        ]['Affixes'].tolist()

        for affix in item_affixes:
            self.listbox.insert(tk.END, affix)

        self.btn_confirm = ttk.Button(self, text='Confirm Selection', command=self.confirm_selection)
        self.btn_confirm.pack(pady=5, padx=5)

    def confirm_selection(self):
        selected_affixes = [self.listbox.get(idx) for idx in self.listbox.curselection()]
        self.selected_affixes_callback(self.item_type, selected_affixes)
        self.destroy()

class ItemCheckerApp(tk.Tk):
    def __init__(self, affixes_df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.affixes_df = affixes_df
        self.title('Item Checker')
        self.geometry('800x600')
        self.is_running = False

        # Werte zur Feinabstimmung der Bildverarbeitung:
        self.blur_value = 9
        self.adaptive_block_size = 15
        self.adaptive_C = 1
        self.contour_area_threshold = 5000
        self.contour_approximation_accuracy = 0.01

        # Initialize item types
        self.item_types = ["Helm", "Chest", "Gloves", "Pants", "Boots", "Amulet", "Ring", "Weapon", "Off-Hands", "Shield"]
        # Generate regular expressions for item types
        self.item_type_patterns = [re.compile(re.escape(item_type), re.IGNORECASE) for item_type in self.item_types]
        
        # Initialize selected affixes and other dictionaries
        self.selected_affixes = {item_type: [] for item_type in self.item_types}
        self.listboxes = {}
        self.check_vars = {}

        self.create_widgets()

    def create_widgets(self):
        # Class selection
        self.class_var = tk.StringVar()
        classes = sorted(list(set([item for sublist in self.affixes_df['Class'].tolist() for item in sublist])))
        self.class_dropdown = ttk.Combobox(self, textvariable=self.class_var, values=classes)
        self.class_dropdown.grid(row=0, column=0, columnspan=4)
        self.class_dropdown.bind('<<ComboboxSelected>>', self.clear_selection)


        # Initialize item types
        item_types = ["Helm", "Chest", "Gloves", "Pants", "Boots", "Amulet", "Ring", "Weapon", "Off-Hands", "Shield"]

        # For each item type, create a frame with a label and a button
        for i, item_type in enumerate(self.item_types):
            frame = ttk.Frame(self)
            frame.grid(row=i//5+1, column=i%5)
            label = tk.Label(frame, text=item_type)
            label.pack()
            button = ttk.Button(frame, text='Select Affixes', command=lambda item_type=item_type: self.open_affix_selection_window(item_type))
            button.pack(pady=5)
            self.listboxes[item_type] = label


        # Check button
        btn_start_stop = ttk.Button(self, text='Start/Stop', command=self.start_stop_screenshot)
        btn_start_stop.grid(row=3, column=0, columnspan=4)

        
        # Clear selection button
        btn_clear_selection = ttk.Button(self, text='Clear Selection', command=self.clear_selection)
        btn_clear_selection.grid(row=4, column=0, columnspan=4)

    def clear_selection(self, event=None):
        self.selected_affixes = {item_type: [] for item_type in self.item_types}

        # Clear the labels
        for item_type in self.item_types:
            self.listboxes[item_type].config(text=f'{item_type}:')


    def open_affix_selection_window(self, item_type):
        selected_class = self.class_var.get()
        AffixSelectionWindow(self, item_type, self.affixes_df, selected_class, self.update_selected_affixes)    
    
    def update_selected_affixes(self, item_type, selected_affixes):
        print(f"Updating selected affixes for {item_type}.")  # Debug print statement
        print(f"Received affixes: {selected_affixes}")  # Debug print statement
        self.selected_affixes[item_type] = selected_affixes
        print(f"Selected affixes: {self.selected_affixes}")  # Debug print statement
        affixes_string = "\n".join(selected_affixes)
        self.listboxes[item_type].config(text=f'{item_type}:\n{affixes_string}')




    def update_affix_list(self, event):
        # Get selected class
        selected_class = self.class_var.get()

        # Clear checkboxes and add new affixes
        for item_type, frame in self.listboxes.items():
            for widget in frame.winfo_children():
                widget.destroy()
            self.check_vars[item_type] = []

        for item_type, listbox in self.listboxes.items():
            item_affixes = self.affixes_df[
                (self.affixes_df['Class'].apply(lambda x: selected_class in x)) &
                (self.affixes_df['Type'].apply(lambda x: item_type in x))
            ]['Affixes'].tolist()

            for affix in item_affixes:
                var = IntVar()
                chk = ttk.Checkbutton(listbox, text=affix, variable=var)
                chk.pack(anchor='w')
                self.check_vars[item_type].append((var, affix))


    def start_stop_screenshot(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.start_screenshot()

    def start_screenshot(self):
        if self.is_running:
           screenshot = pyautogui.screenshot()
           self.process_screenshot(np.array(screenshot))
           threading.Timer(1, self.start_screenshot).start()


    def process_screenshot(self, image):


        # Get selected affixes
        for item_type in self.item_types:
            print(f"Selected affixes new: {self.selected_affixes}")  # Debug print statement

        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), 0)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.adaptive_block_size, self.adaptive_C)

        # Morphological operations to reduce noise
        kernel = np.ones((3,3),np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations = 2)
        binary = cv2.dilate(binary,kernel,iterations = 1)

        # Find contours in the image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")

        # Initialize found affixes
        found_affixes = []

        # Go through each contour
        for cnt in contours:
            # Ignore smaller areas
            if cv2.contourArea(cnt) > self.contour_area_threshold:
                # Approximate the contour and check if it is rectangular
                epsilon = self.contour_approximation_accuracy * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4:  # Check if it's a rectangle
                    # Get the bounding box of the contour
                    x, y, w, h = cv2.boundingRect(cnt)

                    # Extract the region of the contour for OCR
                    roi = gray[y:y+h, x:x+w]

                    # OCR
                    text = pytesseract.image_to_string(roi, config='--psm 11 --oem 3')
                    print(f"OCR output: {text}")  # Add this line

                    # Replace line breaks with spaces
                    text = re.sub(r'\n+', ' ', text)
                    print(f'Found Text: {text}')

                    # Determine the item type
                    item_type = None
                    for pattern in self.item_type_patterns:
                        match = pattern.search(text)
                        if match:
                            item_type = match.group()
                            break

                    if item_type is None:
                        messagebox.showwarning("Warning", "Could not determine item type.")
                        return
                    
                    print(f'Found item type: {item_type}')  # print found item type

                    # Get affix patterns for the determined item type
                    self.affix_patterns = [re.compile(r"([\d\.]+%?\s*"+re.escape(affix)+r")", re.IGNORECASE) for affix in self.selected_affixes.get(item_type, [])]

                    # Check if affixes exist in the OCR text
                    for pattern in self.affix_patterns:
                        match = pattern.search(text)
                        if match:
                            found_affix = match.group()
                            print(f'Found affixes: {found_affixes}')  # print found affixes
                            found_affixes.append(found_affix)
                    print(f'Found affixes: {found_affixes}')  # print found affixes
            
        result = "Item check completed! Found Affixes:\n\n" + "\n".join(found_affixes)
        # messagebox.showinfo("Result", result)

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def process_string_list(string):
        # Split by pipe
        return [item.strip() for item in string.split("|")]

    affixes_df = pd.read_csv('D4_affixes_eng.csv', sep=';', converters={'Class': process_string_list, 'Type': process_string_list})

    app = ItemCheckerApp(affixes_df)
    app.mainloop()
