import tkinter as tk
from tkinter import filedialog
import cv2
import os
import shutil


class MyGUI:
    def __init__(self):
        self.MyEmail = "something"
        self.root = tk.Tk()
        self.root.geometry("500x500")
        self.root.title("Face_recognition_system")

        self.label = tk.Label(self.root, text="Enter your email", font=('Arial', 18))
        self.label.place(x=0, y=0, height=100, width=180)

        self.myentry = tk.Entry(self.root)
        self.myentry.pack(padx=20, pady=40)

        self.button = tk.Button(self.root, text="Enter", font=('Arial', 18), command=self.show_message)
        self.button.pack(padx=10, pady=10)

        self.import_button = tk.Button(self.root, text="Import Faces", font=('Arial', 18), command=self.import_faces)
        self.import_button.pack(padx=10, pady=10)
        self.label2 = tk.Label(self.root, text="'r' - face recognition on/off\n'y' - movement detection\n't'- mood analysis\n'w' - summon thi menu\n'q' - exit", font=('Arial', 12))
        self.label2.place(x=155, y=310, height=100, width=180)

        self.root.mainloop()

    def show_message(self):
        self.MyEmail = self.myentry.get()
        print(self.MyEmail)
        self.root.destroy()

    def return_email(self):
        return self.MyEmail

    def import_faces(self):
        file_paths = filedialog.askopenfilenames(title="Select image files", filetypes=[("Image files", "*.jpg;*.png")])
        for file_path in file_paths:
            # Copy the selected images to the 'Faces' folder
            shutil.copy(file_path, 'Faces')


