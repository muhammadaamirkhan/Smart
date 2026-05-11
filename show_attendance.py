import pandas as pd
from glob import glob
import os
import tkinter
import csv
import tkinter as tk
from tkinter import *

# Color Scheme Constants
PURPLE_BG = "#4a2c82"          # Main purple background
PURPLE_DARK = "#3a1c72"        # Darker purple for accents
PURPLE_LIGHT = "#b589d6"       # Light purple for highlights
TEXT_COLOR = "#ffffff"         # White text
ACCENT_COLOR = "#ffcc00"       # Yellow for accents
ENTRY_BG = "#5d3a9b"           # Purple for entry fields

def subjectchoose(text_to_speech):
    def calculate_attendance():
        Subject = tx.get()
        if Subject=="":
            t='Please enter the subject name.'
            text_to_speech(t)
    
        filenames = glob(
            f"Attendance\\{Subject}\\{Subject}*.csv"
        )
        df = [pd.read_csv(f) for f in filenames]
        newdf = df[0]
        for i in range(1, len(df)):
            newdf = newdf.merge(df[i], how="outer")
        newdf.fillna(0, inplace=True)
        newdf["Attendance"] = 0
        for i in range(len(newdf)):
            newdf["Attendance"].iloc[i] = str(int(round(newdf.iloc[i, 2:-1].mean() * 100)))+'%'
        newdf.to_csv(f"Attendance\\{Subject}\\attendance.csv", index=False)

        root = tkinter.Tk()
        root.title("Attendance of "+Subject)
        root.configure(background=PURPLE_BG)
        cs = f"Attendance\\{Subject}\\attendance.csv"
        
        # Create a frame for the table
        table_frame = Frame(root, bg=PURPLE_BG)
        table_frame.pack(pady=10)
        
        # Create a canvas for scrolling
        canvas = Canvas(table_frame, bg=PURPLE_BG, highlightthickness=0)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = Scrollbar(table_frame, orient=VERTICAL, command=canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create another frame inside the canvas
        inner_frame = Frame(canvas, bg=PURPLE_BG)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        with open(cs) as file:
            reader = csv.reader(file)
            r = 0

            for col in reader:
                c = 0
                for row in col:
                    label = tkinter.Label(
                        inner_frame,
                        width=15,
                        height=1,
                        fg=TEXT_COLOR,
                        font=("Verdana", 12),
                        bg=PURPLE_DARK,
                        text=row,
                        relief=tkinter.FLAT,
                        padx=5,
                        pady=5
                    )
                    label.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
                    c += 1
                r += 1
        
        # Update the scrollregion
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Configure column weights
        for i in range(c):
            inner_frame.grid_columnconfigure(i, weight=1)
            
        # Add buttons frame
        btn_frame = Frame(root, bg=PURPLE_BG)
        btn_frame.pack(pady=10)
        
        Button(
            btn_frame,
            text="Close",
            command=root.destroy,
            bg=PURPLE_LIGHT,
            fg=PURPLE_DARK,
            font=("Verdana", 12, "bold"),
            relief=FLAT,
            padx=20,
            pady=5
        ).pack()
        
        root.mainloop()
        print(newdf)

    subject = Tk()
    subject.title("Subject Attendance")
    subject.geometry("600x300")
    subject.resizable(0, 0)
    subject.configure(background=PURPLE_BG)
    
    # Header
    header = Frame(subject, bg=PURPLE_BG)
    header.pack(fill=X, pady=10)
    
    titl = tk.Label(
        header,
        text="Which Subject of Attendance?",
        bg=PURPLE_BG,
        fg=ACCENT_COLOR,
        font=("Verdana", 20, "bold")
    )
    titl.pack()
    
    # Input frame
    input_frame = Frame(subject, bg=PURPLE_BG)
    input_frame.pack(pady=20)
    
    sub = tk.Label(
        input_frame,
        text="Enter Subject:",
        width=12,
        height=2,
        bg=PURPLE_BG,
        fg=TEXT_COLOR,
        font=("Verdana", 12)
    )
    sub.grid(row=0, column=0, padx=5)

    tx = tk.Entry(
        input_frame,
        width=20,
        bg=ENTRY_BG,
        fg=TEXT_COLOR,
        relief=FLAT,
        font=("Verdana", 14),
        highlightbackground=PURPLE_LIGHT,
        highlightthickness=2,
        insertbackground=TEXT_COLOR
    )
    tx.grid(row=0, column=1, padx=10)
    
    # Button frame
    btn_frame = Frame(subject, bg=PURPLE_BG)
    btn_frame.pack(pady=20)
    
    def Attf():
        sub = tx.get()
        if sub == "":
            t="Please enter the subject name!!!"
            text_to_speech(t)
        else:
            os.startfile(f"Attendance\\{sub}")

    # Check Sheets button
    attf = tk.Button(
        btn_frame,
        text="Check Sheets",
        command=Attf,
        bg=PURPLE_LIGHT,
        fg=PURPLE_DARK,
        activebackground="#a579ce",
        activeforeground=PURPLE_DARK,
        font=("Verdana", 12, "bold"),
        height=1,
        width=12,
        relief=FLAT,
        padx=10
    )
    attf.grid(row=0, column=0, padx=10)
    
    # View Attendance button
    fill_a = tk.Button(
        btn_frame,
        text="View Attendance",
        command=calculate_attendance,
        bg=PURPLE_LIGHT,
        fg=PURPLE_DARK,
        activebackground="#a579ce",
        activeforeground=PURPLE_DARK,
        font=("Verdana", 12, "bold"),
        height=1,
        width=15,
        relief=FLAT,
        padx=10
    )
    fill_a.grid(row=0, column=1, padx=10)
    
    subject.mainloop()