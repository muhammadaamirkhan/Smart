# ====================== attendance.py (Full Code - YuNet for Capture, Haar for Attendance) ======================
import tkinter as tk
from tkinter import *
import os
import cv2
from PIL import ImageTk, Image
import tkinter.font as font
import pyttsx3

# Project modules
import show_attendance
import takeImage
import trainImage
import automaticAttedance

# ====================== STYLE CONSTANTS ======================
BG_COLOR = "#2c3e50"          # Dark blue-gray
PRIMARY_COLOR = "#3498db"     # Bright blue
SECONDARY_COLOR = "#e74c3c"   # Red
TEXT_COLOR = "#ecf0f1"        # Light gray
ENTRY_BG = "#34495e"          # Darker blue-gray
BUTTON_HOVER = "#2980b9"      # Darker blue
FONT_NAME = "Segoe UI"
REGISTER_BG = "#4a2c82"       # Purple for registration
REGISTER_ENTRY_BG = "#5d3a9b"
REGISTER_HIGHLIGHT = "#b589d6"

# ====================== MAIN WINDOW ======================
window = Tk()
window.title("Smart Class Attendance System")
window.geometry("1280x720")
window.configure(background=BG_COLOR)

# ====================== CUSTOM STYLES ======================
title_font = (FONT_NAME, 28, "bold")
button_font = (FONT_NAME, 14, "bold")
label_font = (FONT_NAME, 12)

# Text-to-Speech Function
def text_to_speech(user_text):
    try:
        engine = pyttsx3.init()
        engine.say(user_text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# ====================== PATH CONFIGURATIONS ======================
# Haar Cascade Models (for both capture fallback and attendance)
haarcasecade_frontal_path = "haarcascade_frontalface_default.xml"
haarcasecade_alt_path = "haarcascade_frontalface_alt.xml"
haarcasecade_alt2_path = "haarcascade_frontalface_alt2.xml"
haarcasecade_profile_path = "haarcascade_profileface.xml"
haarcasecade_upperbody_path = "haarcascade_upperbody.xml"

# YuNet Model (for image capture only)
yunet_model_path = "face_detection_yunet_2023mar.onnx"

# Paths for training and data storage
trainimagelabel_path = "./TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "./StudentDetails/studentdetails.csv"
attendance_path = "Attendance"

# Create necessary directories
os.makedirs(trainimage_path, exist_ok=True)
os.makedirs(os.path.dirname(studentdetail_path), exist_ok=True)
os.makedirs(attendance_path, exist_ok=True)

# ====================== HEADER SECTION ======================
header_frame = Frame(window, bg=BG_COLOR)
header_frame.pack(fill=X, pady=(20, 0))

center_container = Frame(header_frame, bg=BG_COLOR)
center_container.pack(expand=True)

logo_title_frame = Frame(center_container, bg=BG_COLOR)
logo_title_frame.pack()

# Logo
try:
    logo = Image.open("UI_Image/0001.png")
    logo = logo.resize((60, 60), Image.LANCZOS)
    logo1 = ImageTk.PhotoImage(logo)
    logo_label = Label(logo_title_frame, image=logo1, bg=BG_COLOR)
    logo_label.image = logo1
    logo_label.pack(side=LEFT, padx=10)
except:
    print("Logo image not found. Skipping logo.")

# Title
title_label = Label(
    logo_title_frame, 
    text="SMART CLASS MONITORING",
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    font=title_font
)
title_label.pack(side=LEFT, padx=10)

# ====================== MAIN CONTENT ======================
content_frame = Frame(window, bg=BG_COLOR)
content_frame.pack(expand=True, fill=BOTH, padx=40, pady=20)

def create_feature_card(parent, image_path, title, command):
    card = Frame(parent, bg=ENTRY_BG, bd=0, highlightthickness=0, padx=10, pady=10)
    
    try:
        img = Image.open(image_path).resize((180, 180), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        img_label = Label(card, image=photo, bg=ENTRY_BG)
        img_label.image = photo
        img_label.pack(pady=(10, 20))
    except:
        Label(card, text="[Image]", bg=ENTRY_BG, fg=TEXT_COLOR, font=("Arial", 20)).pack(pady=(10, 20))
    
    Label(card, text=title, bg=ENTRY_BG, fg=TEXT_COLOR, font=button_font).pack()
    
    btn = Button(
        card, 
        text="Open", 
        command=command, 
        font=label_font,
        bg=PRIMARY_COLOR,
        fg=TEXT_COLOR,
        activebackground=BUTTON_HOVER,
        activeforeground=TEXT_COLOR,
        bd=0,
        padx=20,
        pady=5
    )
    btn.pack(pady=10)
    
    return card

# Register New Student Card
register_card = create_feature_card(
    content_frame,
    "UI_Image/register.png",
    "Register New Student",
    lambda: TakeImageUI()
)
register_card.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

# Take Attendance Card
attendance_card = create_feature_card(
    content_frame,
    "UI_Image/attendance.png",
    "Take Attendance",
    lambda: automaticAttedance.subjectChoose(text_to_speech)
)
attendance_card.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

# View Records Card
records_card = create_feature_card(
    content_frame,
    "UI_Image/verifyy.png",
    "View Records",
    lambda: show_attendance.subjectchoose(text_to_speech)
)
records_card.grid(row=0, column=2, padx=20, pady=10, sticky="nsew")

# Make columns expandable
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_columnconfigure(2, weight=1)

# ====================== FOOTER ======================
footer_frame = Frame(window, bg=BG_COLOR)
footer_frame.pack(side=BOTTOM, fill=X, pady=(0, 20))

exit_btn = Button(
    footer_frame, 
    text="Exit System", 
    command=window.quit,
    font=button_font,
    bg=SECONDARY_COLOR,
    fg=TEXT_COLOR,
    activebackground="#c0392b",
    activeforeground=TEXT_COLOR,
    padx=30,
    pady=10,
    bd=0
)
exit_btn.pack()

# ====================== REGISTER STUDENT UI ======================
def TakeImageUI():
    ImageUI = Toplevel(window)
    ImageUI.title("Register Student")
    ImageUI.geometry("800x600")
    ImageUI.configure(background=REGISTER_BG)
    ImageUI.resizable(False, False)
    
    # Header
    Label(
        ImageUI, 
        text="Register Your Face", 
        bg=REGISTER_BG, 
        fg=TEXT_COLOR, 
        font=title_font
    ).pack(pady=20)
    
    form_frame = Frame(ImageUI, bg=REGISTER_BG)
    form_frame.pack(expand=True, padx=40, pady=20)
    
    # Info label about YuNet
    info_label = Label(
        form_frame,
        text=" ",
        bg=REGISTER_BG,
        fg="#ffcc00",
        font=("Segoe UI", 10, "italic")
    )
    info_label.pack(pady=(0, 10))
    
    # Enrollment
    Label(form_frame, text="Enrollment No:", bg=REGISTER_BG, fg=TEXT_COLOR, font=label_font).pack(pady=5)
    txt1 = Entry(
        form_frame, width=25, validate="key",
        bg=REGISTER_ENTRY_BG, fg=TEXT_COLOR, font=label_font,
        relief="flat", highlightbackground=REGISTER_HIGHLIGHT, highlightthickness=2
    )
    txt1['validatecommand'] = (txt1.register(testVal), '%P', '%d')
    txt1.pack(pady=10)
    
    # Name
    Label(form_frame, text="Name:", bg=REGISTER_BG, fg=TEXT_COLOR, font=label_font).pack(pady=5)
    txt2 = Entry(
        form_frame, width=25,
        bg=REGISTER_ENTRY_BG, fg=TEXT_COLOR, font=label_font,
        relief="flat", highlightbackground=REGISTER_HIGHLIGHT, highlightthickness=2
    )
    txt2.pack(pady=10)
    
    # Message Area
    message = Label(
        form_frame, text="", width=45, height=4,
        bg=REGISTER_ENTRY_BG, fg=TEXT_COLOR, font=label_font,
        relief="flat", highlightbackground=REGISTER_HIGHLIGHT, highlightthickness=2
    )
    message.pack(pady=20)
    
    # Buttons
    btn_frame = Frame(form_frame, bg=REGISTER_BG)
    btn_frame.pack(pady=20)
    
    def take_image():
        l1 = txt1.get().strip()
        l2 = txt2.get().strip()
        takeImage.TakeImage(
            l1, l2,
            haarcasecade_frontal_path,
            haarcasecade_alt_path,
            haarcasecade_alt2_path,
            haarcasecade_profile_path,
            haarcasecade_upperbody_path,
            yunet_model_path,                    # YuNet model for capture
            trainimage_path,
            message,
            err_screen,
            text_to_speech
        )
        txt1.delete(0, "end")
        txt2.delete(0, "end")
    
    Button(
        btn_frame, text="Take Images", command=take_image,
        font=button_font, bg=REGISTER_HIGHLIGHT, fg="#2c3e50",
        activebackground="#9c6fc4", padx=20, pady=10, bd=0
    ).pack(side=LEFT, padx=10)
    
    Button(
        btn_frame, text="Train Model",
        command=lambda: trainImage.TrainImage(
            haarcasecade_frontal_path,
            trainimage_path,
            trainimagelabel_path,
            message,
            text_to_speech
        ),
        font=button_font, bg=REGISTER_HIGHLIGHT, fg="#2c3e50",
        activebackground="#9c6fc4", padx=20, pady=10, bd=0
    ).pack(side=LEFT, padx=10)
    
    # Info about the process
    info_frame = Frame(form_frame, bg=REGISTER_BG)
    info_frame.pack(pady=15)
    
    Label(
        info_frame,
        text="ℹ️ Capture Process:",
        bg=REGISTER_BG,
        fg="#ffcc00",
        font=("Segoe UI", 10, "bold")
    ).pack()
    
    Label(
        info_frame,
        text="• YuNet DNN model will capture high-quality face images\n"
             "• Haar cascades (5 models) act as fallback if YuNet fails\n"
             "• After capture, click 'Train Model' to train the recognition system\n"
             "• Attendance will use only Haar cascades for detection",
        bg=REGISTER_BG,
        fg=TEXT_COLOR,
        font=("Segoe UI", 9),
        justify=LEFT
    ).pack(pady=5)

# ====================== ERROR SCREEN ======================
def del_sc1():
    global sc1
    sc1.destroy()

def err_screen():
    global sc1
    sc1 = Toplevel(window)
    sc1.geometry("400x150")
    sc1.title("Warning!")
    sc1.configure(background=BG_COLOR)
    sc1.resizable(False, False)
    
    Label(
        sc1,
        text="Enrollment & Name required!",
        fg=TEXT_COLOR,
        bg=BG_COLOR,
        font=(FONT_NAME, 16, "bold")
    ).pack(pady=20)
    
    Button(
        sc1, text="OK", command=del_sc1,
        fg=TEXT_COLOR, bg=SECONDARY_COLOR,
        width=10, height=1, activebackground="#c0392b",
        font=(FONT_NAME, 14), bd=0
    ).pack()

def testVal(inStr, acttyp):
    if acttyp == "1":
        if not inStr.isdigit():
            return False
    return True

# ====================== START APPLICATION ======================
if __name__ == "__main__":
    window.mainloop()