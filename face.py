
# ============================================================
# FACIAL EMOTION DETECTION SYSTEM
# ------------------------------------------------------------
# Technologies Used:
# 1. Python
# 2. OpenCV (Image Processing & Webcam)
# 3. DeepFace (Emotion Detection)
# 4. Tkinter (GUI)
#
# Features:
# 1. Upload Image
# 2. Capture Image from Webcam
# 3. Detect Facial Emotion
# 4. Display Emotion Confidence Score
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import cv2                          # OpenCV for image processing
from deepface import DeepFace       # DeepFace for emotion detection
import tkinter as tk                # Tkinter for GUI
from tkinter import filedialog      # Open file dialog
from tkinter import messagebox      # Show popup messages


# ============================================================
# FUNCTION: DETECT EMOTION
# ============================================================
# This function receives an image and:
# 1. Detects face
# 2. Predicts emotion
# 3. Draws rectangle around face
# 4. Displays result on image
# ============================================================

def detect_emotion(img):

    try:

        # Analyze image using DeepFace
        result = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=False
        )

        # Convert result into dictionary if multiple faces found
        if isinstance(result, list):
            result = result[0]

        # Get dominant emotion
        emotion = result["dominant_emotion"]

        # Get confidence score of detected emotion
        confidence = result["emotion"][emotion]

        # Get face coordinates
        region = result["region"]

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        # ----------------------------------------------------
        # Draw rectangle around detected face
        # ----------------------------------------------------
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Display emotion text above face
        # ----------------------------------------------------
        cv2.putText(
            img,
            f"{emotion} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Show result image
        # ----------------------------------------------------
        cv2.imshow("Emotion Detection Result", img)

        # Print result in console
        print("\n========== RESULT ==========")
        print("Emotion   :", emotion)
        print("Confidence:", round(confidence, 2), "%")
        print("============================")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:

        print("Error:", e)


# ============================================================
# FUNCTION: UPLOAD IMAGE
# ============================================================
# Allows user to select image from computer
# ============================================================

def upload_image():

    # Open file selection dialog
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    # If user cancels file selection
    if not file_path:
        return

    # Read image using OpenCV
    img = cv2.imread(file_path)

    # Check image loaded successfully
    if img is None:
        messagebox.showerror(
            "Error",
            "Unable to load image."
        )
        return

    # Detect emotion
    detect_emotion(img)


# ============================================================
# FUNCTION: WEBCAM CAPTURE
# ============================================================
# Opens webcam
# SPACE = Capture Image
# ESC   = Exit Webcam
# ============================================================

def webcam_capture():

    # Start webcam
    cap = cv2.VideoCapture(0)

    # Check webcam availability
    if not cap.isOpened():

        messagebox.showerror(
            "Error",
            "Cannot access webcam"
        )

        return

    # Instructions popup
    messagebox.showinfo(
        "Instructions",
        "Press SPACE to capture image\nPress ESC to exit webcam"
    )

    while True:

        # Read frame from webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Show live webcam feed
        cv2.imshow("Webcam", frame)

        # Read keyboard input
        key = cv2.waitKey(1)

        # ----------------------------------------------------
        # SPACE Key = Capture Image
        # ----------------------------------------------------
        if key == 32:

            captured = frame.copy()

            cap.release()
            cv2.destroyAllWindows()

            # Detect emotion in captured image
            detect_emotion(captured)

            return

        # ----------------------------------------------------
        # ESC Key = Exit Webcam
        # ----------------------------------------------------
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# CREATE MAIN GUI WINDOW
# ============================================================

root = tk.Tk()

# Window title
root.title("Facial Emotion Detection System")

# Window size
root.geometry("450x300")

# Disable resizing
root.resizable(False, False)


# ============================================================
# APPLICATION TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="Facial Emotion Detection System",
    font=("Arial", 16, "bold")
)

title_label.pack(pady=20)


# ============================================================
# BUTTON: UPLOAD IMAGE
# ============================================================

upload_button = tk.Button(
    root,
    text="Upload Image",
    width=25,
    height=2,
    command=upload_image
)

upload_button.pack(pady=10)


# ============================================================
# BUTTON: CAPTURE FROM WEBCAM
# ============================================================

webcam_button = tk.Button(
    root,
    text="Capture From Webcam",
    width=25,
    height=2,
    command=webcam_capture
)

webcam_button.pack(pady=10)


# ============================================================
# BUTTON: EXIT APPLICATION
# ============================================================

exit_button = tk.Button(
    root,
    text="Exit",
    width=25,
    height=2,
    command=root.destroy
)

exit_button.pack(pady=10)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()

