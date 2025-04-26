import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Assuming you already have a face anti-spoofing model function like this:
def detect_spoofing(image):
    # Add your actual face anti-spoofing model here
    # For now, it just returns a random True/False result
    result = np.random.choice([True, False])
    return result

# GUI class definition
class AntiSpoofingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Anti-Spoofing System")
        self.root.geometry("600x400")
        
        # Instruction label
        self.label = tk.Label(root, text="Upload an image to check for spoofing", font=("Helvetica", 14))
        self.label.pack(pady=20)

        # Upload button
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        # Image display label
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # Check spoofing button
        self.process_button = tk.Button(root, text="Check Spoofing", command=self.check_spoofing)
        self.process_button.pack(pady=10)

        self.image_path = None

    def upload_image(self):
        # Open file dialog to select image
        self.image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        
        if self.image_path:
            # Display the uploaded image
            img = Image.open(self.image_path)
            img = img.resize((250, 250), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

    def check_spoofing(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Read the image using OpenCV
        img = cv2.imread(self.image_path)
        # Call the spoof detection function (adjust as needed based on your model)
        result = detect_spoofing(img)
        
        if result:
            messagebox.showinfo("Result", "The image is not a spoof!")
        else:
            messagebox.showinfo("Result", "The image is a spoof!")

# Create the main Tkinter window and run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AntiSpoofingGUI(root)
    root.mainloop()
