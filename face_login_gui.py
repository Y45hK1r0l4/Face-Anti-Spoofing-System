import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from spoof_detection import detect_spoofing  # Your spoof detection function

# Main Application Class
class FaceAntiSpoofApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Anti-Spoofing System")
        self.geometry("800x600")  # Adjust window size to fit everything
        
        # Container for all pages
        container = tk.Frame(self, bg="#f0f0f0")  # Light gray background for a clean look
        container.pack(side="top", fill="both", expand=True)
        
        # Dictionary to keep track of different frames (pages)
        self.frames = {}
        self.frames["LoginPage"] = LoginPage(parent=container, controller=self)
        self.frames["LoginPage"].pack(side="top", fill="both", expand=True)

        self.show_frame("LoginPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

# Login Page: Display live camera and show "True" or "Fake" status
class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.cap = None  # Camera object
        self.current_frame = None  # Current frame from camera
        self.is_camera_running = False  # Flag to check if camera is running
        self.login_button = None  # Variable to hold the login button
        self.back_button = None  # Variable to hold the back button

        label = tk.Label(self, text="Login with Live Face Detection", font=("Helvetica", 16), bg="#f0f0f0")
        label.pack(pady=20)

        # Button to start camera
        start_cam_button = tk.Button(self, text="Start Camera", command=self.start_camera, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        start_cam_button.pack(pady=10)

        # Label to show camera feed in a styled frame
        self.camera_frame = tk.Frame(self, bd=10, bg="#ffffff", relief="solid", width=400, height=300)  # Reduced size
        self.camera_frame.pack(pady=10)

        self.camera_label = tk.Label(self.camera_frame, bg="#ffffff")  # Camera display label
        self.camera_label.pack(padx=10, pady=10)

        # Label to show whether the face is Real or Fake
        self.status_label = tk.Label(self, text="Status: Waiting...", font=("Helvetica", 12), bg="#f0f0f0")
        self.status_label.pack(pady=10)

        # Frame to hold login and back buttons at the same level
        self.button_frame = tk.Frame(self, bg="#f0f0f0")
        self.button_frame.pack(pady=20)

    def start_camera(self):
        # Clear previous login and back buttons if they exist
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)  # Start capturing video from the webcam
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Unable to access the camera.")
                return
            self.is_camera_running = True
            self.update_frame()

            # After starting the camera, add the login and back buttons dynamically
            self.login_button = tk.Button(self.button_frame, text="Login", command=self.check_login, bg="#4CAF50", fg="white", font=("Helvetica", 12))
            self.login_button.pack(side="left", padx=10)

            self.back_button = tk.Button(self.button_frame, text="Back", command=self.go_back, bg="#f44336", fg="white", font=("Helvetica", 12))
            self.back_button.pack(side="right", padx=10)

    def update_frame(self):
        if self.is_camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame  # Save the latest frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect spoofing in the current frame
                result = detect_spoofing(frame)

                # Set the border color based on the result (real or fake)
                if result:  # If result is True, it's a real face
                    self.status_label.config(text="Status: True (Real Face Detected)", fg="green")
                    border_color = (0, 255, 0)  # Green for real face
                else:  # If result is False, it's a fake face
                    self.status_label.config(text="Status: Fake (Spoof Detected)", fg="red")
                    border_color = (0, 0, 255)  # Red for fake face

                # Perform face detection (Haar Cascade)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

                # If faces are detected, draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 3)  # Draw border

                # Convert frame back to RGB and display in the Tkinter GUI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((350, 200), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            # Schedule the next frame update only if the camera is still running
            if self.is_camera_running:
                self.after(10, self.update_frame)

    def check_login(self):
        if self.current_frame is None:
            messagebox.showerror("Error", "No face detected.")
            return

        # Detect spoofing again on current frame before allowing login
        result = detect_spoofing(self.current_frame)

        if result:  # If result is True, it's a real face
            messagebox.showinfo("Login Successful", "Welcome! You are verified.")
        else:  # If result is False, it's a fake face
            messagebox.showerror("Access Denied", "Spoof Detected! You can't login.")

    def go_back(self):
        if self.cap:
            self.cap.release()  # Release the camera resource
            self.cap = None
        self.is_camera_running = False  # Stop updating the frame
        self.camera_label.configure(image='')  # Clear the camera label
        self.status_label.config(text="Status: Waiting...", fg="black")  # Reset status label
        
        # Clear buttons before going back
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        self.controller.show_frame("LoginPage")  # Go back to the login page

# Running the app
if __name__ == "__main__":
    app = FaceAntiSpoofApp()
    app.mainloop()
