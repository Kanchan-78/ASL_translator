import cv2
import os
import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImage

# Main Config
DATA_PATH = '../dataset'
TOTAL_CLASSES = 1
IMAGES_PER_CLASS = 1000

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DataCollectionApp:
    def __init__(self, root):
        self.master = root
        self.master.title("ASL Data Collector")
        self.master.resizable(False, False)  # Disables resizing (and maximize button)

        self.current_class = 0
        self.image_count = 0
        self.is_capturing = False

        self.cap = cv2.VideoCapture(0)
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set window size based on camera feed + controls
        self.master.geometry(f"{self.cam_width + 40}x{self.cam_height + 220}")

        self.video_frame = ctk.CTkFrame(self.master, corner_radius=10)
        self.video_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.control_frame = ctk.CTkFrame(self.master)
        self.control_frame.pack(pady=(0, 20), padx=20, fill="x")

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack()

        font_large = ctk.CTkFont(size=20, weight="bold")
        font_normal = ctk.CTkFont(size=14)

        self.status_label = ctk.CTkLabel(
            self.control_frame, 
            text=f"Press 'Start' for Class {self.current_class}", 
            font=font_large
        )
        self.status_label.pack(pady=(15, 10))

        self.progress_label = ctk.CTkLabel(
            self.control_frame, 
            text=f"Image: 0 / {IMAGES_PER_CLASS}", 
            font=font_normal
        )
        self.progress_label.pack()

        self.progress_bar = ctk.CTkProgressBar(self.control_frame, width=400)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        self.start_button = ctk.CTkButton(
            self.control_frame, 
            text=f"Start Capturing Class {self.current_class}", 
            font=font_large,
            command=self.start_capture
        )
        self.start_button.pack(pady=15, ipady=10, fill="x", padx=20)
        
        # Bind the 'c' key to the start_capture function
        self.master.bind('c', self.start_capture_event)

        self.update_frame()

    def start_capture_event(self, event):
        self.start_capture()

    def start_capture(self):
        if self.is_capturing:
            return 
            
        if self.current_class >= TOTAL_CLASSES:
            self.status_label.configure(text="All classes collected!")
            self.start_button.configure(text="Finished", state="disabled")
            return

        self.is_capturing = True
        self.image_count = 0
        self.start_button.configure(text="Capturing...", state="disabled")
        
        # Create directory for the current class
        self.class_dir = os.path.join(DATA_PATH, str(self.current_class))
        if not os.path.exists(self.class_dir):
            os.makedirs(self.class_dir)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.master.after(10, self.update_frame)
            return

        if self.is_capturing:
            if self.image_count < IMAGES_PER_CLASS:
                image_path = os.path.join(self.class_dir, f'{self.image_count}.jpg')
                cv2.imwrite(image_path, frame)
                self.image_count += 1

                progress = self.image_count / IMAGES_PER_CLASS
                self.progress_bar.set(progress)
                self.progress_label.configure(text=f"Image: {self.image_count} / {IMAGES_PER_CLASS}")
                self.status_label.configure(text=f"Collecting for Class {self.current_class}")

            else:
                self.is_capturing = False
                self.current_class += 1
                self.progress_bar.set(0)
                
                if self.current_class >= TOTAL_CLASSES:
                    self.status_label.configure(text="All classes collected!")
                    self.progress_label.configure(text="")
                    self.start_button.configure(text="Finished", state="disabled")
                else:
                    self.status_label.configure(text=f"Press 'Start' for Class {self.current_class}")
                    self.progress_label.configure(text=f"Image: 0 / {IMAGES_PER_CLASS}")
                    self.start_button.configure(
                        text=f"Start Capturing Class {self.current_class}", 
                        state="normal"
                    )

        frame_flipped = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        ctk_image = CTkImage(pil_image, size=(self.cam_width, self.cam_height))
    
        self.video_label.configure(image=ctk_image)

        # Schedule the next frame update
        self.master.after(10, self.update_frame)

    def on_close(self):
        print("[INFO] Releasing camera...")
        self.cap.release()
        self.master.destroy()


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    root = ctk.CTk()
    app = DataCollectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()