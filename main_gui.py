import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import threading
import cv2
import time

from src.detector import detect_plate
from src.ocr import extract_text
from src.classifiers.state_classifier import classify

# Configure the modern look
ctk.set_appearance_mode("Dark")  # Default to Dark Mode
ctk.set_default_color_theme("dark-blue")  # Base theme, overridden by our custom colors

class ModernLicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Malaysian License Plate Recognition (LPR) & State Identification")
        self.root.geometry("1100x700")
        self.root.grid_rowconfigure(1, weight=0)
        
        self.image_path = None
        
        #  UI Grid Layout 
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)


        # LEFT SIDEBAR (Controls)
        self.sidebar = ctk.CTkFrame(self.root, width=280, corner_radius=0, fg_color="#1a1a1a")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1) # Pushes the mode toggle to the bottom

        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="LPR & SIS\nPipeline", 
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#F7F7F7"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 30))

        # Action Buttons
        self.btn_load = ctk.CTkButton(
            self.sidebar, 
            text="1. Load Vehicle Image", 
            command=self.load_image,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#F83939", 
            hover_color="#a50000",
            height=40
        )
        self.btn_load.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_process = ctk.CTkButton(
            self.sidebar, 
            text="2. Run LPR & SIS", 
            command=self.process_image,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#46433B", 
            hover_color="#3d3c3c",
            text_color="#ffffff",
            state="disabled",
            height=40
        )
        self.btn_process.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Dark/Light Mode Toggle
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", font=ctk.CTkFont(size=12))
        self.mode_label.grid(row=4, column=0, padx=20, pady=(0, 0), sticky="s")
        
        self.mode_menu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode_event,
            fg_color="#2b2b2b",
            button_color="#3b3b3b",
            button_hover_color="#4b4b4b"
        )
        self.mode_menu.grid(row=5, column=0, padx=20, pady=(10, 30), sticky="s")

        
        # RIGHT PANEL (Display & Results)
        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=3) # Image area gets more vertical space
        self.main_frame.grid_rowconfigure(1, weight=1) # Text area gets less vertical space
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Image Display Area
        self.img_frame = ctk.CTkFrame(self.main_frame, corner_radius=15)
        self.img_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        self.img_frame.grid_rowconfigure(0, weight=1)
        self.img_frame.grid_columnconfigure(0, weight=1)

        self.img_label = ctk.CTkLabel(
            self.img_frame, 
            text="No Image Loaded", 
            font=ctk.CTkFont(size=18),
            text_color="gray"
        )
        self.img_label.grid(row=0, column=0)

        # Terminal / Results Output Box
        self.result_text = ctk.CTkTextbox(
            self.main_frame, 
            font=ctk.CTkFont(family="Consolas", size=14),
            corner_radius=15,
            border_width=2,
            border_color="#333333",
            fg_color="#0d0d0d",
            text_color="#00FF41" 
        )
        self.result_text.grid(row=1, column=0, sticky="nsew")
        self.result_text.insert("0.0", "> System Initialized. Awaiting image input...\n")
        self.result_text.configure(state="disabled")
        self.status_bar = ctk.CTkFrame(self.root, height=6, corner_radius=0, fg_color="#1f6feb") # Default Blue
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def log_result(self, text):
        
        def append_char(char):
            self.result_text.configure(state="normal")
            self.result_text.insert("end", char)
            self.result_text.see("end")
            self.result_text.configure(state="disabled")

        # Type each character with a tiny delay
        for char in text + "\n":
            self.root.after(0, append_char, char)
            time.sleep(0.005)

    def load_image(self):
        filetypes = (("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        filepath = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)
        
        if filepath:
            self.image_path = filepath
            
            # Load and display the image using CTkImage
            img_data = Image.open(filepath)
            
            # Calculate dynamic sizing to fit the screen nicely
            my_image = ctk.CTkImage(light_image=img_data, dark_image=img_data, size=(600, 400))
            
            self.img_label.configure(image=my_image, text="")
            self.btn_process.configure(state="normal")
            
            self.result_text.configure(state="normal")
            self.result_text.delete("0.0", "end")
            self.result_text.configure(state="disabled")
            self.log_result(f"> Image loaded: {os.path.basename(filepath)}")
            self.log_result("> Ready to process.")

    def process_image(self):
        """Disables the button and spins up a background thread so the GUI doesn't freeze."""
        self.btn_process.configure(state="disabled")
        self.btn_load.configure(state="disabled")
        self.log_result("\n> Running pipeline in background thread...")
        
        # Start the heavy lifting in a separate thread
        worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        worker_thread.start()

    def _pipeline_worker(self):
        """This is the background thread where the heavy AI math happens."""
        try:
            # Change status bar to YELLOW (Processing)
            self.root.after(0, lambda: self.status_bar.configure(fg_color="#F9D348"))
            
            self.log_result("> [1/3] Detecting plate candidates...")
            
            detection_result = detect_plate(self.image_path)
            candidates = detection_result.get("_candidate_objects", [])
            resized_img = detection_result.get("resized_image")
            
            if not candidates or resized_img is None:
                self.log_result("> [ERROR] No potential plate regions found at all.")
                self.root.after(0, lambda: self.status_bar.configure(fg_color="#da3633")) # RED
                return

            valid_plate_found = False
            
            # THE VALIDATION LOOP
            for i, candidate in enumerate(candidates):
                self.log_result(f"> Scanning candidate {i+1}/{len(candidates)}...")
                x, y, w, h = candidate.bbox
                
                #ANIMATION: The Yellow '?' Searching Box 
                img_scan = resized_img.copy()
                img_scan = cv2.cvtColor(img_scan, cv2.COLOR_BGR2RGB)
                cv2.rectangle(img_scan, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(img_scan, " ?", (x, max(20, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                
                img_pil = Image.fromarray(img_scan)
                scan_image = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(600, 400))
                self.root.after(0, lambda: self.img_label.configure(image=scan_image))
                time.sleep(0.3) # Pause for animation
                
                # TEST 1: Try the EXACT crop first (Best for standard cars)
                crop_exact = resized_img[y:y+h, x:x+w]
                plate_text, confidence = extract_text(crop_exact)
                final_box = (x, y, w, h) # Store the successful coordinates
                
                # TEST 2: If exact fails, try the EXPANDED crop (Best for square trucks)
                if not plate_text:
                    pad_top = int(h * 0.6)
                    pad_bottom = int(h * 0.2)
                    pad_x = int(w * 0.05)
                    
                    y1 = max(0, y - pad_top)
                    y2 = min(resized_img.shape[0], y + h + pad_bottom)
                    x1 = max(0, x - pad_x)
                    x2 = min(resized_img.shape[1], x + w + pad_x)
                    
                    crop_expanded = resized_img[y1:y2, x1:x2]
                    plate_text, confidence = extract_text(crop_expanded)
                    final_box = (x1, y1, (x2-x1), (y2-y1)) # Update to the expanded coordinates
                


                if plate_text:
                    valid_plate_found = True
                    bx, by, bw, bh = final_box # Unpack whichever box actually worked
                    
                    self.log_result(f"> [OK] Real plate identified! Text: {plate_text.replace(chr(10), ' ')}")
                    
                    # Draw GREEN Box Around the VALIDATED Area 
                    img_cv = resized_img.copy()
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    
                    cv2.rectangle(img_cv, (bx, by), (bx + bw, by + bh), (0, 255, 0), 4)
                    cv2.putText(img_cv, "LOCKED: PLATE", (bx, max(20, by - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Picture-in-Picture Overlay 
                    plate_inset = img_cv[by:by+bh, bx:bx+bw].copy()
                    inset_h, inset_w = plate_inset.shape[:2]
                    scale_factor = 2.5
                    new_w = int(inset_w * scale_factor)
                    new_h = int(inset_h * scale_factor)
                    
                    if new_w > img_cv.shape[1] // 2:
                        new_w = img_cv.shape[1] // 2
                        new_h = int(new_w * (inset_h / max(1, inset_w)))
                    
                    plate_inset = cv2.resize(plate_inset, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    cv2.rectangle(plate_inset, (0, 0), (new_w-1, new_h-1), (0, 255, 0), 4)
                    
                    pad = 10
                    start_y = img_cv.shape[0] - new_h - pad
                    start_x = img_cv.shape[1] - new_w - pad
                    
                    if start_y > 0 and start_x > 0:
                        img_cv[start_y:start_y+new_h, start_x:start_x+new_w] = plate_inset

                    img_pil = Image.fromarray(img_cv)
                    boxed_image = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(600, 400))
                    self.root.after(0, lambda: self.img_label.configure(image=boxed_image))
                    
                    # --- Step 3: State Classification ---
                    self.log_result("\n> [3/3] Classifying State/Type...")
                    time.sleep(0.5) 
                    result = classify(plate_text, confidence=confidence)
                    
                    if result:
                        self.log_result("\n" + "="*30)
                        self.log_result("       FINAL RESULTS")
                        self.log_result("="*30)
                        clean_print_text = result['plate_text'].replace('\n', ' ')
                        self.log_result(f" Plate Text : {clean_print_text}")
                        self.log_result(f" Plate Type : {result['plate_type']}")
                        self.log_result(f" State      : {result['state']}")
                        self.log_result(f" Confidence : {result['confidence']:.2f}")
                        self.log_result("="*30 + "\n")
                        
                        # Status bar to GREEN (Success)
                        self.root.after(0, lambda: self.status_bar.configure(fg_color="#238636"))
                    else:
                        self.log_result(f"\n> [ERROR] Could not classify state.")
                        self.root.after(0, lambda: self.status_bar.configure(fg_color="#da3633")) # RED
                    
                    break 
                else:
                    self.log_result(f"> False alarm. Rejecting candidate.")
            
            if not valid_plate_found:
                 self.log_result("\n> [ERROR] Checked all candidates. No readable text found.")
                 self.root.after(0, lambda: self.status_bar.configure(fg_color="#da3633")) 
                 
        except Exception as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Pipeline Error", f"An error occurred:\n{str(err)}"))
            self.root.after(0, lambda: self.status_bar.configure(fg_color="#da3633")) 
        
        finally:
            self.root.after(0, lambda: self.btn_process.configure(state="normal"))
            self.root.after(0, lambda: self.btn_load.configure(state="normal"))
            
            # Reset back to BLUE after a few seconds if you want
            
if __name__ == "__main__":
    root = ctk.CTk()
    app = ModernLicensePlateGUI(root)
    root.mainloop()