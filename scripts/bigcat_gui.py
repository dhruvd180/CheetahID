import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import threading
import json
from datetime import datetime
import sys

# Import your inference pipeline
try:
    from inference_pipeline import EnhancedCheetahIdentifier
    INFERENCE_AVAILABLE = True
    print("Successfully imported from inference_pipeline.py")
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Inference pipeline not available - running in demo mode")


class RoundedFrame(tk.Frame):
    def __init__(self, parent, radius=15, bg_color="#1C1C1E", **kwargs):
        super().__init__(parent, **kwargs)
        self.radius = radius
        self.bg_color = bg_color
        self.configure(bg=parent.cget('bg'))
        
        # Create rounded rectangle canvas
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=parent.cget('bg'))
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self._draw_rounded_rect)
        
    def _draw_rounded_rect(self, event=None):
        self.canvas.delete("bg")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        if w <= 1 or h <= 1:
            return
        
        # Create rounded rectangle coordinates
        points = []
        r = min(self.radius, w//4, h//4)  # Adaptive radius
        
        # Top-left corner
        for i in range(r):
            x = r - (r**2 - (r-i)**2)**0.5
            points.extend([x, i])
        
        # Top edge
        points.extend([r, 0, w-r, 0])
        
        # Top-right corner
        for i in range(r):
            x = w - r + (r**2 - (r-i)**2)**0.5
            points.extend([x, i])
        
        # Right edge
        points.extend([w, r, w, h-r])
        
        # Bottom-right corner
        for i in range(r):
            x = w - r + (r**2 - (i)**2)**0.5
            points.extend([x, h-r+i])
        
        # Bottom edge
        points.extend([w-r, h, r, h])
        
        # Bottom-left corner
        for i in range(r):
            x = r - (r**2 - (i)**2)**0.5
            points.extend([x, h-r+i])
        
        # Left edge
        points.extend([0, h-r, 0, r])
        
        if len(points) >= 6:
            self.canvas.create_polygon(points, fill=self.bg_color, outline="", tags="bg")

class ModernButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, bg_color="#FFB366", hover_color="#E5661A", 
                 text_color="white", radius=12, font=("SF Pro Display", 11, "bold"), 
                 width=120, height=36, **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0, 
                        bg=parent.cget('bg'), **kwargs)
        
        self.text = text
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.radius = radius
        self.font = font
        self.is_hovered = False
        self.is_disabled = False
        
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Configure>', self._draw)
        
        self._draw()
    
    def _draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        
        if w <= 1 or h <= 1:
            return
        
        # Choose color based on state
        if self.is_disabled:
            color = "#6C6C70"
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.bg_color
        
        # Draw rounded rectangle with adaptive radius
        r = min(self.radius, w//2, h//2)
        self.create_oval(0, 0, 2*r, 2*r, fill=color, outline="")
        self.create_oval(w-2*r, 0, w, 2*r, fill=color, outline="")
        self.create_oval(0, h-2*r, 2*r, h, fill=color, outline="")
        self.create_oval(w-2*r, h-2*r, w, h, fill=color, outline="")
        self.create_rectangle(r, 0, w-r, h, fill=color, outline="")
        self.create_rectangle(0, r, w, h-r, fill=color, outline="")
        
        # Draw text with adaptive font size
        font_size = max(8, min(14, h//3))
        adaptive_font = (self.font[0], font_size, self.font[2] if len(self.font) > 2 else 'normal')
        text_color = "#AEAEB2" if self.is_disabled else self.text_color
        self.create_text(w//2, h//2, text=self.text, fill=text_color, font=adaptive_font)
    
    def _on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()
    
    def _on_enter(self, event):
        if not self.is_disabled:
            self.is_hovered = True
            self._draw()
    
    def _on_leave(self, event):
        self.is_hovered = False
        self._draw()
    
    def configure_state(self, state):
        self.is_disabled = (state == 'disabled')
        self._draw()

class ResponsiveImageFrame(tk.Frame):
    def __init__(self, parent, radius=20, **kwargs):
        super().__init__(parent, **kwargs)
        self.radius = radius
        self.configure(bg=parent.cget('bg'))
        
        # Create container for rounded image
        self.image_container = tk.Canvas(self, highlightthickness=0, bg=self.cget('bg'))
        self.image_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.current_image = None
        self.placeholder_text = "No image"
        self.photo = None  # Keep reference to prevent garbage collection
        
        # Bind resize event
        self.image_container.bind('<Configure>', self._on_resize)
        
        # Force initial display
        self.after(100, self._update_display)
    
    def set_image(self, image_path_or_pil, placeholder_text="No image"):
        """Set image from path or PIL Image object"""
        self.placeholder_text = placeholder_text
        
        try:
            if isinstance(image_path_or_pil, str):
                # Load from path
                self.current_image = Image.open(image_path_or_pil)
            else:
                # Assume PIL Image
                self.current_image = image_path_or_pil
            
            self._update_display()
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self.current_image = None
            self._update_display()
    
    def clear_image(self):
        """Clear the current image"""
        self.current_image = None
        self._update_display()
    
    def _on_resize(self, event):
        """Handle container resize"""
        self._update_display()
    
    def _update_display(self, event=None):
        """Update the image display"""
        self.image_container.delete("all")
        
        # Get current canvas size
        self.image_container.update_idletasks()
        w = self.image_container.winfo_width()
        h = self.image_container.winfo_height()
        
        # Use minimum size if canvas not ready
        if w <= 1:
            w = 200
        if h <= 1:
            h = 200
        
        if self.current_image:
            try:
                # Calculate size to fit container while maintaining aspect ratio
                img_w, img_h = self.current_image.size
                
                # Calculate scaling to fit container with padding
                padding = max(20, min(w, h) // 20)  # Adaptive padding
                container_w = w - padding
                container_h = h - padding
                
                if container_w > 0 and container_h > 0:
                    scale_w = container_w / img_w
                    scale_h = container_h / img_h
                    scale = min(scale_w, scale_h, 1.0)  # Don't upscale small windows
                    
                    new_w = max(1, int(img_w * scale))
                    new_h = max(1, int(img_h * scale))
                    
                    # Resize image
                    resized_img = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage and keep reference
                    self.photo = ImageTk.PhotoImage(resized_img)
                    
                    # Center the image
                    x = w // 2
                    y = h // 2
                    
                    # Draw image
                    self.image_container.create_image(x, y, image=self.photo)
                    
            except Exception as e:
                print(f"Error displaying image: {e}")
                # Fall back to placeholder
                font_size = max(10, min(16, min(w, h) // 20))
                self.image_container.create_text(
                    w//2, h//2, 
                    text="Error loading image",
                    fill="#8E8E93",
                    font=("Inter", font_size)
                )
        else:
            # Draw placeholder with adaptive font
            font_size = max(10, min(16, min(w, h) // 20))
            self.image_container.create_text(
                w//2, h//2, 
                text=self.placeholder_text,
                fill="#8E8E93",
                font=("Inter", font_size)
            )

class BigCatIDGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BigCatID")
        self.root.geometry("1400x900")
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Safari-inspired color scheme
        self.colors = {
            'bg': '#F5F1E8',           # Light gray background
            'card': '#FFFFFF',          # White cards
            'card_dark': '#1C1C1E',     # Dark cards
            'sidebar': "#ffab4b",       # Sidebar background
            'primary': '#FF8C42',       # Blue primary
            'secondary': '#FFB366',     # Purple secondary
            'success': "#00FF40",       # Green success
            'warning': "#EEFF00",       
            'danger': "#FF0D00",        
            'text_primary': '#000000',  
            'text_secondary': '#8E8E93', 
            'text_light': '#FFFFFF',    
            'border': '#E5E5EA',        
            'accent': '#E5661A'         
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Application state
        self.current_image_path = None
        self.identification_result = None
        self.identifier = None
        self.reference_images_dir = "data/raw/"  # Set default reference directory
        
        # Responsive layout variables
        self.sidebar_min_width = 300
        self.sidebar_max_width = 400
        
        self.setup_fonts()
        self.create_widgets()
        self.initialize_inference()
        
        # Bind window resize event
        self.root.bind('<Configure>', self._on_window_resize)
        
    def setup_fonts(self):
        """Setup modern font families with responsive sizing"""
        try:
            
            self.base_fonts = {
                'title': ('Inter', 32, 'bold'),
                'heading': ('Inter', 18),
                'subheading': ('Inter', 14),
                'body': ('SF Pro Display', 12),
                'caption': ('SF Pro Display', 10),
                'button': ('SF Pro Display', 11, 'bold')
            }
        except:
            # Fallback to system fonts
            self.base_fonts = {
                'title': ('Inter', 32, 'bold'),
                'heading': ('Inter', 18, 'bold'),
                'subheading': ('Inter', 14, 'bold'),
                'body': ('Inter', 12),
                'caption': ('Inter', 10),
                'button': ('Inter', 11, 'bold')
            }
        
        self.fonts = self.base_fonts.copy()
        
    def _update_fonts_for_size(self):
        """Update font sizes based on window size"""
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        
        # Calculate scale factor based on window size
        base_width, base_height = 1400, 900
        scale_factor = min(width / base_width, height / base_height)
        scale_factor = max(0.7, min(1.3, scale_factor))  # Limit scaling
        
        # Update fonts
        for key, (family, base_size, *style) in self.base_fonts.items():
            new_size = max(8, int(base_size * scale_factor))
            self.fonts[key] = (family, new_size, *style)
        
    def _on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            self._update_fonts_for_size()
            self._update_sidebar_width()
            
    def _update_sidebar_width(self):
        """Update sidebar width based on window size"""
        window_width = self.root.winfo_width()
        
        # Calculate responsive sidebar width (20-30% of window width)
        sidebar_width = max(self.sidebar_min_width, 
                           min(self.sidebar_max_width, window_width * 0.23))
        
        self.sidebar_container.configure(width=int(sidebar_width))
        
    def create_widgets(self):
        """Create and layout all GUI components"""
        
        # Main container with responsive padding
        self.main_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_container.pack(fill='both', expand=True)
        
        # Bind to update padding
        self.main_container.bind('<Configure>', self._update_padding)
        
        # Header
        self.create_header()
        
        # Content area
        self.create_content_area()
        
    def _update_padding(self, event=None):
        """Update padding based on window size"""
        if event and event.widget == self.main_container:
            width = event.width
            height = event.height
            
            # Responsive padding (1-3% of window dimensions)
            pad_x = max(10, min(50, width * 0.02))
            pad_y = max(10, min(40, height * 0.02))
            
            # Update header padding
            if hasattr(self, 'header_frame'):
                self.header_frame.pack_configure(padx=pad_x, pady=(pad_y, pad_y//2))
            
            # Update content padding
            if hasattr(self, 'content_frame'):
                self.content_frame.pack_configure(padx=pad_x, pady=(0, pad_y))
    
    def create_header(self):
        """Create responsive header"""
        self.header_frame = tk.Frame(self.main_container, bg=self.colors['bg'])
        self.header_frame.pack(fill='x', padx=30, pady=(30, 15))
        
        # Title with icon
        title_frame = tk.Frame(self.header_frame, bg=self.colors['bg'])
        title_frame.pack(expand=True)
        
        self.title_label = tk.Label(title_frame, text="BigCatID", 
                              font=self.fonts['title'],
                              fg=self.colors['text_primary'],
                              bg=self.colors['bg'])
        self.title_label.pack()
        
        self.subtitle_label = tk.Label(title_frame, text="Advanced Cheetah Identification System", 
                                 font=self.fonts['body'],
                                 fg=self.colors['text_secondary'],
                                 bg=self.colors['bg'])
        self.subtitle_label.pack(pady=(5, 0))
        
    def create_content_area(self):
        """Create responsive content area"""
        self.content_frame = tk.Frame(self.main_container, bg=self.colors['bg'])
        self.content_frame.pack(fill='both', expand=True, padx=30, pady=(0, 30))
        
        # Create sidebar and main area
        self.create_sidebar()
        self.create_main_area()
        
    def create_sidebar(self):
        """Create responsive left sidebar with controls"""
        # Sidebar container with responsive width
        self.sidebar_container = RoundedFrame(self.content_frame, radius=20, bg_color=self.colors['sidebar'])
        self.sidebar_container.pack(side='left', fill='y', padx=(0, 15))
        
        # Initial width
        self._update_sidebar_width()
        self.sidebar_container.pack_propagate(False)
        
        # Scrollable sidebar content
        self.sidebar_canvas = tk.Canvas(self.sidebar_container.canvas, bg=self.colors['card'], 
                                       highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.sidebar_container.canvas, orient="vertical", 
                                 command=self.sidebar_canvas.yview)
        self.scrollable_frame = tk.Frame(self.sidebar_canvas, bg=self.colors['card'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        )
        
        self.sidebar_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollable elements
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Position sidebar canvas in the rounded frame
        self.sidebar_container.canvas.create_window(0, 0, window=self.sidebar_canvas, anchor='nw')
        self.sidebar_container.canvas.bind('<Configure>', self._resize_sidebar_canvas)
        
        # Create sidebar sections
        self.create_upload_section()
        self.create_settings_section()
        self.create_status_section()
        self.create_identify_section()
        
    def _resize_sidebar_canvas(self, event):
        """Resize sidebar canvas to fill rounded frame"""
        self.sidebar_canvas.configure(width=event.width, height=event.height)
        
    def create_upload_section(self):
        """Create upload section with responsive elements"""
        upload_section = tk.Frame(self.scrollable_frame, bg=self.colors['card'])
        upload_section.pack(fill='x', padx=20, pady=(20, 15))
        
        self.upload_title = tk.Label(upload_section, text="Upload Image", 
                               font=self.fonts['heading'],
                               fg=self.colors['text_primary'],
                               bg=self.colors['card'])
        self.upload_title.pack(anchor='w', pady=(0, 15))
        
        # Upload button - responsive width
        self.upload_button = ModernButton(upload_section, text="Select Image", 
                                        command=self.upload_image,
                                        bg_color=self.colors['primary'],
                                        height=33)
        self.upload_button.pack(fill='x', pady=5)
        
        # Image info
        self.image_info_label = tk.Label(upload_section, text="No image selected", 
                                        font=self.fonts['caption'],
                                        fg=self.colors['text_secondary'],
                                        bg=self.colors['card'],
                                        wraplength=250)
        self.image_info_label.pack(pady=(10, 0), anchor='w')
        
    def create_settings_section(self):
        """Create settings section"""
        # Separator
        separator1 = tk.Frame(self.scrollable_frame, bg=self.colors['border'], height=1)
        separator1.pack(fill='x', padx=20, pady=15)
        
        settings_section = tk.Frame(self.scrollable_frame, bg=self.colors['card'])
        settings_section.pack(fill='x', padx=20, pady=15)
        
        self.settings_title = tk.Label(settings_section, text="⚙️ Settings", 
                                 font=self.fonts['heading'],
                                 fg=self.colors['text_primary'],
                                 bg=self.colors['card'])
        self.settings_title.pack(anchor='w', pady=(0, 15))
        
        # Threshold selection
        self.threshold_label = tk.Label(settings_section, text="Confidence Threshold", 
                                  font=self.fonts['subheading'],
                                  fg=self.colors['text_primary'],
                                  bg=self.colors['card'])
        self.threshold_label.pack(anchor='w', pady=(0, 10))
        
        self.threshold_var = tk.StringVar(value='moderate')
        
        # Threshold buttons - responsive
        self.threshold_frame = tk.Frame(settings_section, bg=self.colors['card'])
        self.threshold_frame.pack(fill='x', pady=(0, 15))
        
        self.threshold_buttons = {}
        levels = ['conservative', 'moderate', 'liberal']
        
        for i, level in enumerate(levels):
            btn = ModernButton(self.threshold_frame, text=level.capitalize(),
                              command=lambda l=level: self.set_threshold(l),
                              bg_color=self.colors['primary'] if level == 'moderate' else self.colors['border'],
                              text_color=self.colors['text_light'] if level == 'moderate' else self.colors['text_secondary'],
                              height=32,
                              font=self.fonts['caption'])
            btn.pack(side='left', fill='x', expand=True, padx=(0, 5) if i < len(levels)-1 else 0)
            self.threshold_buttons[level] = btn
        
        # Model configuration - responsive button
        self.config_button = ModernButton(settings_section, text="Configure Paths", 
                                   command=self.configure_paths,
                                   bg_color=self.colors['secondary'],
                                   height=32)
        self.config_button.pack(fill='x', pady=(10, 0))
        
    def create_status_section(self):
        """Create status section"""
        # Separator
        separator2 = tk.Frame(self.scrollable_frame, bg=self.colors['border'], height=1)
        separator2.pack(fill='x', padx=20, pady=15)
        
        status_section = tk.Frame(self.scrollable_frame, bg=self.colors['card'])
        status_section.pack(fill='x', padx=20, pady=15)
        
        self.status_title = tk.Label(status_section, text="📊 System Status", 
                               font=self.fonts['heading'],
                               fg=self.colors['text_primary'],
                               bg=self.colors['card'])
        self.status_title.pack(anchor='w', pady=(0, 15))
        
        self.status_label = tk.Label(status_section, text="Ready", 
                                    font=self.fonts['body'],
                                    fg=self.colors['success'],
                                    bg=self.colors['card'])
        self.status_label.pack(anchor='w')
        
        self.model_status_label = tk.Label(status_section, 
                                          text="Model: Not loaded" if not INFERENCE_AVAILABLE else "Model: Ready",
                                          font=self.fonts['caption'],
                                          fg=self.colors['warning'] if not INFERENCE_AVAILABLE else self.colors['success'],
                                          bg=self.colors['card'],
                                          wraplength=250)
        self.model_status_label.pack(anchor='w', pady=(5, 0))
        
    def create_identify_section(self):
        """Create identify section"""
        identify_section = tk.Frame(self.scrollable_frame, bg=self.colors['card'])
        identify_section.pack(fill='x', padx=20, pady=(25, 20))
        
        self.identify_button = ModernButton(identify_section, text="Identify Cheetah", 
                                          command=self.identify_cheetah,
                                          bg_color=self.colors['accent'],
                                          height=32,
                                          font=self.fonts['button'])
        self.identify_button.pack(fill='x')
        self.identify_button.configure_state('disabled')
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(identify_section, variable=self.progress_var, 
                                          mode='indeterminate',
                                          style='Modern.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x', pady=(15, 0))
        
        # Configure progress bar style
        style = ttk.Style()
        style.configure('Modern.Horizontal.TProgressbar',
                       background=self.colors['primary'],
                       troughcolor=self.colors['border'],
                       borderwidth=0,
                       lightcolor=self.colors['primary'],
                       darkcolor=self.colors['primary'])
        
    def create_main_area(self):
        """Create responsive main content area"""
        self.main_area = tk.Frame(self.content_frame, bg=self.colors['bg'])
        self.main_area.pack(side='right', fill='both', expand=True)
        
        # Results header
        self.results_header = tk.Frame(self.main_area, bg=self.colors['bg'])
        self.results_header.pack(fill='x', pady=(0, 15))
        
        self.results_title = tk.Label(self.results_header, text="Identification Results", 
                                font=self.fonts['heading'],
                                fg=self.colors['text_primary'],
                                bg=self.colors['bg'])
        self.results_title.pack(anchor='w')
        
        # Results info card - responsive height
        self.results_card = RoundedFrame(self.main_area, radius=20, bg_color=self.colors['card'])
        self.results_card.pack(fill='x', pady=(0, 15))
        self.results_card.configure(height=100)  # or whatever height you want
        self.results_card.pack_propagate(False)  # <- This stops it resizing to fit content

        
        # Bind resize to make results card responsive
        self.results_card.bind('<Configure>', self._update_results_card_height)
        
        self.results_content = tk.Frame(self.results_card.canvas, bg=self.colors['card'])
        self.results_card.canvas.create_window(0, 0, window=self.results_content, anchor='nw')
        
        # Default results message
        self.default_results_label = tk.Label(self.results_content, 
                                             text="Upload an image and run identification to see results", 
                                             font=self.fonts['body'],
                                             fg=self.colors['text_secondary'],
                                             bg=self.colors['card'])
        self.default_results_label.pack(expand=True, pady=30)
        
        # Image comparison area - responsive
        self.create_image_comparison_area()
        
    def _update_results_card_height(self, event):
        """Update results card height responsively"""
        if event.widget == self.results_card:
            window_height = self.root.winfo_height()
            # Results card should be 15-20% of window height
            card_height = max(100, min(200, window_height * 0.15))
            self.results_card.configure(height=int(card_height))
        
    def create_image_comparison_area(self):
        """Create responsive image comparison area"""
        self.images_frame = tk.Frame(self.main_area, bg=self.colors['bg'])
        self.images_frame.pack(fill='both', expand=True)
        
        # Configure grid weights for responsive layout
        self.images_frame.grid_columnconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(1, weight=1)
        self.images_frame.grid_rowconfigure(0, weight=1)
        
        # Uploaded image container
        self.uploaded_container = RoundedFrame(self.images_frame, radius=20, bg_color=self.colors['card'])
        self.uploaded_container.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        
        # Uploaded image header
        uploaded_header = tk.Frame(self.uploaded_container.canvas, bg=self.colors['card'])
        self.uploaded_container.canvas.create_window(0, 0, window=uploaded_header, anchor='nw', 
                                                    tags="header")
        
        self.uploaded_title = tk.Label(uploaded_header, text="📤 Uploaded Image", 
                                 font=self.fonts['subheading'],
                                 fg=self.colors['text_primary'],
                                 bg=self.colors['card'])
        self.uploaded_title.pack(pady=15)
        
        # Uploaded image frame
        self.uploaded_image_frame = ResponsiveImageFrame(self.uploaded_container.canvas, radius=15)
        self.uploaded_container.canvas.create_window(0, 0, window=self.uploaded_image_frame, 
                                                    anchor='nw', tags="image")
        
        # Reference image container
        self.reference_container = RoundedFrame(self.images_frame, radius=20, bg_color=self.colors['card'])
        self.reference_container.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        
        # Reference image header
        reference_header = tk.Frame(self.reference_container.canvas, bg=self.colors['card'])
        self.reference_container.canvas.create_window(0, 0, window=reference_header, anchor='nw', 
                                                     tags="header")
        
        self.reference_title = tk.Label(reference_header, text="📋 Reference Image", 
                                  font=self.fonts['subheading'],
                                  fg=self.colors['text_primary'],
                                  bg=self.colors['card'])
        self.reference_title.pack(pady=15)
        
        # Reference image frame
        self.reference_image_frame = ResponsiveImageFrame(self.reference_container.canvas, radius=15)
        self.reference_container.canvas.create_window(0, 0, window=self.reference_image_frame, 
                                                     anchor='nw', tags="image")
        
        # Bind resize events for both image containers
        self.uploaded_container.canvas.bind('<Configure>', self._on_uploaded_container_resize)
        self.reference_container.canvas.bind('<Configure>', self._on_reference_container_resize)
        
        # Handle window resize to update layout
        self.images_frame.bind('<Configure>', self._on_images_frame_resize)
    
    def _on_images_frame_resize(self, event):
        """Handle images frame resize for responsive layout"""
        if event.widget == self.images_frame:
            width = event.width
            height = event.height
            
            # On very small windows, stack vertically
            if width < 800:
                self.uploaded_container.grid_configure(row=0, column=0, columnspan=2, sticky='ew', 
                                                      padx=0, pady=(0, 8))
                self.reference_container.grid_configure(row=1, column=0, columnspan=2, sticky='ew', 
                                                       padx=0, pady=(8, 0))
                self.images_frame.grid_rowconfigure(1, weight=1)
            else:
                # Side by side layout
                self.uploaded_container.grid_configure(row=0, column=0, columnspan=1, sticky='nsew', 
                                                      padx=(0, 8), pady=0)
                self.reference_container.grid_configure(row=0, column=1, columnspan=1, sticky='nsew', 
                                                       padx=(8, 0), pady=0)
                self.images_frame.grid_rowconfigure(1, weight=0)
    
    def _on_uploaded_container_resize(self, event):
        """Handle uploaded image container resize"""
        if event.widget == self.uploaded_container.canvas:
            canvas_width = event.width
            canvas_height = event.height
            
            # Calculate header height responsively
            header_height = max(50, canvas_height * 0.1)
            
            # Update header positioning and size
            self.uploaded_container.canvas.itemconfig("header", width=canvas_width)
            for item in self.uploaded_container.canvas.find_withtag("header"):
                self.uploaded_container.canvas.coords(item, 0, 0)
            
            # Update image frame size and position
            image_y = header_height
            image_height = canvas_height - header_height
            
            self.uploaded_image_frame.configure(width=canvas_width, height=max(100, image_height))
            self.uploaded_container.canvas.itemconfig("image", width=canvas_width, height=max(100, image_height))
            self.uploaded_container.canvas.coords(
                self.uploaded_container.canvas.find_withtag("image")[0], 
                0, image_y
            )
    
    def _on_reference_container_resize(self, event):
        """Handle reference image container resize"""
        if event.widget == self.reference_container.canvas:
            canvas_width = event.width
            canvas_height = event.height
            
            # Calculate header height responsively
            header_height = max(50, canvas_height * 0.1)
            
            # Update header positioning and size
            self.reference_container.canvas.itemconfig("header", width=canvas_width)
            for item in self.reference_container.canvas.find_withtag("header"):
                self.reference_container.canvas.coords(item, 0, 0)
            
            # Update image frame size and position
            image_y = header_height
            image_height = canvas_height - header_height
            
            self.reference_image_frame.configure(width=canvas_width, height=max(100, image_height))
            self.reference_container.canvas.itemconfig("image", width=canvas_width, height=max(100, image_height))
            self.reference_container.canvas.coords(
                self.reference_container.canvas.find_withtag("image")[0], 
                0, image_y
            )
    
    def set_threshold(self, level):
        """Set the confidence threshold"""
        self.threshold_var.set(level)
        
        # Update button appearances
        for btn_level, btn in self.threshold_buttons.items():
            if btn_level == level:
                btn.bg_color = self.colors['primary']
                btn.text_color = self.colors['text_light']
            else:
                btn.bg_color = self.colors['border']
                btn.text_color = self.colors['text_secondary']
            btn._draw()
    
    def initialize_inference(self):
        """Initialize the inference pipeline"""
        if INFERENCE_AVAILABLE:
            try:
                # Default paths - user can change these
                model_path = 'models/cheetah_cropped_embedder.pt'
                embeddings_path = 'models/reference_embeddings_cropped.pkl'
                
                print(f"Checking for model files:")
                print(f"  Model path: {model_path} - {'EXISTS' if os.path.exists(model_path) else 'NOT FOUND'}")
                print(f"  Embeddings path: {embeddings_path} - {'EXISTS' if os.path.exists(embeddings_path) else 'NOT FOUND'}")
                
                if os.path.exists(model_path) and os.path.exists(embeddings_path):
                    import torch
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"Initializing model on device: {device}")
                    
                    self.identifier = EnhancedCheetahIdentifier(
                        model_path, embeddings_path, device
                    )
                    self.model_status_label.configure(text=f"Model: Ready ({device})", 
                                                    fg=self.colors['success'])
                    self.status_label.configure(text="Ready for identification", 
                                              fg=self.colors['success'])
                    print("Model initialized successfully!")
                else:
                    self.model_status_label.configure(text="Model: Paths not found", 
                                                    fg=self.colors['warning'])
                    self.status_label.configure(text="Configure model paths", 
                                              fg=self.colors['warning'])
                    print("Model files not found. Please configure paths.")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Error initializing model: {error_msg}")
                self.model_status_label.configure(text=f"Model: Error - {error_msg[:20]}...", 
                                                fg=self.colors['danger'])
                self.status_label.configure(text="Model loading failed", 
                                          fg=self.colors['danger'])
        else:
            print("Inference pipeline not available - running in demo mode")
            self.model_status_label.configure(text="Model: Demo mode", 
                                            fg=self.colors['warning'])
            self.status_label.configure(text="Demo mode - no real inference", 
                                      fg=self.colors['warning'])
        
    def configure_paths(self):
        """Configure model and reference image paths"""
        dialog = ModelConfigDialog(self.root, self.colors)
        if dialog.result:
            model_path, embeddings_path, ref_images_dir = dialog.result
            self.reference_images_dir = ref_images_dir
            
            if INFERENCE_AVAILABLE:
                try:
                    import torch
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.identifier = EnhancedCheetahIdentifier(
                        model_path, embeddings_path, device
                    )
                    self.model_status_label.configure(text=f"Model: Ready ({device})", 
                                                    fg=self.colors['success'])
                    self.status_label.configure(text="Ready for identification", 
                                              fg=self.colors['success'])
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load model: {e}")
                    self.model_status_label.configure(text="Model: Load failed", 
                                                    fg=self.colors['danger'])
            
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select Cheetah Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            filename = os.path.basename(file_path)
            
            # Truncate filename if too long for display
            max_length = max(20, self.sidebar_container.winfo_width() // 10)
            if len(filename) > max_length:
                filename = filename[:max_length-3] + "..."
                
            self.image_info_label.configure(text=f"Selected: {filename}")
            
            # Load and display uploaded image
            self.uploaded_image_frame.set_image(file_path)
            
            # Enable identify button
            if self.identifier or not INFERENCE_AVAILABLE:
                self.identify_button.configure_state('normal')
                self.status_label.configure(text="Ready to identify", fg=self.colors['success'])
            
            # Clear previous results
            self.clear_results()
            
            # Update fonts after image load
            self._update_font_labels()
    
    def _update_font_labels(self):
        """Update all label fonts after window resize"""
        try:
            self.title_label.configure(font=self.fonts['title'])
            self.subtitle_label.configure(font=self.fonts['body'])
            self.upload_title.configure(font=self.fonts['heading'])
            self.image_info_label.configure(font=self.fonts['caption'])
            self.settings_title.configure(font=self.fonts['heading'])
            self.threshold_label.configure(font=self.fonts['subheading'])
            self.status_title.configure(font=self.fonts['heading'])
            self.status_label.configure(font=self.fonts['body'])
            self.model_status_label.configure(font=self.fonts['caption'])
            self.results_title.configure(font=self.fonts['heading'])
            self.uploaded_title.configure(font=self.fonts['subheading'])
            self.reference_title.configure(font=self.fonts['subheading'])
            
            # Update button fonts
            for btn in self.threshold_buttons.values():
                btn.font = self.fonts['caption']
                btn._draw()
            
            self.upload_button.font = self.fonts['button']
            self.upload_button._draw()
            self.config_button.font = self.fonts['button']
            self.config_button._draw()
            self.identify_button.font = self.fonts['button']
            self.identify_button._draw()
            
        except AttributeError:
            # Some widgets might not be created yet
            pass
    
    def clear_results(self):
        """Clear previous identification results"""
        # Clear results content
        for widget in self.results_content.winfo_children():
            widget.destroy()
            
        # Add default message
        self.default_results_label = tk.Label(self.results_content, 
                                             text="Upload an image and run identification to see results", 
                                             font=self.fonts['body'],
                                             fg=self.colors['text_secondary'],
                                             bg=self.colors['card'])
        self.default_results_label.pack(expand=True, pady=30)
        
        # Clear reference image
        self.reference_image_frame.clear_image()
        self.identification_result = None
    
    def display_reference_image(self, individual_name):
        """Display reference image for identified individual"""
        if not self.reference_images_dir:
            self.reference_image_frame.set_image(None, "Reference directory not configured")
            return
        
        # Look for reference image in individual_00x folder structure
        reference_path = None
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Try different folder naming patterns
        possible_folder_names = [
            individual_name,  # exact match
            f"individual_{individual_name.split('_')[-1].zfill(3)}",  # individual_00x format
            f"individual_{individual_name.split('_')[-1]}",  # individual_x format
        ]
        
        # If individual_name already contains numbers, extract and format them
        import re
        numbers = re.findall(r'\d+', individual_name)
        if numbers:
            last_number = numbers[-1]
            possible_folder_names.extend([
                f"individual_{last_number.zfill(3)}",  # pad with zeros
                f"individual_{last_number}",  # as-is
            ])
        
        for folder_name in possible_folder_names:
            folder_path = os.path.join(self.reference_images_dir, folder_name)
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # Look for any image file in this directory
                try:
                    image_files = []
                    for file in os.listdir(folder_path):
                        if any(file.lower().endswith(ext) for ext in possible_extensions):
                            image_files.append(os.path.join(folder_path, file))
                    
                    if image_files:
                        # Use the first image found (you could add logic to pick the best one)
                        reference_path = image_files[0]
                        break
                        
                except Exception as e:
                    continue
        
        if reference_path:
            self.reference_image_frame.set_image(reference_path)
            # Show which image was loaded
            filename = os.path.basename(reference_path)
            folder = os.path.basename(os.path.dirname(reference_path))
            print(f"Loaded reference image: {folder}/{filename}")
        else:
            self.reference_image_frame.set_image(None, f"No reference found for {individual_name}")
            print(f"Searched for reference images in: {self.reference_images_dir}")
            print(f"Tried folder names: {possible_folder_names}")
    
    def identify_cheetah(self):
        """Run cheetah identification in a separate thread"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        # Start progress animation
        self.progress_bar.start()
        self.identify_button.configure_state('disabled')
        self.status_label.configure(text="Identifying...", fg=self.colors['warning'])
        
        # Run identification in separate thread
        thread = threading.Thread(target=self._run_identification)
        thread.daemon = True
        thread.start()
    
    def _run_identification(self):
        """Run identification in background thread"""
        try:
            if INFERENCE_AVAILABLE and self.identifier:
                # Real identification
                result = self.identifier.identify(
                    self.current_image_path,
                    rejection_level=self.threshold_var.get(),
                    return_top_k=3
                )
            else:
                # Demo mode
                import time
                time.sleep(2)  # Simulate processing time
                result = self._create_demo_result()
            
            # Update GUI in main thread
            self.root.after(0, self._display_results, result)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, self._handle_identification_error, error_msg)
    
    def _create_demo_result(self):
        """Create demo result for testing GUI"""
        import random
        demo_individuals = ['individual_001', 'individual_042', 'individual_087', 'Unknown']
        
        if random.random() < 0.8:  # 80% identification rate
            individual = random.choice(demo_individuals[:-1])
            confidence = random.uniform(0.6, 0.95)
            similarity = random.uniform(0.65, 0.92)
            
            return {
                'status': 'identified',
                'individual': individual,
                'confidence': confidence,
                'similarity': similarity,
                'confidence_level': 'High' if confidence > 0.8 else 'Medium',
                'threshold_used': self.threshold_var.get(),
                'all_matches': [
                    {'individual': individual, 'similarity': similarity, 'confidence': confidence},
                    {'individual': 'individual_099', 'similarity': similarity-0.1, 'confidence': confidence-0.15},
                    {'individual': 'individual_156', 'similarity': similarity-0.2, 'confidence': confidence-0.25}
                ]
            }
        else:  # Unknown individual
            return {
                'status': 'unknown',
                'individual': None,
                'best_match_similarity': random.uniform(0.3, 0.5),
                'threshold_value': 0.6,
                'threshold_used': self.threshold_var.get(),
                'all_matches': []
            }
    
    def _display_results(self, result):
        """Display identification results in GUI with responsive layout"""
        try:
            self.identification_result = result
            
            # Stop progress bar
            self.progress_bar.stop()
            self.identify_button.configure_state('normal')
            
            # Clear previous results
            for widget in self.results_content.winfo_children():
                widget.destroy()
            
            # Calculate responsive padding
            results_width = self.results_card.winfo_width()
            padding = max(15, min(40, results_width * 0.05))
            
            if result['status'] == 'identified':
                # Successful identification
                individual = result['individual']
                confidence = result['confidence']
                similarity = result.get('similarity', 0)
                
                self.status_label.configure(text=f"Identified: {individual}", fg=self.colors['success'])
                
                # Create results display
                result_main = tk.Frame(self.results_content, bg=self.colors['card'])
                result_main.pack(fill='both', expand=True, padx=padding, pady=15)
                
                # Individual name with icon
                name_frame = tk.Frame(result_main, bg=self.colors['card'])
                name_frame.pack(fill='x', pady=(0, 10))
                
                success_icon = tk.Label(name_frame, text="✅", font=('SF Pro Display', 16),
                                       bg=self.colors['card'])
                success_icon.pack(side='left')
                
                name_label = tk.Label(name_frame, text=f"Individual: {individual}", 
                                     font=self.fonts['subheading'],
                                     fg=self.colors['text_primary'],
                                     bg=self.colors['card'])
                name_label.pack(side='left', padx=(10, 0))
                
                # Metrics - responsive layout
                metrics_frame = tk.Frame(result_main, bg=self.colors['card'])
                metrics_frame.pack(fill='x')
                
                # Configure grid for responsive metrics
                metrics_frame.grid_columnconfigure(0, weight=1)
                metrics_frame.grid_columnconfigure(1, weight=1)
                metrics_frame.grid_columnconfigure(2, weight=1)
                
                # Confidence
                conf_frame = tk.Frame(metrics_frame, bg=self.colors['card'])
                conf_frame.grid(row=0, column=0, sticky='w', padx=(0, 10))
                
                tk.Label(conf_frame, text="Confidence", 
                        font=self.fonts['caption'],
                        fg=self.colors['text_secondary'],
                        bg=self.colors['card']).pack(anchor='w')
                
                tk.Label(conf_frame, text=f"{confidence:.1%}", 
                        font=self.fonts['subheading'],
                        fg=self.colors['success'],
                        bg=self.colors['card']).pack(anchor='w')
                
                # Similarity
                sim_frame = tk.Frame(metrics_frame, bg=self.colors['card'])
                sim_frame.grid(row=0, column=1, sticky='w', padx=(0, 10))
                
                tk.Label(sim_frame, text="Similarity", 
                        font=self.fonts['caption'],
                        fg=self.colors['text_secondary'],
                        bg=self.colors['card']).pack(anchor='w')
                
                tk.Label(sim_frame, text=f"{similarity:.1%}", 
                        font=self.fonts['subheading'],
                        fg=self.colors['primary'],
                        bg=self.colors['card']).pack(anchor='w')
                
                # Threshold
                thresh_frame = tk.Frame(metrics_frame, bg=self.colors['card'])
                thresh_frame.grid(row=0, column=2, sticky='w')
                
                tk.Label(thresh_frame, text="Threshold", 
                        font=self.fonts['caption'],
                        fg=self.colors['text_secondary'],
                        bg=self.colors['card']).pack(anchor='w')
                
                tk.Label(thresh_frame, text=result.get('threshold_used', 'N/A').capitalize(), 
                        font=self.fonts['subheading'],
                        fg=self.colors['text_primary'],
                        bg=self.colors['card']).pack(anchor='w')
                
                # Display reference image
                self.display_reference_image(individual)
                
            elif result['status'] == 'unknown':
                # Unknown individual
                best_sim = result.get('best_match_similarity', 0)
                threshold = result.get('threshold_value', 0)
                
                self.status_label.configure(text="Unknown Individual", fg=self.colors['warning'])
                
                result_main = tk.Frame(self.results_content, bg=self.colors['card'])
                result_main.pack(fill='both', expand=True, padx=padding, pady=15)
                
                # Unknown indicator
                unknown_frame = tk.Frame(result_main, bg=self.colors['card'])
                unknown_frame.pack(fill='x', pady=(0, 10))
                
                warning_icon = tk.Label(unknown_frame, text="❓", font=('SF Pro Display', 16),
                                       bg=self.colors['card'])
                warning_icon.pack(side='left')
                
                unknown_label = tk.Label(unknown_frame, text="Unknown Individual", 
                                        font=self.fonts['subheading'],
                                        fg=self.colors['warning'],
                                        bg=self.colors['card'])
                unknown_label.pack(side='left', padx=(10, 0))
                
                # Details
                details_frame = tk.Frame(result_main, bg=self.colors['card'])
                details_frame.pack(fill='x')
                
                tk.Label(details_frame, text=f"Best match similarity: {best_sim:.1%}", 
                        font=self.fonts['body'],
                        fg=self.colors['text_secondary'],
                        bg=self.colors['card']).pack(anchor='w')
                
                tk.Label(details_frame, text=f"Threshold: {threshold:.1%}", 
                        font=self.fonts['body'],
                        fg=self.colors['text_secondary'],
                        bg=self.colors['card']).pack(anchor='w', pady=(5, 0))
                
                tk.Label(details_frame, text="This cheetah is not in the reference database.", 
                        font=self.fonts['body'],
                        fg=self.colors['text_secondary'],
                        bg=self.colors['card'],
                        wraplength=max(200, results_width-100)).pack(anchor='w', pady=(10, 0))
                
                self.reference_image_frame.set_image(None, "No reference - unknown individual")
            
            else:
                # Error
                self.status_label.configure(text="Identification failed", fg=self.colors['danger'])
                
                error_frame = tk.Frame(self.results_content, bg=self.colors['card'])
                error_frame.pack(fill='both', expand=True, padx=padding, pady=15)
                
                error_icon = tk.Label(error_frame, text="❌", font=('SF Pro Display', 16),
                                     bg=self.colors['card'])
                error_icon.pack(side='left')
                
                error_label = tk.Label(error_frame, text="Identification failed", 
                                      font=self.fonts['subheading'],
                                      fg=self.colors['danger'],
                                      bg=self.colors['card'])
                error_label.pack(side='left', padx=(10, 0))
                
        except Exception as e:
            self._handle_identification_error(str(e))
    
    def _handle_identification_error(self, error_msg):
        """Handle identification errors"""
        self.progress_bar.stop()
        self.identify_button.configure_state('normal')
        self.status_label.configure(text="Error occurred", fg=self.colors['danger'])
        
        # Display error
        for widget in self.results_content.winfo_children():
            widget.destroy()
            
        # Calculate responsive padding
        results_width = self.results_card.winfo_width()
        padding = max(15, min(40, results_width * 0.05))
            
        error_frame = tk.Frame(self.results_content, bg=self.colors['card'])
        error_frame.pack(fill='both', expand=True, padx=padding, pady=15)
        
        error_icon = tk.Label(error_frame, text="❌", font=('SF Pro Display', 16),
                             bg=self.colors['card'])
        error_icon.pack(side='left')
        
        error_title = tk.Label(error_frame, text="Error", 
                              font=self.fonts['subheading'],
                              fg=self.colors['danger'],
                              bg=self.colors['card'])
        error_title.pack(side='left', padx=(10, 0))
        
        error_detail = tk.Label(error_frame, text=error_msg, 
                               font=self.fonts['body'],
                               fg=self.colors['text_secondary'],
                               bg=self.colors['card'],
                               wraplength=max(200, results_width-100))
        error_detail.pack(anchor='w', pady=(10, 0))
        
        messagebox.showerror("Identification Error", f"Error during identification:\n{error_msg}")

class ModelConfigDialog:
    def __init__(self, parent, colors):
        self.result = None
        self.colors = colors
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configure Model Paths")
        self.dialog.configure(bg=colors['bg'])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Make dialog responsive
        self.dialog.minsize(500, 400)
        self._set_dialog_size()
        
        self.create_dialog_widgets()
        
        # Center dialog
        self._center_dialog()
        
        # Bind resize event
        self.dialog.bind('<Configure>', self._on_dialog_resize)
    
    def _set_dialog_size(self):
        """Set dialog size based on parent window"""
        parent_width = self.dialog.master.winfo_width()
        parent_height = self.dialog.master.winfo_height()
        
        # Dialog should be 60-80% of parent size
        dialog_width = max(500, min(800, int(parent_width * 0.7)))
        dialog_height = max(400, min(600, int(parent_height * 0.7)))
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}")
    
    def _center_dialog(self):
        """Center dialog on parent window"""
        self.dialog.update_idletasks()
        
        parent_x = self.dialog.master.winfo_x()
        parent_y = self.dialog.master.winfo_y()
        parent_width = self.dialog.master.winfo_width()
        parent_height = self.dialog.master.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _on_dialog_resize(self, event):
        """Handle dialog resize"""
        if event.widget == self.dialog:
            # Update wraplength for labels
            width = event.width
            for widget in self.dialog.winfo_children():
                self._update_wraplength_recursive(widget, width)
    
    def _update_wraplength_recursive(self, widget, width):
        """Recursively update wraplength for all labels"""
        if isinstance(widget, tk.Label):
            widget.configure(wraplength=max(200, width - 100))
        
        for child in widget.winfo_children():
            self._update_wraplength_recursive(child, width)
    
    def create_dialog_widgets(self):
        # Main container with responsive padding
        self.main_container = RoundedFrame(self.dialog, radius=20, bg_color=self.colors['card'])
        self.main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.content_frame = tk.Frame(self.main_container.canvas, bg=self.colors['card'])
        self.main_container.canvas.create_window(0, 0, window=self.content_frame, anchor='nw')
        
        # Bind canvas configure to update scroll region
        self.content_frame.bind('<Configure>', self._update_scroll_region)
        
        # Title with responsive font
        self.title_label = tk.Label(self.content_frame, text="Model Configuration", 
                              font=('SF Pro Display', 20, 'bold'),
                              bg=self.colors['card'], fg=self.colors['text_primary'])
        self.title_label.pack(pady=(20, 25))
        
        # Model path
        self._create_path_section(self.content_frame, "Model Path", 
                                 "models/cheetah_cropped_embedder.pt",
                                 self._browse_model)
        
        # Embeddings path
        self._create_path_section(self.content_frame, "Reference Embeddings Path", 
                                 "models/reference_embeddings_cropped.pkl",
                                 self._browse_embeddings)
        
        # Reference images directory
        self._create_path_section(self.content_frame, "Reference Images Directory", 
                                 "data/raw/",
                                 self._browse_ref_dir, is_directory=True)
        
        # Buttons with responsive layout
        button_frame = tk.Frame(self.content_frame, bg=self.colors['card'])
        button_frame.pack(fill='x', pady=(30, 20))
        
        # Configure button frame for responsive layout
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=0)
        button_frame.grid_columnconfigure(2, weight=0)
        
        # Spacer
        spacer = tk.Frame(button_frame, bg=self.colors['card'])
        spacer.grid(row=0, column=0, sticky='ew')
        
        self.apply_btn = ModernButton(button_frame, text="Apply", command=self.apply,
                                bg_color=self.colors['primary'],
                                width=100, height=40)
        self.apply_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.cancel_btn = ModernButton(button_frame, text="Cancel", command=self.cancel,
                                 bg_color=self.colors['border'],
                                 text_color=self.colors['text_secondary'],
                                 width=100, height=40)
        self.cancel_btn.grid(row=0, column=2)
    
    def _update_scroll_region(self, event):
        """Update scroll region when content changes"""
        self.main_container.canvas.configure(scrollregion=self.main_container.canvas.bbox("all"))
    
    def _create_path_section(self, parent, title, default_value, browse_command, is_directory=False):
        section_frame = tk.Frame(parent, bg=self.colors['card'])
        section_frame.pack(fill='x', pady=(0, 20))
        
        # Title
        title_label = tk.Label(section_frame, text=title, 
                              font=('SF Pro Display', 14, 'bold'),
                              bg=self.colors['card'], fg=self.colors['text_primary'])
        title_label.pack(anchor='w', pady=(0, 8))
        
        # Path frame with responsive layout
        path_frame = tk.Frame(section_frame, bg=self.colors['card'])
        path_frame.pack(fill='x')
        
        # Configure grid for responsive entry and button
        path_frame.grid_columnconfigure(0, weight=1)
        path_frame.grid_columnconfigure(1, weight=0)
        
        # Entry
        var_name = f"{title.lower().replace(' ', '_').replace(':', '')}_var"
        setattr(self, var_name, tk.StringVar(value=default_value))
        
        entry = tk.Entry(path_frame, textvariable=getattr(self, var_name), 
                        font=('SF Pro Display', 11),
                        bg=self.colors['bg'], fg=self.colors['text_primary'],
                        relief='flat', bd=5)
        entry.grid(row=0, column=0, sticky='ew', ipady=6, padx=(0, 10))
        
        # Browse button
        browse_btn = ModernButton(path_frame, text="Browse", command=browse_command,
                                 bg_color=self.colors['secondary'],
                                 width=80, height=34)
        browse_btn.grid(row=0, column=1)
    
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pt *.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)
    
    def _browse_embeddings(self):
        path = filedialog.askopenfilename(
            title="Select Embeddings File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if path:
            self.reference_embeddings_path_var.set(path)
    
    def _browse_ref_dir(self):
        path = filedialog.askdirectory(title="Select Reference Images Directory")
        if path:
            self.reference_images_directory_var.set(path)
    
    def apply(self):
        self.result = (
            self.model_path_var.get(),
            self.reference_embeddings_path_var.get(),
            self.reference_images_directory_var.get()
        )
        self.dialog.destroy()
    
    def cancel(self):
        self.dialog.destroy()

def main():
    """Main application entry point"""
    # Create main window
    root = tk.Tk()
    
    # Set window icon if available
    try:
        # You can add an icon file here
        # root.iconbitmap('bigcatid_icon.ico')
        pass
    except:
        pass
    
    # Make window responsive
    root.resizable(True, True)
    
    # Create and run application
    app = BigCatIDGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit BigCatID?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Bind global resize events for better responsiveness
    def on_global_resize(event):
        if event.widget == root:
            # Force update of all responsive elements
            root.update_idletasks()
            app._update_fonts_for_size()
            app._update_font_labels()
    
    root.bind('<Configure>', on_global_resize)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    print("Starting BigCatID GUI...")
    print("=" * 50)
    
    if not INFERENCE_AVAILABLE:
        print("=" * 60)
        print("WARNING: Inference pipeline not available.")
        print("The GUI will run in demo mode.")
        print("To use real inference, ensure the following:")
        print("1. Your inference script is saved as inference_pipeline.py")
        print("2. All required dependencies are installed:")
        print("   pip install torch torchvision ultralytics pillow")
        print("3. Model files are available in models/ directory")
        print("=" * 60)
    else:
        print("Inference pipeline loaded successfully!")
        print("=" * 50)
    
    main()