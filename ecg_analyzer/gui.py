import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import cv2

from .extractor import process_file


class ECGAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        root.title('ECG Analyzer')
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(self.frame)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Button(btn_frame, text='Open Image', command=self.open_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text='Export JSON', command=self.export_json).pack(side=tk.LEFT)

        self.img_label = tk.Label(self.frame)
        self.img_label.pack(side=tk.TOP, padx=4, pady=4)

        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # overlay options
        opts_frame = tk.Frame(self.frame)
        opts_frame.pack(side=tk.TOP, fill=tk.X)
        self.show_grid = tk.BooleanVar(value=False)
        self.show_markers = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_frame, text='Show Markers', variable=self.show_markers, command=self.redraw).pack(side=tk.LEFT)
        tk.Checkbutton(opts_frame, text='Show Grid', variable=self.show_grid, command=self.redraw).pack(side=tk.LEFT)

        self.current_result = None

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.pdf')])
        if not path:
            return
        try:
            res = process_file(path, output=None, out_format='json', show_plot=False)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return
        self.current_result = res
        # show original image thumbnail
        from .extractor import load_image, extract_waveform

        img = load_image(path)
        h, w = img.shape[:2]
        pil = Image.fromarray(img[:, :, ::-1])
        pil.thumbnail((600, 400))
        tkimg = ImageTk.PhotoImage(pil)
        self.img_label.configure(image=tkimg)
        self.img_label.image = tkimg

        # plot waveform and markers (store and redraw)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        signal, seconds_per_pixel = extract_waveform(gray)
        self.current_signal = signal
        self.current_seconds_per_pixel = seconds_per_pixel
        self.current_features = res.get('features', {})
        self.redraw()

    def redraw(self):
        if getattr(self, 'current_signal', None) is None:
            return
        signal = self.current_signal
        seconds_per_pixel = self.current_seconds_per_pixel
        features = self.current_features
        r_peaks = features.get('r_peaks_indices', [])

        self.ax.clear()
        t = (np.arange(len(signal)) * seconds_per_pixel)
        self.ax.plot(t, signal, '-k')
        if self.show_markers.get() and len(r_peaks):
            self.ax.plot(np.array(r_peaks) * seconds_per_pixel, signal[r_peaks], 'ro')
        if self.show_grid.get():
            # simple vertical gridlines using seconds_per_pixel and 0.04s small boxes
            if seconds_per_pixel and seconds_per_pixel > 0:
                box_pixels = int(0.04 / seconds_per_pixel)
                if box_pixels < 1:
                    box_pixels = 25
            else:
                box_pixels = 25
            for x in range(0, len(signal), box_pixels * 5):
                self.ax.axvline(x * seconds_per_pixel, color='#ddd', linewidth=0.5)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Normalized amplitude')
        self.canvas.draw()

    def export_json(self):
        if not self.current_result:
            messagebox.showinfo('Info', 'No result to export')
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not path:
            return
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.current_result, f, indent=2)
        messagebox.showinfo('Saved', f'Saved results to {path}')


def run_gui():
    root = tk.Tk()
    app = ECGAnalyzerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    run_gui()
