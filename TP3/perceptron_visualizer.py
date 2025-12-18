import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

class PerceptronVisualizer:
    def __init__(self, L_ens, W_history, *meta_tracks):
        """
        meta_tracks: Variable number of tuples (list_of_values, "Label Name")
        """
        self.W_history = [np.ravel(w) for w in W_history]
        self.tracks = meta_tracks 
        self.ind = 0
        
        # Prepare Point Data
        self.x1_vals = np.array([x[1] for x, t in L_ens])
        self.x2_vals = np.array([x[2] for x, t in L_ens])
        labels = [t for x, t in L_ens]
        colors = ['red' if t == -1 else 'blue' for t in labels]
        
        # Setup Figure Layout
        n_tracks = len(self.tracks)
        grid_size = 1 + n_tracks
        height_ratios = [3] + [1] * n_tracks
        
        # Increased bottom margin to house UI elements comfortably
        self.fig, self.axes = plt.subplots(
            grid_size, 1, 
            figsize=(10, 6 + (1.5 * n_tracks)), 
            gridspec_kw={'height_ratios': height_ratios}
        )
        plt.subplots_adjust(bottom=0.2, hspace=0.4) # Reserve space for buttons/slider

        if n_tracks == 0: 
            self.axes = [self.axes]
            
        self.ax_main = self.axes[0]
        self.ax_tracks = self.axes[1:] if n_tracks > 0 else []

        # 1. Initialize Main Plot
        self.ax_main.scatter(self.x1_vals, self.x2_vals, c=colors, s=60, edgecolors='k', alpha=0.7)
        self.line, = self.ax_main.plot([], [], 'g-', lw=3, label="Boundary")
        
        margin = 1.0
        self.ax_main.set_xlim(self.x1_vals.min() - margin, self.x1_vals.max() + margin)
        self.ax_main.set_ylim(self.x2_vals.min() - margin, self.x2_vals.max() + margin)
        self.ax_main.grid(True, linestyle='--', alpha=0.3)

        # 2. Initialize Meta-Data Plots
        self.track_dots = []
        for i, (data, name) in enumerate(self.tracks):
            ax = self.ax_tracks[i]
            ax.plot(data, color='steelblue', alpha=0.4, lw=1.5) 
            dot, = ax.plot([0], [data[0]], 'ro', markersize=6) 
            ax.set_ylabel(name, fontsize=10)
            ax.grid(True, alpha=0.2)
            self.track_dots.append(dot)

        # --- UI ELEMENTS ---
        # Slider: Positioned at the bottom center
        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Step ', 0, len(self.W_history)-1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update_from_slider)

        # Buttons: Grouped at the bottom right to avoid blocking the left side
        ax_prev = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.81, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)
        
        self.update_plot()
        plt.show()

    def update_plot(self):
        # Update Decision Boundary
        w_k = self.W_history[self.ind]
        w0, w1, w2 = w_k[0], w_k[1], w_k[2]
        
        x1_lims = np.array(self.ax_main.get_xlim())
        
        # Stability check for division
        if abs(w2) > 1e-9:
            x2_line = -(w1 * x1_lims + w0) / w2
            self.line.set_data(x1_lims, x2_line)
        elif abs(w1) > 1e-9:
            # Vertical line case
            x_val = -w0 / w1
            self.line.set_data([x_val, x_val], self.ax_main.get_ylim())

        # Update Red Dots on the meta-graphs
        for i, (data, name) in enumerate(self.tracks):
            if self.ind < len(data):
                self.track_dots[i].set_data([self.ind], [data[self.ind]])

        self.ax_main.set_title(f"Iteration: {self.ind} | w0: {w0:.2f}, w1: {w1:.2f}, w2: {w2:.2f}", pad=10)
        self.fig.canvas.draw_idle()

    def update_from_slider(self, val):
        self.ind = int(val)
        self.update_plot()

    def next(self, event):
        if self.ind < len(self.W_history) - 1:
            self.slider.set_val(self.ind + 1)

    def prev(self, event):
        if self.ind > 0:
            self.slider.set_val(self.ind - 1)