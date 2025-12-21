import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import pandas as pd

class PerceptronVisualizer:
    def __init__(self, data_input, W_history, *meta_tracks, show_plot=True):
        self.W_history = [np.ravel(w) for w in W_history]
        self.dim = len(self.W_history[0])
        self.is_2d = (self.dim == 3)
        self.tracks = meta_tracks # List of tuples: (data_list, "Name")
        self.ind = 0
        
        if isinstance(data_input, pd.DataFrame):
            self.raw_data = data_input.values.tolist()
        else:
            self.raw_data = data_input

        n_tracks = len(self.tracks)
        grid_size = (1 + n_tracks) if self.is_2d else n_tracks
        height_ratios = ([3] + [1] * n_tracks) if self.is_2d else ([1] * n_tracks)
        
        fig_height = 6 + (1.5 * n_tracks) if self.is_2d else (2 * n_tracks)
        self.fig, self.axes = plt.subplots(
            grid_size, 1, 
            figsize=(10, fig_height), 
            gridspec_kw={'height_ratios': height_ratios}
        )
        plt.subplots_adjust(bottom=0.2, hspace=0.4)

        if not isinstance(self.axes, (list, np.ndarray)): 
            self.axes = [self.axes]
        
        if self.is_2d:
            self.ax_main = self.axes[0]
            self.ax_tracks = self.axes[1:]
            self._setup_2d_scatter()
        else:
            self.ax_main = None
            self.ax_tracks = self.axes

        self.track_dots = []
        for i, (data, name) in enumerate(self.tracks):
            ax = self.ax_tracks[i]
            ax.plot(data, color='steelblue', alpha=0.4, lw=1.5) 
            dot, = ax.plot([0], [data[0]], 'ro', markersize=6) 
            ax.set_ylabel(name, fontsize=10)
            ax.grid(True, alpha=0.2)
            self.track_dots.append(dot)

        # UI Elements
        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Step ', 0, len(self.W_history)-1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update_from_slider)

        ax_prev = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.81, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev)
        self.btn_next.on_clicked(self.next)
        
        self.update_plot()
        if show_plot: plt.show()

    def _setup_2d_scatter(self):
        """Setup the 2D scatter plot with data points and decision boundary line."""
        # Extract x and y coordinates from the data
        # L_ens format: [[point, label], ...] where point = [bias, x1, x2]
        x_pos = []
        y_pos = []
        x_neg = []
        y_neg = []
        
        for item in self.raw_data:
            point, label = item
            # Skip bias (first element), extract x1 and x2
            x_coord = point[1]
            y_coord = point[2]
            
            if label > 0:
                x_pos.append(x_coord)
                y_pos.append(y_coord)
            else:
                x_neg.append(x_coord)
                y_neg.append(y_coord)
        
        # Plot positive and negative points
        if x_pos:
            self.ax_main.scatter(x_pos, y_pos, c='blue', marker='o', s=50, alpha=0.6, label='Class +1')
        if x_neg:
            self.ax_main.scatter(x_neg, y_neg, c='red', marker='x', s=50, alpha=0.6, label='Class -1')
        
        # Set axis limits with some padding
        all_x = [item[0][1] for item in self.raw_data]
        all_y = [item[0][2] for item in self.raw_data]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        self.ax_main.set_xlim(x_min - x_pad, x_max + x_pad)
        self.ax_main.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Create initial decision boundary line (will be updated in update_plot)
        xlim = self.ax_main.get_xlim()
        self.line, = self.ax_main.plot(xlim, [0, 0], 'g-', lw=2, label='Decision Boundary')
        
        self.ax_main.set_xlabel('x1', fontsize=12)
        self.ax_main.set_ylabel('x2', fontsize=12)
        self.ax_main.set_title('Perceptron Decision Boundary', fontsize=14)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(loc='best')

    def save_tracks_separately(self, prefix="plot"):
        """Saves each tracking variable into its own individual PNG file."""
        for i, (data, name) in enumerate(self.tracks):
            # Create a temporary figure for just this track
            temp_fig, temp_ax = plt.subplots(figsize=(8, 4))
            temp_ax.plot(data, color='steelblue', lw=2)
            temp_ax.set_title(f"Evolution of {name}")
            temp_ax.set_ylabel(name)
            temp_ax.set_xlabel("Iteration")
            temp_ax.grid(True, alpha=0.3)
            
            # Create filename: e.g., "plot_Temperature.png"
            clean_name = name.replace(" ", "_").lower()
            filename = f"{prefix}_{clean_name}.png"
            
            temp_fig.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(temp_fig) # Close to free up memory
            print(f"Saved: {filename}")

    def update_plot(self):
        w_k = self.W_history[self.ind]
        if self.is_2d and self.ax_main:
            w0, w1, w2 = w_k[0], w_k[1], w_k[2]
            xlim = np.array(self.ax_main.get_xlim())
            if abs(w2) > 1e-9:
                y_line = -(w1 * xlim + w0) / w2
                self.line.set_data(xlim, y_line)
        
        for i, (data, name) in enumerate(self.tracks):
            if self.ind < len(data):
                self.track_dots[i].set_data([self.ind], [data[self.ind]])
        self.fig.canvas.draw_idle()

    def update_from_slider(self, val):
        self.ind = int(val); self.update_plot()

    def next(self, event):
        if self.ind < len(self.W_history) - 1: self.slider.set_val(self.ind + 1)

    def prev(self, event):
        if self.ind > 0: self.slider.set_val(self.ind - 1)