import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd

class PerceptronVisualizer:
    def __init__(self, data_input, W_history, *meta_tracks, show_plot=True):
        self.W_history = [np.ravel(w) for w in W_history]
        self.dim = len(self.W_history[0])
        self.is_2d = (self.dim == 3)
        self.tracks = meta_tracks 
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
        self.x_pos, self.y_pos, self.x_neg, self.y_neg = [], [], [], []
        for item in self.raw_data:
            point, label = item
            x_coord, y_coord = point[1], point[2]
            if label > 0:
                self.x_pos.append(x_coord); self.y_pos.append(y_coord)
            else:
                self.x_neg.append(x_coord); self.y_neg.append(y_coord)
        
        if self.x_pos: self.ax_main.scatter(self.x_pos, self.y_pos, c='blue', marker='o', s=50, alpha=0.6, label='Class +1')
        if self.x_neg: self.ax_main.scatter(self.x_neg, self.y_neg, c='red', marker='x', s=50, alpha=0.6, label='Class -1')
        
        all_x = [item[0][1] for item in self.raw_data]; all_y = [item[0][2] for item in self.raw_data]
        self.x_lims = (min(all_x), max(all_x)); self.y_lims = (min(all_y), max(all_y))
        x_pad, y_pad = (self.x_lims[1] - self.x_lims[0]) * 0.1, (self.y_lims[1] - self.y_lims[0]) * 0.1
        self.ax_main.set_xlim(self.x_lims[0] - x_pad, self.x_lims[1] + x_pad)
        self.ax_main.set_ylim(self.y_lims[0] - y_pad, self.y_lims[1] + y_pad)
        
        xlim = self.ax_main.get_xlim()
        self.line, = self.ax_main.plot(xlim, [0, 0], 'g-', lw=2, label='Decision Boundary')
        self.ax_main.set_xlabel('x1'); self.ax_main.set_ylabel('x2')
        self.ax_main.grid(True, alpha=0.3); self.ax_main.legend(loc='best')

    def save_tracks_separately(self, prefix="plot"):
        for i, (data, name) in enumerate(self.tracks):
            temp_fig, temp_ax = plt.subplots(figsize=(8, 4))
            temp_ax.plot(data, color='steelblue', lw=2)
            temp_ax.set_title(f"Evolution of {name}")
            temp_ax.set_ylabel(name); temp_ax.set_xlabel("Iteration")
            temp_ax.grid(True, alpha=0.3)
            clean_name = name.replace(" ", "_").lower()
            filename = f"{prefix}{clean_name}.png"
            temp_fig.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(temp_fig)
            print(f"Saved track: {filename}")

    def save_final_2d_plot(self, filename="final_boundary.png"):
        """Saves ONLY the 2D plot using the final weight in the history."""
        if not self.is_2d: return
        
        # 1. Create a clean standalone figure for the final result
        temp_fig, temp_ax = plt.subplots(figsize=(8, 6))
        
        # 2. Re-plot the data points
        if self.x_pos: temp_ax.scatter(self.x_pos, self.y_pos, c='blue', marker='o', s=50, alpha=0.6, label='Class +1')
        if self.x_neg: temp_ax.scatter(self.x_neg, self.y_neg, c='red', marker='x', s=50, alpha=0.6, label='Class -1')
        
        # 3. Calculate and plot the FINAL boundary (last index of history)
        w_final = self.W_history[-1]
        w0, w1, w2 = w_final[0], w_final[1], w_final[2]
        
        # Set limits same as main plot
        x_pad = (self.x_lims[1] - self.x_lims[0]) * 0.1
        y_pad = (self.y_lims[1] - self.y_lims[0]) * 0.1
        temp_ax.set_xlim(self.x_lims[0] - x_pad, self.x_lims[1] + x_pad)
        temp_ax.set_ylim(self.y_lims[0] - y_pad, self.y_lims[1] + y_pad)
        
        xlim = np.array(temp_ax.get_xlim())
        if abs(w2) > 1e-9:
            y_line = -(w1 * xlim + w0) / w2
            temp_ax.plot(xlim, y_line, 'g-', lw=2, label='Final Decision Boundary')
        
        temp_ax.set_title(f"Final Perceptron Boundary (Iter {len(self.W_history)-1})")
        temp_ax.set_xlabel('x1'); temp_ax.set_ylabel('x2')
        temp_ax.grid(True, alpha=0.3); temp_ax.legend(loc='best')
        
        # 4. Save and close
        temp_fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(temp_fig)
        print(f"Saved isolated final 2D plot: {filename}")

    def save_training_video(self, filename="training_evolution.mp4", fps=10):
        if not self.is_2d: return
        print("Generating video...")
        def animate(i):
            self.ind = i
            self.update_plot()
            return [self.line] + self.track_dots
        anim = FuncAnimation(self.fig, animate, frames=len(self.W_history), interval=1000/fps)
        try:
            anim.save(filename, writer=FFMpegWriter(fps=fps))
        except:
            anim.save(filename.replace(".mp4", ".gif"), writer='pillow', fps=fps)
        print(f"Video saved: {filename}")

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