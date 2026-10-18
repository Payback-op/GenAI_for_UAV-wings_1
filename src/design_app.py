"""Interactive 2D airfoil visualizer, following the BTP design form workflow."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from common import ROOT, save_json
from generate_airfoil import generate
from generate_combined import draw_section, validate_airfoil
from predict import load_bundle


class DesignApp:
    def __init__(self, root):
        self.root = root
        self.config = json.loads((ROOT / "config/design_conditions.json").read_text())
        self.bundle = load_bundle()
        self.events = queue.Queue()
        self.fields = {}
        self.entries = []
        self.busy = False
        self.output = None
        self.generated_conditions = None
        self.combined_window = None
        root.title("BTP1 | Airfoil visualizer")
        root.geometry("1220x800")
        root.minsize(1000, 720)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f8fafc")
        style.configure("TLabel", background="#f8fafc", foreground="#0f172a", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=9)
        shell = ttk.Frame(root, padding=22)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text="Generate an airfoil", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Open 2D + 3D visualizer", command=self.open_combined).pack(side="right")
        ttk.Label(shell, text="Set your targets, generate a shape, and inspect its 2D cross-section.").pack(anchor="w", pady=(5, 6))
        ttk.Label(shell, text="NASA model estimates: Cl, Cd and Cm. Predicted matches remain physically unverified.",
                  foreground="#9a3412").pack(anchor="w", pady=(0, 12))
        body = ttk.Frame(shell)
        body.pack(fill="both", expand=True)
        form = ttk.Frame(body, padding=(0, 0, 24, 0))
        form.pack(side="left", fill="y")
        definitions = [
            ("angle_of_attack_deg", "Angle of attack (degrees)", 1),
            ("reynolds_number", "Reynolds number", 1),
            ("ncrit", "Ncrit", 1),
            ("target_cl", "Desired lift coefficient (Cl)", 1),
            ("cl_tolerance", "Allowed lift difference (+/-)", 1),
            ("maximum_cd", "Maximum drag coefficient (Cd)", 1),
            ("minimum_thickness_ratio", "Minimum thickness (% chord)", 100),
            ("maximum_thickness_ratio", "Maximum thickness (% chord)", 100),
        ]
        for key, label, scale in definitions:
            ttk.Label(form, text=label).pack(anchor="w", pady=(6, 2))
            var = tk.StringVar(value=f"{self.config[key] * scale:g}")
            entry = ttk.Entry(form, textvariable=var, width=29)
            entry.pack(fill="x")
            self.fields[key] = (var, scale)
            self.entries.append(entry)
        bounds = self.bundle["metadata"]["condition_bounds"]
        note = "Training ranges\n" + "\n".join(f"{key}: {lo:g} to {hi:g}" for key, (lo, hi) in bounds.items())
        ttk.Label(form, text=note, foreground="#475569").pack(anchor="w", pady=10)
        self.button = ttk.Button(form, text="Generate shape", command=self.start)
        self.button.pack(fill="x", pady=(0, 5))
        self.open_button = ttk.Button(form, text="Open saved results", command=self.open_results, state="disabled")
        self.open_button.pack(fill="x")
        preview = ttk.Frame(body)
        preview.pack(side="left", fill="both", expand=True)
        self.status = tk.StringVar(value="Set your targets and click Generate shape.")
        self.status_label = ttk.Label(preview, textvariable=self.status, wraplength=800, font=("Segoe UI", 11, "bold"))
        self.status_label.pack(anchor="w", pady=(4, 8))
        self.metrics = tk.StringVar(value="Predictions will appear here.")
        ttk.Label(preview, textvariable=self.metrics, font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=8)
        self.figure = Figure(figsize=(9, 4), facecolor="#f8fafc")
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=.10, right=.97, bottom=.16, top=.90)
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, master=preview)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, preview, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.detail = tk.StringVar(value="Outputs: airfoil image, DAT/CSV coordinates and a traceable design.json.\nUse the toolbar to pan, zoom, reset the view or save an image.")
        self.detail_label = ttk.Label(preview, textvariable=self.detail, wraplength=800, foreground="#475569")
        self.detail_label.pack(anchor="w", pady=(10, 0))
        preview.bind("<Configure>", self.resize_labels)
        latest = ROOT / "results/2d/latest.json"
        if latest.exists():
            try:
                self.show(json.loads(latest.read_text()))
            except (OSError, KeyError, ValueError):
                self.status.set("Previous output unavailable. Click Generate shape.")
        for var, _ in self.fields.values():
            var.trace_add("write", self.inputs_changed)
        self.poll_id = root.after(150, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def resize_labels(self, event):
        for label in (self.detail_label, self.status_label):
            label.configure(wraplength=max(200, event.width - 16))

    def conditions(self):
        conditions = deepcopy(self.config)
        for key, (var, scale) in self.fields.items():
            conditions[key] = float(var.get()) / scale
        validate_airfoil(conditions, self.bundle)
        return conditions

    def inputs_changed(self, *_):
        if self.generated_conditions is not None and not self.busy:
            try:
                current = self.conditions()
            except ValueError:
                current = None
            self.status.set(self.result_status if current == self.generated_conditions else
                            "Inputs changed. Click Generate shape to update the preview.")

    def start(self):
        if self.busy:
            return
        try:
            conditions = self.conditions()
            save_json(ROOT / "config/design_conditions.json", conditions)
        except (ValueError, KeyError, OSError) as error:
            messagebox.showerror("Check inputs", str(error), parent=self.root)
            return
        self.config = conditions
        self.busy = True
        self.button.configure(state="disabled")
        for entry in self.entries:
            entry.configure(state="disabled")
        self.status.set("Generating and checking airfoil candidates...")

        def worker():
            try:
                result = generate(conditions, progress=lambda text: self.events.put(("progress", text)))
                self.events.put(("result", result))
            except Exception as error:
                self.events.put(("error", str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def show(self, summary):
        folder = Path(summary["output_folder"])
        shape = np.load(folder / "airfoil_shape.npy")
        self.ax.set_axis_on()
        draw_section(self.ax, shape)
        self.toolbar.update()
        self.canvas.draw_idle()
        self.output = folder
        self.generated_conditions = deepcopy(summary["conditions"])
        self.result_status = ("Candidate meets the predicted targets (unverified)." if summary["status"] == "predicted_match"
                              else "No predicted match found. Closest valid candidate shown.")
        self.status.set(self.result_status)
        p = summary["predictions"]
        self.metrics.set(f"Cl  {p['Cl']:.4f}     Cd  {p['Cd']:.5f}     Cm  {p['Cm']:.4f}")
        self.detail.set(f"Maximum thickness: {100 * summary['maximum_thickness_ratio']:.2f}% chord\n"
                        f"Shape: {summary['geometry_provenance']['method']}\n"
                        f"Coordinates normalized by chord. Saved: {folder}\n"
                        "Use the toolbar to pan, zoom, reset or save the view.")
        self.open_button.configure(state="normal")
        self.inputs_changed()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                    continue
                self.busy = False
                self.button.configure(state="normal")
                for entry in self.entries:
                    entry.configure(state="normal")
                if kind == "result":
                    try:
                        self.show(value)
                    except (OSError, KeyError, ValueError) as error:
                        self.status.set("Could not display the generated output.")
                        messagebox.showerror("Preview failed", str(error), parent=self.root)
                else:
                    self.status.set("Generation stopped. Previous output remains displayed.")
                    messagebox.showerror("Generation failed", value, parent=self.root)
        except queue.Empty:
            pass
        self.poll_id = self.root.after(150, self.poll)

    def open_results(self):
        if self.output:
            os.startfile(self.output)

    def open_combined(self):
        if self.combined_window is not None and self.combined_window.winfo_exists():
            self.combined_window.deiconify()
            self.combined_window.lift()
            return
        try:
            conditions = self.conditions()
        except (ValueError, KeyError) as error:
            messagebox.showerror("Check inputs", str(error), parent=self.root)
            return
        from combined_app import CombinedApp
        window = tk.Toplevel(self.root)
        try:
            self.combined_app = CombinedApp(window, airfoil_conditions=conditions)
        except Exception as error:
            window.destroy()
            messagebox.showerror("Could not open combined visualizer", str(error), parent=self.root)
            return
        self.combined_window = window

    def close(self):
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    if args.smoke_test:
        root.withdraw()
    app = DesignApp(root)
    if args.smoke_test:
        root.update_idletasks()
        app.canvas.draw()
        assert len(app.fields) == 8 and app.output is not None
        assert len(app.ax.lines) == 2
        var, _ = app.fields["target_cl"]
        previous = var.get()
        var.set("0.7")
        assert "Inputs changed" in app.status.get()
        var.set(previous)
        assert app.status.get() == app.result_status
        app.close()
        print("2D visualizer smoke test passed: 8 controls, airfoil preview, predictions and changed-input status")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
