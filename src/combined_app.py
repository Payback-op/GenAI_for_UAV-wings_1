"""Local desktop form for NASA-based 2D section and synchronized 3D wing design."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox,ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from common import ROOT,save_json
from generate_combined import generate_combined,validate_combined,draw_section,draw_planform,draw_wing
from predict import load_bundle

class CombinedApp:
    def __init__(self,root,airfoil_conditions=None):
        self.root=root
        self.config=json.loads((ROOT/"config/combined_conditions.json").read_text())
        if airfoil_conditions is not None:
            self.config["airfoil"].update(deepcopy(airfoil_conditions))
        self.bundle=load_bundle()
        self.events=queue.Queue()
        self.busy=False
        self.output=None
        self.generated_conditions=None
        self.fields={}; self.entries=[]
        root.title("BTP1 | Combined 2D + 3D visualizer")
        root.geometry("1500x900"); root.minsize(1150,750)
        style=ttk.Style(root); style.theme_use("clam")
        style.configure("TFrame",background="#f8fafc")
        style.configure("TLabel",background="#f8fafc",font=("Segoe UI",10))
        style.configure("Title.TLabel",font=("Segoe UI",22,"bold"))
        style.configure("TButton",font=("Segoe UI",10),padding=8)
        shell=ttk.Frame(root,padding=16); shell.pack(fill="both",expand=True)
        ttk.Label(shell,text="Airfoil + wing | 2D and 3D",style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell,text="Set section targets and wing dimensions. Generate once to update both views.").pack(anchor="w",pady=(5,3))
        ttk.Label(shell,text="Cl, Cd and Cm are section model estimates. Whole-wing performance remains unverified.",foreground="#9a3412").pack(anchor="w")
        self.metrics=tk.StringVar(value="Generate a design to see section predictions and wing dimensions.")
        ttk.Label(shell,textvariable=self.metrics,font=("Segoe UI",12,"bold")).pack(anchor="w",pady=(8,0))
        self.status=tk.StringVar(value="Ready. Starting values are editable examples.")
        ttk.Label(shell,textvariable=self.status,font=("Segoe UI",11,"bold"),wraplength=1250).pack(anchor="w",pady=10)
        body=ttk.Frame(shell); body.pack(fill="both",expand=True)
        controls=ttk.Frame(body,padding=(0,0,12,0)); controls.pack(side="left",fill="y")
        viewport=ttk.Frame(controls); viewport.pack(fill="both",expand=True)
        canvas=tk.Canvas(viewport,width=315,background="#f8fafc",highlightthickness=0)
        scrollbar=ttk.Scrollbar(viewport,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left",fill="both",expand=True); scrollbar.pack(side="right",fill="y")
        form=ttk.Frame(canvas); window=canvas.create_window((0,0),window=form,anchor="nw")
        form.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfigure(window,width=e.width))
        canvas.bind("<MouseWheel>",lambda e:canvas.yview_scroll(int(-e.delta/120),"units"))
        groups=[("airfoil","Airfoil section",[
            ("angle_of_attack_deg","Angle of attack (deg)"),("reynolds_number","Reynolds number"),("ncrit","Ncrit"),
            ("target_cl","Desired lift Cl"),("cl_tolerance","Allowed Cl difference"),("maximum_cd","Maximum drag Cd"),
            ("minimum_thickness_ratio","Minimum thickness (%)"),("maximum_thickness_ratio","Maximum thickness (%)")]),
            ("wing","Wing dimensions",[("span_m","Full span (m)"),("root_chord_m","Root chord (m)"),("tip_chord_m","Tip chord (m)"),
            ("leading_edge_sweep_deg","Leading-edge sweep (deg)"),("dihedral_deg","Dihedral (deg)"),
            ("root_twist_deg","Root twist (deg)"),("tip_twist_deg","Tip twist (deg)")])]
        row=0
        for section,title,items in groups:
            ttk.Label(form,text=title,font=("Segoe UI",11,"bold")).grid(row=row,column=0,columnspan=2,sticky="w",pady=(10,5)); row+=1
            for key,label in items:
                v=self.config[section][key]*(100 if "thickness_ratio" in key else 1)
                var=tk.StringVar(value=f"{v:g}"); self.fields[(section,key)]=var
                ttk.Label(form,text=label,wraplength=205).grid(row=row,column=0,sticky="w",pady=5,padx=(0,8))
                entry=ttk.Entry(form,textvariable=var,width=12); entry.grid(row=row,column=1,sticky="ew",pady=5)
                self.entries.append(entry); row+=1
            if section=="airfoil":
                bounds=self.bundle["metadata"]["condition_bounds"]
                note="Training ranges: "+" | ".join(f"{k}: {v[0]:g} to {v[1]:g}" for k,v in bounds.items())
                ttk.Label(form,text=note,wraplength=300,foreground="#475569").grid(row=row,column=0,columnspan=2,sticky="w",pady=8); row+=1
        form.columnconfigure(0,weight=1)
        self.button=ttk.Button(controls,text="Generate 2D + 3D",command=self.start); self.button.pack(fill="x",pady=(10,5))
        self.open_button=ttk.Button(controls,text="Open saved files",command=self.open_results,state="disabled"); self.open_button.pack(fill="x")
        panels=ttk.Panedwindow(body,orient="horizontal"); panels.pack(fill="both",expand=True)
        left,right=ttk.Frame(panels),ttk.Frame(panels); panels.add(left,weight=1); panels.add(right,weight=1)
        ttk.Label(left,text="2D | Airfoil section and wing top view",font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8,pady=6)
        view_controls=ttk.Frame(right); view_controls.pack(fill="x",padx=8,pady=3)
        ttk.Label(view_controls,text="3D | Drag to rotate",font=("Segoe UI",11,"bold")).pack(side="left")
        for label,view in [("Perspective","perspective"),("Top","top"),("Front","front")]:
            ttk.Button(view_controls,text=label,command=lambda v=view:self.set_view(v)).pack(side="left",padx=(6,0))
        self.figure_2d=Figure(figsize=(5,6),facecolor="#f8fafc")
        self.section_ax=self.figure_2d.add_subplot(211); self.planform_ax=self.figure_2d.add_subplot(212)
        self.figure_2d.subplots_adjust(left=.16,right=.97,bottom=.1,top=.94,hspace=.45)
        self.figure_3d=Figure(figsize=(6,6),facecolor="#f8fafc"); self.wing_ax=self.figure_3d.add_subplot(111,projection="3d")
        self.figure_3d.subplots_adjust(left=0,right=1,bottom=.02,top=.92)
        for ax in [self.section_ax,self.planform_ax,self.wing_ax]: ax.set_axis_off()
        self.canvas_2d=FigureCanvasTkAgg(self.figure_2d,master=left); self.canvas_3d=FigureCanvasTkAgg(self.figure_3d,master=right)
        self.canvas_2d.get_tk_widget().pack(fill="both",expand=True); self.canvas_3d.get_tk_widget().pack(fill="both",expand=True)
        self.toolbar_2d=NavigationToolbar2Tk(self.canvas_2d,left,pack_toolbar=False)
        self.toolbar_3d=NavigationToolbar2Tk(self.canvas_3d,right,pack_toolbar=False)
        self.toolbar_2d.pack(fill="x"); self.toolbar_3d.pack(fill="x")
        self.toolbar_2d.update(); self.toolbar_3d.update()
        self.detail=tk.StringVar(value="Each run saves images, coordinates, and STL/OBJ files together.")
        self.detail_label=ttk.Label(shell,textvariable=self.detail,wraplength=1400)
        self.detail_label.pack(anchor="w",pady=(10,0))
        shell.bind("<Configure>",lambda e:self.detail_label.configure(wraplength=max(300,e.width-32)))
        latest=ROOT/"results/generated/latest.json"
        if latest.exists():
            try: self.show(json.loads(latest.read_text()))
            except (OSError,KeyError,ValueError): self.status.set("Previous output unavailable. Click Generate.")
        for var in self.fields.values(): var.trace_add("write",self.inputs_changed)
        self.poll_id=root.after(150,self.poll)
        root.protocol("WM_DELETE_WINDOW",self.close)

    def conditions(self):
        c=deepcopy(self.config)
        for (section,key),var in self.fields.items():
            c[section][key]=float(var.get())/(100 if "thickness_ratio" in key else 1)
        validate_combined(c,self.bundle)
        return c

    def inputs_changed(self,*args):
        if self.generated_conditions is not None and not self.busy:
            try: current=self.conditions()
            except ValueError: current=None
            self.status.set(self.result_status if current==self.generated_conditions else "Inputs changed. Click Generate 2D + 3D to update both views.")

    def start(self):
        if self.busy: return
        try:
            c=self.conditions()
            save_json(ROOT/"config/combined_conditions.json",c)
        except (ValueError,KeyError,OSError) as error:
            messagebox.showerror("Check inputs",str(error)); return
        self.config=c
        self.busy=True; self.button.configure(state="disabled")
        for entry in self.entries: entry.configure(state="disabled")
        self.status.set("Searching airfoil candidates...")
        def worker():
            try:
                result=generate_combined(c,progress=lambda m:self.events.put(("progress",m)))
                self.events.put(("result",result))
            except Exception as error: self.events.put(("error",str(error)))
        threading.Thread(target=worker,daemon=True).start()

    def show(self,summary):
        folder=Path(summary["output_folder"])
        shape=np.load(folder/"airfoil_shape.npy")
        with np.load(folder/"wing_mesh.npz") as mesh: v,f=mesh["vertices_m"],mesh["faces"]
        self.section_ax.set_axis_on(); self.planform_ax.set_axis_on()
        draw_section(self.section_ax,shape); draw_planform(self.planform_ax,v); draw_wing(self.wing_ax,v,f)
        self.toolbar_2d.update(); self.toolbar_3d.update()
        self.canvas_2d.draw_idle(); self.canvas_3d.draw_idle()
        self.output=folder; self.generated_conditions=deepcopy(summary["conditions"])
        self.open_button.configure(state="normal")
        self.result_status="Ready. Predicted section targets matched (unverified)." if summary["status"]=="predicted_match" else "No predicted target match. Showing the closest valid candidate."
        self.status.set(self.result_status)
        p,d=summary["predictions"],summary["dimensions"]
        self.metrics.set(f"Section Cl {p['Cl']:.4f}    Cd {p['Cd']:.5f}    Cm {p['Cm']:.4f}    |    Span {d['span_m']:g} m    Area {d['reference_planform_area_m2']:.3f} m2")
        self.detail.set(f"Section Cl {p['Cl']:.4f} | Cd {p['Cd']:.5f} | Cm {p['Cm']:.4f} | Thickness {100*summary['maximum_thickness_ratio']:.1f}%\n"
            f"Wing area {d['reference_planform_area_m2']:.3f} m2 | Aspect ratio {d['reference_aspect_ratio']:.2f} | STL: mm; OBJ: m\nSaved: {folder}")
        self.inputs_changed()

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=="progress": self.status.set(value); continue
                self.busy=False; self.button.configure(state="normal")
                for entry in self.entries: entry.configure(state="normal")
                if kind=="result":
                    try: self.show(value)
                    except (OSError,KeyError,ValueError) as error:
                        self.status.set("Could not display the generated output.")
                        messagebox.showerror("Preview failed",str(error),parent=self.root)
                else:
                    self.status.set("Generation stopped; previous output remains displayed.")
                    messagebox.showerror("Generation failed",value)
        except queue.Empty: pass
        self.poll_id=self.root.after(150,self.poll)

    def set_view(self,view):
        elev,azim={"perspective":(28,-125),"top":(90,-90),"front":(0,0)}[view]
        self.wing_ax.view_init(elev=elev,azim=azim)
        self.canvas_3d.draw_idle()

    def close(self):
        self.root.after_cancel(self.poll_id)
        self.root.destroy()

    def open_results(self):
        if self.output: os.startfile(self.output)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke-test",action="store_true"); args=parser.parse_args()
    root=tk.Tk()
    if args.smoke_test: root.withdraw()
    app=CombinedApp(root)
    if args.smoke_test:
        root.update_idletasks(); app.canvas_2d.draw(); app.canvas_3d.draw()
        assert len(app.fields)==15 and app.output is not None
        var=app.fields[("wing","span_m")]; previous=var.get(); var.set("2.1")
        assert "Inputs changed" in app.status.get(); var.set(previous)
        assert app.status.get()==app.result_status
        assert len(app.section_ax.lines)==2 and len(app.planform_ax.lines)==2
        assert len(app.wing_ax.collections)==1
        app.set_view("top"); assert app.wing_ax.elev==90
        app.set_view("front"); assert app.wing_ax.azim==0
        app.set_view("perspective"); assert app.wing_ax.elev==28
        app.close(); print("GUI smoke test passed: 15 controls, synchronized 2D/3D views, camera presets and changed-input status")
    else: root.mainloop()

if __name__=="__main__": main()
