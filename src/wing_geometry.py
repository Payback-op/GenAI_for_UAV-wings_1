"""Closed full-wing loft using the exact sampled NASA airfoil section."""
from collections import Counter
import struct
import numpy as np
from common import X
from geometry import evaluate_surfaces, geometry_checks

def validate_wing(c):
    limits={"span_m":(.2,20),"root_chord_m":(.03,3),"tip_chord_m":(.03,3),
        "leading_edge_sweep_deg":(-30,45),"dihedral_deg":(-15,15),"root_twist_deg":(-15,15),"tip_twist_deg":(-15,15)}
    for key,(low,high) in limits.items():
        value=c.get(key)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or not low<=value<=high:
            raise ValueError(f"{key} must be between {low} and {high}")
    if c["tip_chord_m"]>c["root_chord_m"]:
        raise ValueError("Tip chord must not exceed root chord")
    if abs(c["tip_twist_deg"]-c["root_twist_deg"])>20:
        raise ValueError("Root to tip twist difference must be at most 20 degrees")
    n=c.get("span_stations")
    if isinstance(n,bool) or not isinstance(n,int) or not 5<=n<=101 or n%2!=1:
        raise ValueError("span_stations must be odd, between 5 and 101")

def check_mesh(v,f):
    triangles=v[f]
    area2=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    directed=Counter((int(a),int(b)) for face in f for a,b in zip(face,np.roll(face,-1)))
    undirected=Counter(tuple(sorted(e)) for e in directed.elements())
    volume=np.einsum("ij,ij->i",triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6
    return {"vertices":len(v),"triangles":len(f),"closed_manifold":bool(all(n==2 for n in undirected.values())),
        "consistent_orientation":bool(all(n==1 and directed[(b,a)]==1 for (a,b),n in directed.items())),
        "degenerate_faces":int((area2<=1e-14).sum()),"signed_volume_m3":float(volume),"finite_vertices":bool(np.isfinite(v).all())}

def build_mesh(c,shape):
    validate_wing(c)
    valid,thickness,_=geometry_checks(np.asarray(shape)[None,:])
    if not valid[0]:
        raise ValueError("Invalid airfoil for wing loft")
    _,u,l=evaluate_surfaces(shape)
    u,l=u[0],l[0]
    n=len(X)
    finite_te=abs(u[-1]-l[-1])>1e-9
    # One shared leading-edge vertex; preserve a finite trailing-edge gap.
    ring_x=np.r_[X[::-1],X[1:] if finite_te else X[1:-1]]
    ring_z=np.r_[u[::-1],l[1:] if finite_te else l[1:-1]]
    count=len(ring_x)
    vertices,stations=[],[]
    for y in np.linspace(-c["span_m"]/2,c["span_m"]/2,c["span_stations"]):
        eta=abs(y)/(c["span_m"]/2)
        chord=(1-eta)*c["root_chord_m"]+eta*c["tip_chord_m"]
        twist=np.deg2rad((1-eta)*c["root_twist_deg"]+eta*c["tip_twist_deg"])
        dx,z=(ring_x-.25)*chord,ring_z*chord
        datum_x=abs(y)*np.tan(np.deg2rad(c["leading_edge_sweep_deg"]))
        datum_z=abs(y)*np.tan(np.deg2rad(c["dihedral_deg"]))
        vertices.extend(np.column_stack([datum_x+.25*chord+np.cos(twist)*dx+np.sin(twist)*z,
            np.full(count,y),datum_z-np.sin(twist)*dx+np.cos(twist)*z]))
        stations.append({"y_m":float(y),"chord_m":float(chord),"twist_deg":float(np.rad2deg(twist)),
            "leading_edge_datum_x_m":float(datum_x),"dihedral_datum_z_m":float(datum_z),"maximum_thickness_ratio":float(thickness[0])})
    faces=[]
    for j in range(c["span_stations"]-1):
        for k in range(count):
            a,b=j*count+k,j*count+(k+1)%count
            cc,d=(j+1)*count+k,(j+1)*count+(k+1)%count
            faces.extend([(a,cc,d),(a,d,b)])
    def upper(j): return n-1-j
    def lower(j): return n-1 if j==0 else (0 if j==n-1 and not finite_te else n-1+j)
    for station in [0,c["span_stations"]-1]:
        offset=station*count
        for j in range(n-1):
            polygon=list(dict.fromkeys([upper(j),upper(j+1),lower(j+1),lower(j)]))
            for k in range(1,len(polygon)-1):
                face=[offset+polygon[0],offset+polygon[k],offset+polygon[k+1]]
                faces.append(face[::-1] if station==0 else face)
    v,f=np.asarray(vertices),np.asarray(faces,dtype=np.int32)
    checks=check_mesh(v,f)
    if checks["signed_volume_m3"]<0:
        f=f[:,::-1].copy(); checks=check_mesh(v,f)
    if not checks["closed_manifold"] or not checks["consistent_orientation"] or checks["degenerate_faces"] or checks["signed_volume_m3"]<=0:
        raise ValueError(f"Invalid loft: {checks}")
    area=c["span_m"]*(c["root_chord_m"]+c["tip_chord_m"])/2
    dimensions={"span_m":c["span_m"],"reference_planform_area_m2":area,"reference_aspect_ratio":c["span_m"]**2/area,
        "taper_ratio":c["tip_chord_m"]/c["root_chord_m"],"area_definition":"Untwisted trapezoidal reference planform"}
    return v,f,stations,dimensions,checks

def write_obj(path,v,f):
    with path.open("w",encoding="utf-8") as stream:
        stream.write("# BTP1 wing; coordinates in metres\n")
        for p in v: stream.write("v "+" ".join(f"{a:.10f}" for a in p)+"\n")
        for face in f: stream.write("f "+" ".join(str(int(i)+1) for i in face)+"\n")

def write_stl(path,v,f):
    with path.open("wb") as stream:
        stream.write(b"BTP1 NASA section loft; coordinates mm".ljust(80,b" "))
        stream.write(struct.pack("<I",len(f)))
        for face in f:
            p=v[face]*1000
            normal=np.cross(p[1]-p[0],p[2]-p[0]); normal/=np.linalg.norm(normal)
            stream.write(struct.pack("<12fH",*normal,*p.flatten(),0))
