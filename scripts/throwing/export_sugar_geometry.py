"""Export original-asset static sugar/palm convex hulls; no simulation changes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import ConvexHull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    asset = "atec_robot_model/objects/task_e/pick_objects/004_sugar_box.usd"
    hand = "demo/throw_cem/gripper_geometry.json"
    stage = Usd.Stage.Open(str(project/asset))
    cache = UsdGeom.XformCache()
    points = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            transform = np.asarray(cache.GetLocalToWorldTransform(prim))
            points.append((np.c_[vertices,np.ones(len(vertices))]@transform)[:,:3])
    palm = json.loads((project/hand).read_text())["parts"]["gripper_base"]["vertices"]
    payload = {"format":"atec_sugar_palm_hulls_v1",
               "source_sha256":{p:hashlib.sha256((project/p).read_bytes()).hexdigest()
                                for p in (asset,hand)}}
    for name, vertices in (("sugar",np.concatenate(points)),("palm",np.asarray(palm))):
        hull = ConvexHull(vertices)
        payload[name] = {"vertices":vertices[hull.vertices].tolist(),
                         "normals":hull.equations[:,:3].tolist()}
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Preserve the existing geometry; choose a new output")
    destination.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"output":str(destination),"sugar_vertices":len(payload["sugar"]["vertices"])}))


if __name__ == "__main__":
    main()
