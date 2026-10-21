"""Training-only fault injection and bounded expert retries; no environment mutation."""
from __future__ import annotations

import numpy as np

SCENARIOS = (
    ("miss_grasp", 3, 1), ("miss_grasp", 2, -1),
    ("slip", 3, 1), ("slip", 2, -1),
    ("place_miss", 3, 1), ("place_miss", 1, -1),
)


class RecoveryController:
    def __init__(self, scenario, pick_objects, basket_params, table_center, table_half):
        self.kind, self.target_object, self.sign = scenario
        self.pick_objects = list(pick_objects)
        self.params = basket_params
        self.table_center = np.asarray(table_center, dtype=np.float32)
        self.table_half = np.asarray(table_half, dtype=np.float32)
        self.retries = {i: 0 for i in pick_objects}
        self.successful_attempts = {}
        self.events = []
        self.applied = False
        self.place_target = None
        self.record_index = 0

    def event(self, event, obj, **details):
        item = dict(step=self.record_index, event=event, object=int(obj),
                    attempt=self.retries[obj], **details)
        self.events.append(item)
        print("[RECOVERY] " + str(item), flush=True)

    def inside(self, position):
        p = np.asarray(position, dtype=np.float32)
        c = np.asarray(self.params["center"], dtype=np.float32)
        return bool(abs(p[0]-c[0]) <= self.params["half_x"]
                    and abs(p[1]-c[1]) <= self.params["half_y"]
                    and np.float32(self.params["table_top_z"]) <= p[2] <= np.float32(self.params["table_top_z"]+0.15))

    def in_support(self, xy, margin=0.12):
        return bool(np.all(np.abs(np.asarray(xy)-self.table_center) <= self.table_half-margin))

    def choose_missed_place(self, positions, obj):
        center = np.asarray(self.params["center"][:2], dtype=np.float32)
        for offset in ((-.26, .18*self.sign), (-.26, 0.), (-.26, -.18*self.sign)):
            xy = center+np.asarray(offset)
            if (not self.in_support(xy) or
                    any(np.linalg.norm(xy-p[:2]) < .22
                        for i,p in enumerate(positions,1) if i != obj)):
                continue
            return xy
        return None

    def adjust(self, obj, state, count, target, gripper, positions):
        """Return normal action targets only; all positions are environment-local."""
        p = np.asarray(target, dtype=np.float32).copy()
        intervention = False
        if obj != self.target_object or self.retries[obj] != 0:
            return p, gripper, intervention
        if self.kind == "miss_grasp" and state in ("PRE_GRASP","ALIGN","REACH","DESCEND","CLOSE"):
            p[1] += .09*self.sign
            intervention = True
        elif self.kind == "slip" and state == "LIFT":
            obj_pos = positions[obj-1]
            safe_height = self.params["table_top_z"]+.12 <= obj_pos[2] <= self.params["table_top_z"]+.35
            if (self.applied or (count >= 60 and safe_height and self.in_support(obj_pos[:2]))):
                gripper = "open"  # low vertical gravity release over support, never a throw
                intervention = True
        elif self.kind == "place_miss" and state in ("TRANSPORT","PLACE","OPEN","LIFT_RETRACT"):
            if self.place_target is None:
                self.place_target = self.choose_missed_place(positions, obj)
            if self.place_target is not None:
                p[:2] = self.place_target
                intervention = True
        if intervention and not self.applied:
            self.applied = True
            self.event("fault_applied", obj, kind=self.kind, stage=state)
        return p, gripper, intervention

    def check(self, obj, stage, stage_finished, positions, ee_position):
        if not stage_finished or stage not in ("LIFT","RETRACT"):
            return None
        pos = positions[obj-1]
        if self.inside(pos):
            self.successful_attempts[obj] = self.retries[obj]
            self.event("inside_basket_verified", obj)
            return "skip" if stage == "LIFT" else None
        if stage == "LIFT":
            held = bool(pos[2] >= self.params["table_top_z"]+.12
                        and np.linalg.norm(pos-np.asarray(ee_position)) <= .25)
            if held:
                self.event("lift_verified", obj)
                return None
            reason = "slip" if self.applied and self.kind == "slip" and obj == self.target_object else "grasp_failed"
        else:
            reason = "placement_failed"
        self.event("failure_observed", obj, reason=reason)
        if self.retries[obj] < 1:
            self.retries[obj] += 1
            self.event("retry_started", obj, reason=reason)
            return "retry"
        self.event("retry_exhausted", obj, reason=reason)
        return "skip"

    def annotate(self, record, obj, attempt, intervention):
        labels = record["training_labels"]
        labels.update(active_object=int(obj), recovery_attempt=int(attempt),
                      fault_request=self.kind if obj == self.target_object else "none",
                      fault_applied=bool(intervention), actor_valid=False)

    def finalize(self, records):
        if records:
            positions = records[-1]["training_labels"]["object_pos"]
            for obj in self.pick_objects:
                if self.inside(positions[obj-1]) and obj not in self.successful_attempts:
                    self.successful_attempts[obj] = self.retries[obj]
        for record in records:
            labels = record["training_labels"]
            obj, attempt = int(labels["active_object"]), int(labels["recovery_attempt"])
            deliberately_bad = obj == self.target_object and attempt == 0
            labels["actor_valid"] = bool(self.successful_attempts.get(obj) == attempt
                                          and not deliberately_bad)
        return dict(requested_kind=self.kind, target_object=self.target_object,
                    fault_applied=self.applied, retries=self.retries,
                    successful_attempts=self.successful_attempts, events=self.events,
                    actor_valid_steps=sum(r["training_labels"]["actor_valid"] for r in records),
                    actor_mask_note="Only verified successful object attempts; injected first attempt excluded")
