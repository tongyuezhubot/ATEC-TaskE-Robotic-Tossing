"""Visual grasp primitive -> CEM upward swing -> visual settling observation."""
import numpy as np
from .config import DEFAULT_Q, HOME_Q, ARM_LOWER, ARM_UPPER
from .kinematics import PiperKinematics, grasp_rotations, rotation_log, quaternion_rotation, hand_table_clearance, mustard_palm_gap, sugar_palm_gap, open_hand_bottle_gap
from .planner import ThrowPlanner, NoFeasibleThrow


class ThrowController:
    def __init__(self, cfg, perception, residual=None, joint_model=None, profiles=None):
        self.profiles = profiles or {}
        self.cfg, self.perception = cfg, perception
        self.kin = PiperKinematics(cfg)
        self.planner = ThrowPlanner(cfg, self.kin, residual, seed=cfg.planner_seed, joint_model=joint_model)
        self.reset()

    def reset(self):
        self.step = self.stage_steps = self.stable = self.phase = self.retries = 0
        self.stage = "startup"; self.reason = ""
        self.order = self.cfg.object_order; self.poses = None; self.pose_step = 0
        self.visual_velocity = np.zeros((3, 3)); self.goal = self.target = None
        self.move_start = None; self.move_cursor = 0; self.plan = None; self.cursor = 0
        self.last_q = None
        self.last_hold_local = None
        self.preopen_hold_failures = 0
        self.trusted_grasp_geometry = False
        self.grasp_joint_goals = None
        self.stance_alignment_attempted = False
        self.stance_goal = None
        self.first_plan_failure = ""
        self.plan_attempts = 0
        self.plans = self.candidates = 0; self.visual_success = [False]*3
        self.upward_seen = self.separated_seen = False
        self.rotation = np.diag([1., -1., -1.])
        self.completed_profiles = {}
        for profile in self.profiles.values():
            for counter in ("calls","rejections"):
                if hasattr(profile["residual"],counter):
                    setattr(profile["residual"],counter,0)
        self.activate_profile(self.order[0])

    def activate_profile(self, obj):
        if not self.profiles:
            return
        profile = self.profiles[obj]
        self.cfg = profile["config"]; self.perception = profile["perception"]
        self.kin = PiperKinematics(self.cfg)
        self.planner = ThrowPlanner(self.cfg,self.kin,profile["residual"],
                                    seed=self.cfg.planner_seed,joint_model=profile["joint_model"])
        self.poses = None; self.pose_step = self.step
        self.visual_velocity = np.zeros((3,3)); self.last_hold_local = None
        self.plan = None; self.grasp_joint_goals = None
        self.stance_alignment_attempted = False; self.stance_goal = None
        self.first_plan_failure = ""; self.plan_attempts = 0
        self.profile_plan_baseline = self.plans
        self.profile_candidate_baseline = self.candidates
        for name in ("planning_start","planning_offset","planning_q"):
            if hasattr(self,name):
                delattr(self,name)

    def enter(self, stage):
        self.stage = stage; self.stage_steps = self.stable = 0; self.goal = None
        self.move_start = None; self.move_cursor = 0
        if stage == "swing":
            self.preopen_hold_failures = 0
        if stage == "home":
            self.trusted_grasp_geometry = False

    def abort(self, reason):
        self.reason = reason; self.enter("aborted")

    def observe(self, p, rgb, force=False):
        if self.poses is None or self.step % 5 == 0 or force:
            new = self.perception.observe(p, rgb)
            if (new.positions.shape != (3, 3) or new.quaternions.shape != (3, 4) or new.confidence.shape != (3,)
                    or not all(np.isfinite(x).all() for x in (new.positions, new.quaternions, new.confidence))):
                raise ValueError("Invalid visual pose estimates")
            if new.position_confidence is not None and (new.position_confidence.shape != (3,)
                    or not np.isfinite(new.position_confidence).all()):
                raise ValueError("Invalid root/COM confidence estimates")
            if self.poses is not None and self.step > self.pose_step:
                self.visual_velocity = (new.positions-self.poses.positions)/((self.step-self.pose_step)*self.cfg.dt)
            self.poses = new; self.pose_step = self.step

    def position_confidence(self, obj):
        scores = self.poses.position_confidence
        return self.poses.confidence[obj] if scores is None else scores[obj]

    def select_training_stance(self,q,obj,offset):
        model=self.planner.residual
        supported_grasp = ((obj==1 and self.cfg.mustard_body_grasp)
                           or (obj==0 and self.cfg.sugar_centered_grasp))
        if (not self.cfg.normalize_throw_stance or not supported_grasp
                or not getattr(model,"world_model",False)
                or not hasattr(model,"candidate_stances")
                or np.any(q[:6]<ARM_LOWER) or np.any(q[:6]>ARM_UPPER)):
            return None
        current=self.kin.forward(q);local=current[:3,:3].T@offset
        poses=model.candidate_stances(obj,q)
        if obj==0:
            # Move toward training poses only as far as needed, not beyond
            # the original 5cm carry cap. These are COMMAND targets: measured
            # observations remain untouched and are checked again after carry.
            poses=[q[:6]+fraction*(pose-q[:6]) for pose in poses for fraction in (.25,.5,.75,1.)]
            poses=sorted(poses,key=lambda pose:np.max(np.abs(pose-q[:6])))
        for pose in poses:
            target=q.copy();target[:6]=np.clip(pose,ARM_LOWER,ARM_UPPER)
            if obj==0 and hasattr(model,"bounds"):
                if (np.any(target[:6]<model.bounds[:6,0]+.03)
                        or np.any(target[:6]>model.bounds[:6,1]-.03)):
                    continue
            goal=self.kin.forward(target);change=np.max(np.abs(target[:6]-q[:6]))
            if not .01<=change<=.35 or np.linalg.norm(goal[:3,3]-current[:3,3])>.05:
                continue
            try:model.candidate_bounds(obj,target,goal[:3,:3]@local)
            except ValueError:continue
            safe=True
            for joint in np.linspace(q,target,9):
                tool=self.kin.forward(joint);point=tool[:3,3]+tool[:3,:3]@local
                near_rim=np.all(np.abs(point[:2]-np.asarray(self.cfg.basket_xyz[:2]))
                                <=np.asarray(self.cfg.basket_outer_half_xy)+self.cfg.held_rim_clearance)
                if (np.linalg.norm(tool[:3,3]-current[:3,3])>.05
                        or hand_table_clearance(tool[:3,3],tool[:3,:3],self.cfg.table_z)<.003
                        or point[2]<self.cfg.table_z+.06
                        or (near_rim and point[2]<self.cfg.basket_rim_z+self.cfg.held_rim_clearance)):
                    safe=False;break
            if safe:return target[:6].copy()
        return None

    def hold_proxy(self, q, obj):
        position = self.poses.positions[obj]
        tool = self.kin.forward(q); ee = tool[:3, 3]
        if obj == 2:
            com = position+quaternion_rotation(self.poses.quaternions[obj])@np.asarray(self.cfg.object_com_offsets[obj])
            local = tool[:3, :3].T@(com-ee)
            self.last_hold_local = local.copy()
            lo, hi = self.cfg.banana_held_local_z_range
            # Lift/proximity alone admits a banana dangling at the pad tips
            # (~14 cm), which escaped before opening in seed60. No force proof.
            if not lo <= local[2] <= hi or np.max(np.abs(local[:2])) > self.cfg.banana_held_lateral_max:
                return False
        return (self.position_confidence(obj) >= self.cfg.confidence_min
                and position[2] >= self.initial_object_z+.06
                and np.linalg.norm(position-ee) <= .20 and abs(q[6]-q[7]) <= self.cfg.held_gap_max[obj])

    def select_grasp(self, q, quat, obj, xyz):
        body_grasp = obj == 1 and self.cfg.mustard_body_grasp
        sugar_grasp = obj == 0 and self.cfg.sugar_centered_grasp
        neighbor = None
        if sugar_grasp and self.poses is not None:
            if min(self.position_confidence(1),self.poses.confidence[1])<self.cfg.confidence_min:
                raise ValueError("Neighbor bottle pose is not confident enough for sugar approach")
            neighbor = (self.poses.positions[1].copy(),quaternion_rotation(self.poses.quaternions[1]))
        self.grasp_joint_goals = None
        tilts = ((-45.,45.,-60.,60.,-75.,75.,-80.,80.) if body_grasp else
                 (-30.,30.,-45.,45.,-60.,60.,-75.,75.) if sugar_grasp else None)
        for rotation in grasp_rotations(quat, obj, tilts=tilts):
            candidate = np.asarray(xyz, float).copy()
            candidates = [candidate]
            check_clearance = body_grasp or sugar_grasp or (obj == 2 and self.cfg.static_hand_clearance)
            if sugar_grasp:
                actor_root = candidate-[0.,0.,self.cfg.grasp_offsets[obj]]
                object_rotation = quaternion_rotation(quat)
                center = actor_root+object_rotation@np.asarray(self.cfg.object_com_offsets[obj])
                # All pinch points stay inside the original 88mm half-length.
                # Geometry gates reject palm overlap before considering IK.
                axis = object_rotation[:,1].copy(); axis[2] = 0.
                axis /= max(np.linalg.norm(axis),1e-6)
                axis *= 1. if axis[1] >= 0. else -1.
                candidates = [center+shift*axis+[0.,0.,height]-rotation[:,2]*.120
                              for shift in (.045,-.045,.065,-.065,0.) for height in (.030,.015,0.)]
            if body_grasp:
                # Undo the legacy offset. A tall bottle cannot always be
                # reached at COM height without its cap hitting the palm;
                # bounded upper-body pinch points retain original pad depth.
                # Uses a predicted pose and static COM, never simulator truth.
                actor_root = candidate-np.array([.05, 0., self.cfg.grasp_offsets[obj]])
                object_rotation = quaternion_rotation(quat)
                center = actor_root+object_rotation@np.asarray(self.cfg.object_com_offsets[obj])
                candidates = [center+[0.,0.,height]-rotation[:,2]*.120 for height in (.030,.045,.060)]
            if obj == 2:
                # A vertical wrist offset is only valid for a vertical hand.
                # Center the banana between pads even when bounded IK selects
                # a tilted hand; otherwise the pads squeeze its edge (~5 cm off).
                actor_root = candidate-[0., 0., self.cfg.grasp_offsets[obj]]
                centered = actor_root-rotation[:, 2]*self.cfg.banana_pinch_distance
                # Shift along the banana's long axis, still inside the existing
                # 30mm lateral/132mm depth hold guard. This raises the hand,
                # rather than accepting a shallow pinch or driving into table.
                lateral = -np.sign(rotation[2,0])*min(.020,self.cfg.banana_held_lateral_max-.008)
                depth = min(self.cfg.banana_pinch_distance+.005,self.cfg.banana_held_local_z_range[1]-.007)
                shifted = actor_root-rotation@np.array([lateral,0.,self.cfg.banana_pinch_distance])
                raised = actor_root-rotation@np.array([lateral,0.,depth])
                candidates = [centered, shifted, raised] if self.cfg.static_hand_clearance else [centered]
            for candidate in candidates:
                if check_clearance and hand_table_clearance(candidate,rotation,self.cfg.table_z)<.003:
                    continue
                if body_grasp and mustard_palm_gap(candidate,rotation,actor_root,object_rotation)<.010:
                    continue
                if sugar_grasp and sugar_palm_gap(candidate,rotation,actor_root,object_rotation)<.010:
                    continue
                if neighbor is not None and open_hand_bottle_gap(candidate,rotation,*neighbor)<.010:
                    continue
                try:
                    if body_grasp:
                        # A top-down COM grasp puts the tall cap into the palm.
                        # Enter along tool-Z from the side. Solve from the grasp
                        # branch first, then retain these exact joint goals so
                        # runtime IK cannot jump back to an unreachable branch.
                        down = self.kin.solve(candidate,rotation,q,max_iterations=120)
                        pre = self.kin.solve(candidate-rotation[:,2]*self.cfg.pre_grasp_clearance,
                                             rotation,down,max_iterations=120)
                        lifted = self.kin.solve(candidate+[0.,0.,self.cfg.lift_clearance],rotation,
                                                down,max_iterations=120)
                    elif sugar_grasp:
                        # Solve at the actual pad point first, retaining one
                        # branch through the short vertical approach and lift.
                        down = self.kin.solve(candidate,rotation,q,max_iterations=120)
                        pre = self.kin.solve(candidate+[0.,0.,self.cfg.pre_grasp_clearance],
                                             rotation,down,max_iterations=120)
                        # Keep wrist joints fixed while the arm raises the
                        # held box. Fixed WORLD orientation can be unreachable
                        # at the wrist limit, even when raising is feasible.
                        lifted = self.kin.solve_position(candidate+[0.,0.,self.cfg.lift_clearance],
                                                         down,max_iterations=120)
                        down_tool = self.kin.forward(down)
                        held_local = down_tool[:3,:3].T@(actor_root-down_tool[:3,3])
                        lift_tool = self.kin.forward(lifted)
                        if (lift_tool[:3,3]+lift_tool[:3,:3]@held_local)[2]<actor_root[2]+.060:
                            continue
                        if min(hand_table_clearance(self.kin.forward(j)[:3,3],self.kin.forward(j)[:3,:3],self.cfg.table_z)
                               for j in np.linspace(down,lifted,9))<.003:
                            continue
                        if neighbor is not None:
                            # Also screen the home->pre and down->lift samples;
                            # target clearance alone cannot prove approach safe.
                            path = list(np.linspace(q[:6],pre,9))+list(np.linspace(down,lifted,9))
                            if any(open_hand_bottle_gap(self.kin.forward(j)[:3,3],self.kin.forward(j)[:3,:3],*neighbor)<.010
                                   for j in path):
                                continue
                    else:
                        pre = self.kin.solve(candidate+[0., 0., self.cfg.pre_grasp_clearance], rotation, q)
                        down = self.kin.solve(candidate, rotation, pre)
                        lifted = self.kin.solve(candidate+[0., 0., self.cfg.lift_clearance], rotation, down)
                    if check_clearance:
                        # Check achieved IK orientation and the bounded joint
                        # interpolation, not only the nominal Cartesian target.
                        clearances = []
                        for joint in np.linspace(pre,down,9):
                            tool = self.kin.forward(joint)
                            clearances.append(hand_table_clearance(tool[:3,3],tool[:3,:3],self.cfg.table_z))
                            if body_grasp and mustard_palm_gap(tool[:3,3],tool[:3,:3],actor_root,object_rotation)<.010:
                                raise ValueError("Approach palm/bottle projection gap below 10mm")
                            if sugar_grasp and sugar_palm_gap(tool[:3,3],tool[:3,:3],actor_root,object_rotation)<.010:
                                raise ValueError("Approach palm/sugar projection gap below 10mm")
                            if neighbor is not None and open_hand_bottle_gap(tool[:3,3],tool[:3,:3],*neighbor)<.010:
                                raise ValueError("Approach hand/neighbor-bottle projection gap below 10mm")
                        if min(clearances)<.003:
                            continue
                    if obj == 2 and self.cfg.banana_keep_grasp_orientation and self.cfg.throw_primitive != "joint_swing":
                        self.kin.solve(np.asarray(self.cfg.release_ee_xyz)+self.cfg.pre_throw_delta,
                                       rotation, lifted, max_iterations=120,
                                       position_tolerance=1e-5, rotation_tolerance=1e-4)
                    if body_grasp or sugar_grasp:
                        self.grasp_joint_goals = dict(pre_grasp=pre,descend=down,lift=lifted)
                    return rotation, candidate
                except ValueError:
                    continue
        raise ValueError("No bounded pre-grasp/descent/lift orientation")

    def move(self, q, velocity, position, rotation, opened=False, contact_approach=False, precise=False,
             joint_goal=None):
        if self.goal is None:
            if joint_goal is not None:
                self.goal = np.asarray(joint_goal, float).copy()
            elif precise:
                self.goal = self.kin.solve(position, rotation, q, max_iterations=120,
                                           position_tolerance=1e-5, rotation_tolerance=1e-4)
            else:
                self.goal = self.kin.solve(position, rotation, q)
            if self.goal.shape != (6,) or np.any(self.goal < ARM_LOWER) or np.any(self.goal > ARM_UPPER):
                raise ValueError("Carry target violates native joint limits")
            active = self.order[self.phase]
            if contact_approach and ((active==2 and self.cfg.static_hand_clearance)
                                     or (active==1 and self.cfg.mustard_body_grasp)
                                     or (active==0 and self.cfg.sugar_centered_grasp)):
                actual_goal = self.kin.forward(self.goal)
                if hand_table_clearance(actual_goal[:3,3],actual_goal[:3,:3],self.cfg.table_z)<.003:
                    raise ValueError("Approach target violates static hand/table clearance")
            self.move_start = q[:6].copy()
            distance = float(np.max(np.abs(self.goal-self.move_start)))
            # Grasp/transport is not the throw: avoid applying the faster swing
            # caps to a long reorientation while holding a slippery object.
            self.move_duration = max(.4, 1.875*distance/self.cfg.carry_max_joint_speed,
                                     np.sqrt(5.774*distance/self.cfg.carry_max_joint_acceleration))
        self.move_cursor += 1
        s = min(1., self.move_cursor*self.cfg.dt/self.move_duration)
        blend = 10*s**3-15*s**4+6*s**5
        result = q.copy(); result[:6] = self.move_start+blend*(self.goal-self.move_start)
        result[6:] = [.035, -.035] if opened else [-.015, .015]
        close = np.max(np.abs(q[:6]-self.goal)) < (.025 if precise else .04) and np.max(np.abs(velocity[:6])) < .15
        if contact_approach:
            # Physical fingers can contact the object/support before the nominal
            # EE offset. Requiring exact joint convergence then pushes forever.
            # Banana's pad-tip guard leaves only 12mm above the nominal pinch
            # depth; the generic 35mm contact shortcut accepted shallow grasps.
            # Require <=8mm above target even if the joint convergence gate fires.
            actual = self.kin.forward(q)
            error = actual[:3, 3]-np.asarray(position)
            active = self.order[self.phase]
            max_above = .008 if active == 2 or (active == 1 and self.cfg.mustard_body_grasp) else .035
            aligned = np.linalg.norm(error[:2]) <= .015 and -.005 <= error[2] <= max_above
            if (active == 1 and self.cfg.mustard_body_grasp) or (active == 0 and self.cfg.sugar_centered_grasp):
                aligned = np.linalg.norm(error) <= .008
            near_contact = (s >= 1. and aligned
                            and np.linalg.norm(rotation_log(rotation @ actual[:3, :3].T)) <= self.cfg.pose_rotation_tolerance
                            and np.max(np.abs(velocity[:6])) < .4)
            close = (close and aligned) or near_contact
        self.stable = self.stable+1 if close else 0
        if contact_approach and self.stable >= 3:
            result[:6] = q[:6]  # stop pressing; close at the achieved approach
        return result, self.stable >= 3

    def action(self, p, rgb):
        cfg = self.cfg; q = p[:8]+DEFAULT_Q
        # Native contact constraints can leave nonzero reported joint velocity
        # at an effectively stationary pose. Settle tests use measured position
        # change, not that persistent solver velocity. Still legal proprio only.
        velocity = p[8:16] if self.last_q is None else (q-self.last_q)/cfg.dt
        self.last_q = q.copy()
        self.step += 1; self.stage_steps += 1
        if self.step >= cfg.max_steps:
            self.abort("hard_step_budget")
        if self.stage == "startup":
            if self.step >= 25:
                self.enter("home"); self.target = q.copy()
            return np.zeros(8), False
        if self.stage in ("aborted", "predicted_complete"):
            return (q-DEFAULT_Q)/cfg.action_scale, True
        if self.stage not in ("swing", "wait_landing") and self.stage_steps > cfg.stage_timeout:
            self.abort("stage_timeout")
            return (q-DEFAULT_Q)/cfg.action_scale, True
        # Pad-depth compares visual object position with CURRENT joint FK.
        # During transport, a five-step-old image creates a false slip signal.
        # Use same-frame RGB in hold checks as well as the short upward apex.
        self.observe(p, rgb, force=self.stage in ("swing", "pre_throw", "check_hold", "calibration_wrist", "align_start"))
        obj = self.order[self.phase]
        fixed_grasp_motion = (self.trusted_grasp_geometry and
                              self.stage in ("pre_grasp", "descend", "close", "lift"))
        # These stages already execute the stored grasp pose; they do not use
        # a newly estimated object pose to change the trajectory. Finger
        # occlusion must not deadlock that finite motion. Fresh confidence is
        # still mandatory for check_hold, wrist alignment and CEM planning.
        if (self.stage not in ("home", "swing", "wait_landing") and not fixed_grasp_motion
                and self.position_confidence(obj) < cfg.confidence_min):
            # Position-drive contact force depends on the closing command, not
            # merely on measured jaw width. Replacing it with actual q releases
            # finger pressure and changes the last-action input during a pause.
            waiting = q if self.target is None else self.target
            return (waiting-DEFAULT_Q)/cfg.action_scale, False
        try:
            if self.stage == "home":
                pose = self.kin.forward(HOME_Q)
                target, done = self.move(q, velocity, pose[:3, 3], pose[:3, :3], opened=True)
                if done and self.poses.confidence[obj] >= cfg.confidence_min:
                    xyz = self.poses.positions[obj].copy()
                    self.initial_object_z = xyz[2]
                    xyz += [.05 if obj == 1 else 0., 0., cfg.grasp_offsets[obj]]
                    self.rotation, self.grasp_xyz = self.select_grasp(q, self.poses.quaternions[obj], obj, xyz)
                    self.trusted_grasp_geometry = True
                    self.enter("pre_grasp")
            elif self.stage == "pre_grasp":
                pre_xyz = (self.grasp_xyz-self.rotation[:,2]*cfg.pre_grasp_clearance
                           if obj == 1 and cfg.mustard_body_grasp else self.grasp_xyz+[0.,0.,cfg.pre_grasp_clearance])
                goal = None if self.grasp_joint_goals is None else self.grasp_joint_goals["pre_grasp"]
                target, done = self.move(q, velocity, pre_xyz, self.rotation, opened=True,joint_goal=goal)
                if done:
                    self.enter("descend")
            elif self.stage == "descend":
                goal = None if self.grasp_joint_goals is None else self.grasp_joint_goals["descend"]
                target, done = self.move(q, velocity, self.grasp_xyz, self.rotation, opened=True, contact_approach=True,joint_goal=goal)
                if done:
                    self.enter("close")
            elif self.stage == "close":
                target = self.target.copy(); target[6:] = [-.015, .015]
                if self.stage_steps >= 40:
                    self.enter("lift")
            elif self.stage == "lift":
                goal = None if self.grasp_joint_goals is None else self.grasp_joint_goals["lift"]
                target, done = self.move(q, velocity, self.grasp_xyz+[0., 0., cfg.lift_clearance], self.rotation,joint_goal=goal)
                if done:
                    self.enter("check_hold")
            elif self.stage == "check_hold":
                target = self.target.copy()
                self.stable = self.stable+1 if self.hold_proxy(q, obj) else 0
                if self.stable >= 15:
                    # Same selected grasp orientation, already prechecked for
                    # the carry stance. Native contact tracking may differ by
                    # the small approach tolerance; don't impose a HOME rotation.
                    self.held_rotation = self.rotation.copy()
                    self.enter(("calibration_wrist" if cfg.collection_wrist_q6 is not None else "plan")
                               if cfg.throw_primitive == "joint_swing" else "pre_throw")
                elif self.stage_steps > 80:
                    if self.retries >= 1:
                        self.abort("visual_grasp_not_confirmed")
                    else:
                        self.retries += 1; self.enter("home")
            elif self.stage == "calibration_wrist":
                if not self.hold_proxy(q, obj):
                    self.abort("lost_visual_hold_during_wrist_calibration"); target = q.copy()
                else:
                    goal = q[:6].copy(); goal[5] = cfg.collection_wrist_q6
                    target, done = self.move(q, velocity, None, None, joint_goal=goal)
                    if done:
                        self.enter("plan")
            elif self.stage == "pre_throw":
                if not self.hold_proxy(q, obj):
                    self.abort("lost_visual_hold_before_throw"); target = q.copy()
                else:
                    start = np.asarray(cfg.release_ee_xyz)+cfg.pre_throw_delta
                    # Preserve the confirmed banana grasp orientation when
                    # feasible. Others keep the prior HOME-based throw posture.
                    # Always remeasure the held point before planning.
                    throw_rotation = (self.held_rotation if obj == 2 and cfg.banana_keep_grasp_orientation
                                      else self.kin.throw_rotation())
                    target, done = self.move(q, velocity, start, throw_rotation, precise=True)
                    if done:
                        self.enter("plan")
            elif self.stage == "align_start":
                if not self.hold_proxy(q,obj):
                    self.abort("lost_visual_hold_during_start_alignment");target=q.copy()
                else:
                    target,done=self.move(q,velocity,None,None,joint_goal=self.stance_goal,precise=True)
                    if done:self.enter("plan")
            elif self.stage == "plan":
                target = self.target.copy()
                self.observe(p, rgb, force=True)
                if not self.hold_proxy(q, obj):
                    self.abort("lost_visual_hold_before_plan")
                else:
                    settled = (np.max(np.abs(velocity[:6])) <= .15
                               and np.linalg.norm(self.visual_velocity[obj]) <= .10)
                    self.stable = self.stable+1 if settled else 0
                    if self.stable >= 3:
                        object_com = self.poses.positions[obj]+quaternion_rotation(self.poses.quaternions[obj])@np.asarray(cfg.object_com_offsets[obj])
                        offset = object_com-self.kin.forward(q)[:3, 3]
                        self.planning_start = self.kin.forward(q)[:3, 3].copy()
                        self.planning_offset = offset.copy()
                        self.planning_q = q.copy()
                        self.plan_attempts += 1
                        try:
                            self.plan = self.planner.plan(q, obj, offset)
                        except NoFeasibleThrow as error:
                            stance=(None if self.stance_alignment_attempted else self.select_training_stance(q,obj,offset))
                            if stance is None:raise
                            self.stance_alignment_attempted=True;self.stance_goal=stance
                            self.first_plan_failure=str(error);self.enter("align_start")
                        else:
                            self.plans += 1; self.candidates += self.plan.evaluated_candidates
                            self.cursor = 0; self.enter("swing")
                    elif self.stage_steps >= 50:
                        self.abort("pre_throw_not_stable")
            elif self.stage == "swing":
                target = self.plan.targets[self.cursor].copy()
                if self.cursor < self.plan.prediction.open_step:
                    # Do not let a planned release disguise an object that has
                    # already slipped from the fingers during the fast swing.
                    self.preopen_hold_failures = (0 if self.hold_proxy(q, obj)
                                                  else self.preopen_hold_failures+1)
                    if self.preopen_hold_failures >= 3:
                        self.abort("lost_visual_hold_during_swing"); target = q.copy()
                if self.cursor > 0:
                    error = np.max(np.abs(q[:6]-self.plan.tracking_targets[self.cursor-1, :6]))
                    self.stable = self.stable+1 if error > cfg.max_tracking_error else 0
                    if self.stable >= cfg.tracking_error_steps:
                        self.abort("swing_tracking_limit"); target = q.copy()
                        target[6:] = [.035, -.035] if self.cursor >= self.plan.prediction.open_step else [-.015, .015]
                if self.cursor == self.plan.prediction.open_step:
                    self.release_visual_xyz = self.poses.positions[obj].copy()
                    self.upward_seen = self.separated_seen = False
                if self.cursor >= self.plan.prediction.open_step:
                    self.observe_flight(q, obj)
                self.cursor += 1
                if self.cursor == len(self.plan.targets) and self.stage == "swing":
                    self.enter("wait_landing")
            elif self.stage == "wait_landing":
                target = self.target.copy(); target[6:] = [.035, -.035]
                self.observe_flight(q, obj)
                xyz = self.poses.positions[obj]
                inside = (np.all(np.abs(xyz[:2]-np.asarray(cfg.basket_xyz[:2])) <= cfg.basket_half_xy)
                          and cfg.table_z <= xyz[2] <= cfg.table_z+.15)
                settled = np.linalg.norm(self.visual_velocity[obj]) < .10
                ok = (self.upward_seen and self.separated_seen and inside and settled
                      and self.position_confidence(obj) >= cfg.confidence_min)
                self.stable = self.stable+1 if ok else 0
                if self.stable >= cfg.settle_steps:
                    self.visual_success[obj] = True
                    if self.profiles:
                        self.completed_profiles[str(obj+1)] = dict(
                            visual_settled=True,cem_plans=self.plans-self.profile_plan_baseline,
                            candidates=self.candidates-self.profile_candidate_baseline,
                            world_calls=getattr(self.planner.residual,"calls",0))
                    if self.phase == len(self.order)-1:
                        self.enter("predicted_complete")
                    else:
                        self.phase += 1; self.retries = 0; self.enter("home")
                        self.activate_profile(self.order[self.phase])
                elif self.stage_steps >= cfg.max_flight_steps:
                    self.abort("throw_not_visually_settled_in_basket")
            else:
                raise ValueError("Unknown throw stage")
        except (ValueError, NoFeasibleThrow) as error:
            self.abort(str(error)); target = q.copy()
        self.target = target.copy()
        if not np.isfinite(target).all():
            raise ValueError("Non-finite joint command")
        return (target-DEFAULT_Q)/cfg.action_scale, self.stage in ("aborted", "predicted_complete")

    def observe_flight(self, q, obj):
        position = self.poses.positions[obj]
        if self.position_confidence(obj) < self.cfg.confidence_min:
            return
        self.upward_seen |= bool(self.visual_velocity[obj, 2] > .15)
        self.separated_seen |= bool(abs(q[6]-q[7]) >= self.cfg.open_gap_min and
                                    np.linalg.norm(position-self.release_visual_xyz) > .04)

    def diagnostics(self):
        result = {"stage": self.stage, "object": self.order[self.phase]+1,
                  "steps": self.step, "cem_plans": self.plans, "candidate_evaluations": self.candidates,
                  "visual_success_only": self.visual_success, "reason": self.reason,
                  "actor_used": False, "jepa_throw_scoring_used": False,
                  "residual_calibrated": self.planner.residual.calibrated,
                  "release_world_model_scoring_used": bool(getattr(self.planner.residual, "world_model", False)),
                  "release_world_model_calls": getattr(self.planner.residual, "calls", 0),
                  "release_world_model_domain_rejections": getattr(self.planner.residual, "rejections", 0),
                  "joint_response_calibrated": self.planner.joint_model.calibrated,
                  "joint_response_alpha": self.planner.joint_model.alpha.tolist(),
                  "hold_check": "visual lift/proximity/gripper + banana pad-depth proxy, not contact proof"}
        result["trusted_grasp_geometry"] = self.trusted_grasp_geometry
        if self.profiles:
            result["per_object_profiles_enabled"] = True
            result["completed_profiles"] = dict(self.completed_profiles)
            result["profile_world_calls"] = {str(o+1):getattr(p["residual"],"calls",0)
                                             for o,p in self.profiles.items()}
        result["static_hand_clearance_enabled"] = self.cfg.static_hand_clearance
        result["start_stance_alignment_attempted"] = self.stance_alignment_attempted
        result["cem_plan_attempts"] = self.plan_attempts
        if self.stance_goal is not None:
            result["start_stance_target_q"] = self.stance_goal.tolist()
            result["first_plan_failure"] = self.first_plan_failure
        if self.last_hold_local is not None:
            result["banana_held_local_offset"] = self.last_hold_local.tolist()
        if self.poses is not None:
            result["visual_confidence"] = self.poses.confidence.tolist()
            if self.poses.position_confidence is not None:
                result["visual_position_confidence"] = self.poses.position_confidence.tolist()
            result["visual_object_xyz"] = self.poses.positions.tolist()
        if hasattr(self,"grasp_xyz"):
            result["grasp_target_xyz"] = self.grasp_xyz.tolist()
            result["grasp_rotation"] = self.rotation.tolist()
        if hasattr(self, "planning_q"):
            result.update(plan_start=self.planning_start.tolist(),
                          object_offset=self.planning_offset.tolist(),
                          plan_q_start=self.planning_q.tolist())
        if self.plan:
            result.update(parameters=self.plan.theta.tolist(), predicted_landing=self.plan.prediction.landing_xy.tolist(),
                          swing_cursor=self.cursor, open_step=self.plan.prediction.open_step,
                          predicted_release_xyz=self.plan.prediction.release_position.tolist(),
                          predicted_release_velocity=self.plan.prediction.release_velocity.tolist())
        return result
