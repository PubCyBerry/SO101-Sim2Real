# ECE 4560 Assignment 8 Block Stacking — Engineering Brief

**Task**: Reproduce Assignment 8 Parts 2 & 3 (stack 2–3 blocks) on real SO-101 hardware, then generate LeRobot VLA training data.

**Status**: Briefing document only (no code implementation yet). Synthesized from ECE 4560 assignment pages and demo video metadata.

---

## 1. Task Definition

### Part 2: Stack 2 Blocks
- **Objective**: Pick up two blocks sequentially and stack the second on top of the first.
- **Block dimensions**: 0.0285 m cubic foam blocks (mass ~0.035 kg, physics tuned for stable grasping).
- **Test positions**:
  - Block 1 pickup: [0.25, 0.1, 0.01425] m (world frame, top surface at z=0.01425 m)
  - Block 2 pickup: [0.25, -0.1, 0.01425] m (world frame, same height)
  - Stack placement: Block 1 remains at [0.25, 0.1, ?]; Block 2 placed at [0.25, 0.1, 0.043] m (on top of Block 1)
- **Height reference**: z=0 is workspace floor; z-heights inferred from sim setup:
  - Block 1 floor resting height: z=0.014 m (center of mass above workspace surface)
  - Block 2 stacking height on Block 1: z=0.043 m (0.014 + 0.0285 block thickness ≈ 0.0425; adjusted to 0.043 in spec)
  - Block 3 (Part 3 only) stacking height: z=0.071 m (0.043 + 0.0285 ≈ 0.0715; adjusted to 0.071)
- **Pick order**: Sequential deterministic (Block 1 first, then Block 2 at different [x,y] location → place on Block 1).
- **Gripper states**:
  - Open: 50 (normalized 0–100 range)
  - Closed: 5 (grasp state)

### Part 3: Stack 3 Blocks (Optional Extra Credit)
- **Same pattern** as Part 2, extended to three blocks:
  - Block 1 pickup at z=0.014 m
  - Block 2 placement at z=0.043 m (on Block 1)
  - Block 3 placement at z=0.071 m (on Block 2)
- **Success metric**: All three blocks stacked vertically without toppling.

### Workspace Verification
- All target configurations must be verified against joint limits at each z-height via **acrylic grid reference** (course uses physical grid overlay for validation).
- **NaN detection**: IK solver must flag out-of-reach configurations; stacking sequence must adjust xy placement if necessary to remain within reachable workspace at target z-height.

---

## 2. Kinematic Conventions

### SO-101 Arm Configuration
**6-DOF System**: 5 actuated arm joints + 1 gripper joint
| Joint Name | DOF | Range (rad) | Range (deg) | Motor | Notes |
|---|---|---|---|---|---|
| shoulder_pan (θ₁) | 1 | ±1.919 | ±110 | STS3215 M1 | Base rotation; ground plane |
| shoulder_lift (θ₂) | 2 | ±1.74 | ±100 | STS3215 M2 | Shoulder elevation |
| elbow_flex (θ₃) | 3 | ±1.69 | ±97 | STS3215 M3 | Elbow extension |
| wrist_flex (θ₄) | 4 | ±1.65 | ±94.5 | STS3215 M4 | Wrist flex (pitch) |
| wrist_roll (θ₅) | 5 | ±2.74 to 2.84 | ±157 to 163 | STS3215 M5 | Wrist roll (yaw) |
| gripper (θ₆) | 1 | 0–100 | normalized | STS3215 M6 | Proportional grip; 5=closed, 50=open |

**Key constraint**: 5-DOF arm cannot achieve arbitrary 6-DOF pose (position + orientation). Assignment uses **position-first, orientation best-effort** strategy.

### Kinematic Method (ECE 4560 Course Standard)

**Forward Kinematics**: Product of Exponentials (Lie Groups) / transformation matrices
- **Reference frame**: World frame (W) at base of robot.
- **Tool frame**: End effector (gripper pointing downward, vertical approach grasp).
- **Kinematic chain**: G_wt = G_w1 × G_12 × G_23 × G_34 × G_45 × G_5t
  - G_w1: Base offset + rotation (θ₁ component)
  - G_12, G_23, G_34, G_45, G_5t: Joint transforms via exponential map

**Inverse Kinematics**: Geometric decoupled solution (Assignment 7, required prerequisite for Assignment 8)
- **Sequential solution order**: θ₁ → θ₂, θ₃ → θ₄ → θ₅
  1. **θ₁ (shoulder_pan)**: Ground plane projection from target [x, y] position, accounting for frame x-offset (0.0388353 m).
     ```
     θ₁ = atan2(y, x - frame_offset_x)
     ```
  2. **θ₂, θ₃ (shoulder_lift, elbow_flex)**: Planar 2-link arm in vertical plane via Law of Cosines.
     - Desired wrist position gw4 computed from gwt (desired target) via:
     ```
     gwt = [target position + orientation]
     g4t = G_45(0) × G_5t  (FK from Assignment 6)
     gw4 = gwt × inv(g4t)
     ```
     - Extract z and radial distance from gw4; solve triangle (L1, L2 link lengths) for θ₂, θ₃.
  3. **θ₄ (wrist_flex)**: Corrects wrist orientation to align tool z-axis with world z (vertical grasp).
     - From orientation difference via FK and projection.
  4. **θ₅ (wrist_roll)**: Function of θ₁ only; exploits double negation of opposite joint axes.
     ```
     θ₅ = θ₁  (or -θ₁, depending on convention)
     ```

**Workspace Analysis**: Random target sampling within reachable region; **NaN detection** signals out-of-reach.

### Mapping to Existing `SO101Kinematics` (This Project)

Our codebase references:
- **`sim_to_real/` package** (`src/sim_to_real/__init__.py`): Gym environment registration.
- **Tasks**: `pick_pen/` and `pick_cube/` with env configs + MDP modules.
- **No explicit standalone kinematics module** (IK solver) found in root; likely vectorized via Isaac Sim's Lula IK or ikpy (imported in `pyproject.toml`).

**Action required**: Assignment 8 implementation must either:
1. Call a student-provided `get_inverse_kinematics(target_position, target_orientation)` function (per Assignment 7 spec).
2. Use Isaac Lab's built-in Lula IK solver (if sim path).
3. Use ikpy analytical IK (fallback, pure Python).

**For hardware path**: Implement Assignment 7 geometric IK analytically to avoid runtime overhead.

---

## 3. Cubic Spline Trajectory Generation (Assignment 9 Formulation)

### Mathematical Foundation

**Standard cubic polynomial** for joint-space trajectory:
```
position(t) = a₀ + a₁·t + a₂·t² + a₃·t³
velocity(t) = a₁ + 2·a₂·t + 3·a₃·t²
acceleration(t) = 2·a₂ + 6·a₃·t
```

### Boundary Conditions (Course Specification)

**Zero-velocity (clamped) boundary conditions**:
- p(0) = θ₀ (starting position)
- p(T) = θf (final position)
- v(0) = 0 (zero initial velocity)
- v(T) = 0 (zero final velocity)

### Coefficient Derivation

From boundary conditions:
1. **a₀ = θ₀** (from p(0) = a₀)
2. **a₁ = 0** (from v(0) = a₁ = 0)
3. **a₂ = 3(θf − θ₀) / T²** (from p(T) and v(T) constraints)
4. **a₃ = −2(θf − θ₀) / T³** (from p(T) and v(T) constraints)

**Explicit derivation** (for verification):
- v(0) = a₁ = 0 ✓
- v(T) = a₁ + 2·a₂·T + 3·a₃·T² = 0 + 2·a₂·T + 3·a₃·T² = 0
  - Substituting a₂ and a₃: 2·(3Δ/T²)·T + 3·(−2Δ/T³)·T² = 6Δ/T − 6Δ/T = 0 ✓
- p(T) = a₀ + a₁·T + a₂·T² + a₃·T³ = θ₀ + 0 + (3Δ/T²)·T² + (−2Δ/T³)·T³
  - = θ₀ + 3Δ − 2Δ = θ₀ + Δ = θf ✓

### Per-Joint Independence

**Each joint i is parameterized independently**:
```python
coeff_i = {
    'a0': θ₀[i],
    'a1': 0,
    'a2': 3*(θf[i] - θ₀[i]) / duration²,
    'a3': -2*(θf[i] - θ₀[i]) / duration³
}
```

No coupling between joints; no via-point continuity constraints (each segment is isolated start→goal).

### Trajectory Evaluation

**At control timestep t ∈ [0, duration]**:
```python
# Clamp time to valid range
tlim = min(max(t, 0), duration)

# Evaluate position
pos[i] = a0[i] + a1[i]·tlim + a2[i]·tlim² + a3[i]·tlim³

# Evaluate velocity (optional feedforward)
vel[i] = a1[i] + 2·a2[i]·tlim + 3·a3[i]·tlim²
```

### Comparison to Baseline Methods

| Method | Continuity | Smoothness | Hardware Support |
|---|---|---|---|
| **Direct Commands** | None | Jerky (step changes) | ✓ Position-only |
| **Linear Interpolation** | C⁰ (position) | Piecewise linear; C⁻¹ accel | ✓ Position-only |
| **Cubic Spline** | C¹ (pos, vel) | Smooth vel; C⁻¹ accel | ✓ Position-only (velocity unsupported on SO-101 hardware) |
| **Quintic Spline** | C² (pos, vel, accel) | Smoothest | — (not required for course) |

**Course focus**: Cubic splines chosen for smooth velocity profiles without requiring motor acceleration feedback. Assignment notes velocity commands unsupported on physical SO-101 hardware; velocity is computed but velocity feedforward gains (Kv) are simulation-only tuning parameters.

### Timing & Duration

**Typical per-segment duration**: 2.0 seconds (specified in Assignment 9 test sequence).
- Can be parameterized per motion primitive (e.g., shorter for fast retract, longer for precise placement).
- **Control loop frequency**: 50 Hz (Δt = 0.02 s).
- **Timesteps per segment**: ~100 (2.0 s ÷ 0.02 s).

---

## 4. Pick-Place-Stack State Machine

### Single Pick-Place Cycle (Foundation)

**Waypoint sequence for grasping object at target_position = [x, y, z]**:

#### `pick_up_block(bus, block_position, move_duration=2.0)`

```
State 0: INITIALIZE
  current_config ← read_current_joint_state()

State 1: RAISE (move_duration seconds)
  block_raised ← block_position.copy()
  block_raised[z] += 0.03  # Approach offset 3 cm above block
  config_raised ← get_inverse_kinematics(block_raised, orientation=vertical_down)
  config_raised['gripper'] = 50  # Open
  move_to_pose_cubic(config_raised, duration=move_duration)

State 2: DESCEND (1.0 seconds fixed)
  config_contact ← get_inverse_kinematics(block_position, orientation=vertical_down)
  config_contact['gripper'] = 50  # Still open
  move_to_pose_cubic(config_contact, duration=1.0)

State 3: GRASP (1.0 seconds fixed)
  config_grasped ← config_contact.copy()
  config_grasped['gripper'] = 5  # Close gripper
  move_to_pose_cubic(config_grasped, duration=1.0)

State 4: RETRACT (1.0 seconds fixed)
  config_raised['gripper'] = 5  # Maintain grip
  move_to_pose_cubic(config_raised, duration=1.0)
  [Block is now held 3 cm above pickup location]
```

#### `place_block(bus, target_position, move_duration=2.0)`

```
State 0: INITIALIZE
  current_config ← read_current_joint_state()  [Block held in gripper]

State 1: RAISE (move_duration seconds)
  target_raised ← target_position.copy()
  target_raised[z] += 0.03  # Approach offset 3 cm above placement
  config_raised ← get_inverse_kinematics(target_raised, orientation=vertical_down)
  config_raised['gripper'] = 5  # Maintain grip
  move_to_pose_cubic(config_raised, duration=move_duration)

State 2: DESCEND (1.0 seconds fixed)
  config_contact ← get_inverse_kinematics(target_position, orientation=vertical_down)
  config_contact['gripper'] = 5  # Still gripping
  move_to_pose_cubic(config_contact, duration=1.0)

State 3: RELEASE (1.0 seconds fixed)
  config_open ← config_contact.copy()
  config_open['gripper'] = 50  # Open gripper
  move_to_pose_cubic(config_open, duration=1.0)

State 4: RETRACT (1.0 seconds fixed)
  config_raised['gripper'] = 50  # Maintain open
  move_to_pose_cubic(config_raised, duration=1.0)
  [Block is now placed at target_position; arm retreated safely]
```

### Assignment 8 Part 2: Stack 2 Blocks

```
INITIAL STATE:
  Block 1 location: [0.25, 0.1, 0.014] m (world)
  Block 2 location: [0.25, -0.1, 0.014] m (world)
  Desired stack location: [0.25, 0.1, ?] m (Block 2 placed on Block 1)

EXECUTION SEQUENCE:

Step 1: Move to Block 1
  move_to_pose_cubic(home_config, duration=2.0)

Step 2: Pick up Block 1
  pick_up_block(bus, [0.25, 0.1, 0.014], move_duration=2.0)
  [Block 1 held at [0.25, 0.1, 0.044] m = 0.014 + 0.03 offset]

Step 3: Place Block 1 at stack location (no movement needed; place at same xy)
  place_block(bus, [0.25, 0.1, 0.014], move_duration=2.0)
  [Block 1 lowered back to original location; gripper released]

Step 4: Move to Block 2
  move_to_pose_cubic(config_to_block2, duration=2.0)
  [Arm retracts to safe configuration, approaches Block 2]

Step 5: Pick up Block 2
  pick_up_block(bus, [0.25, -0.1, 0.014], move_duration=2.0)
  [Block 2 held at [0.25, -0.1, 0.044] m]

Step 6: Move to stack location with Block 2
  move_to_pose_cubic(config_approach_stack, duration=2.0)
  [Arm moves Block 2 from [0.25, -0.1, 0.044] to above [0.25, 0.1, ?]]

Step 7: Place Block 2 on top of Block 1
  place_block(bus, [0.25, 0.1, 0.043], move_duration=2.0)
  [Block 2 placed at z=0.043 m (on top of Block 1)]

Step 8: Retreat
  move_to_pose_cubic(home_config, duration=2.0)

SUCCESS CRITERION:
  ✓ Block 1 rests at [0.25, 0.1, 0.014] m
  ✓ Block 2 rests on top of Block 1 at [0.25, 0.1, 0.043] m
  ✓ Both blocks remain stable (no toppling)
```

**Key notes**:
- **Stacking height calculation**: z = 0.014 + (block_index - 1) × 0.0285
  - Block 1: z=0.014 m
  - Block 2: z=0.043 m (= 0.014 + 1×0.0285, adjusted to 0.043)
  - Block 3 (Part 3): z=0.071 m (= 0.014 + 2×0.0285, adjusted to 0.071)
- **Approach offset**: Always 0.03 m above target z.
- **Duration parameterization**: pick_up_block and place_block accept move_duration (default 2.0 s); intermediate descend/ascend/grasp/release phases use fixed 1.0 s.

### Assignment 8 Part 3: Stack 3 Blocks

```
Repeat Part 2 sequence, then:

Step 9: Move to Block 3 location [x, y, 0.014] (location TBD; course likely uses fixed or random)

Step 10: Pick up Block 3

Step 11: Move to stack location with Block 3

Step 12: Place Block 3 on top of Block 2
  place_block(bus, [0.25, 0.1, 0.071], move_duration=2.0)

Step 13: Retreat

SUCCESS CRITERION:
  ✓ All three blocks stacked vertically at [0.25, 0.1, {0.014, 0.043, 0.071}] m
  ✓ No toppling
```

---

## 5. Cubic Spline Pseudocode

### High-Level Control Loop

```python
def move_to_pose_cubic(bus, target_config, duration=2.0, dt=0.02):
    """
    Move from current joint state to target_config using cubic spline.

    Args:
        bus: FeetechMotorsBus communication handle
        target_config: dict {joint_name: angle_deg_or_normalized}
        duration: float, seconds to complete trajectory
        dt: float, control loop timestep (default 0.02 s = 50 Hz)
    """
    # Read current joint state
    current_config = bus.sync_read("Present_Position")  # Returns dict

    # Compute cubic spline coefficients for each joint
    coeffs = {}
    for joint_name in target_config:
        theta_0 = current_config[joint_name]
        theta_f = target_config[joint_name]
        delta_theta = theta_f - theta_0

        coeffs[joint_name] = {
            'a0': theta_0,
            'a1': 0,
            'a2': 3 * delta_theta / (duration ** 2),
            'a3': -2 * delta_theta / (duration ** 3),
        }

    # Execute trajectory over duration
    t_elapsed = 0.0
    while t_elapsed <= duration:
        # Clamp time to [0, duration]
        tlim = min(max(t_elapsed, 0), duration)

        # Evaluate cubic spline for each joint
        position_dict = {}
        velocity_dict = {}
        for joint_name, coeff in coeffs.items():
            a0, a1, a2, a3 = coeff['a0'], coeff['a1'], coeff['a2'], coeff['a3']

            pos = a0 + a1*tlim + a2*(tlim**2) + a3*(tlim**3)
            vel = a1 + 2*a2*tlim + 3*a3*(tlim**2)

            position_dict[joint_name] = pos
            velocity_dict[joint_name] = vel

        # Send command to hardware (velocity ignored on SO-101)
        bus.sync_write("Goal_Position", position_dict, normalize=True)

        # Advance time and sleep
        t_elapsed += dt
        time.sleep(dt)

def pick_up_block(bus, block_position, move_duration=2.0, hold_duration=1.0):
    """
    Pick up a block at given world position.

    Args:
        bus: FeetechMotorsBus handle
        block_position: [x, y, z] in world frame (meters)
        move_duration: seconds for raise/retract phases
        hold_duration: seconds for contact/grasp phases (fixed 1.0 s)
    """
    # Phase 1: Raise above block
    block_raised = block_position.copy()
    block_raised[2] += 0.03  # z += 0.03 m
    config_raised = get_inverse_kinematics(block_raised, orientation=[0, 0, -1])
    config_raised['gripper'] = 50
    move_to_pose_cubic(bus, config_raised, duration=move_duration)

    # Phase 2: Descend to contact
    config_contact = get_inverse_kinematics(block_position, orientation=[0, 0, -1])
    config_contact['gripper'] = 50
    move_to_pose_cubic(bus, config_contact, duration=hold_duration)

    # Phase 3: Close gripper
    config_grasped = config_contact.copy()
    config_grasped['gripper'] = 5
    move_to_pose_cubic(bus, config_grasped, duration=hold_duration)

    # Phase 4: Retract upward
    config_raised['gripper'] = 5
    move_to_pose_cubic(bus, config_raised, duration=move_duration)

def place_block(bus, target_position, move_duration=2.0, hold_duration=1.0):
    """
    Place held block at target world position.

    Args:
        bus: FeetechMotorsBus handle
        target_position: [x, y, z] in world frame (meters)
        move_duration: seconds for approach/retract phases
        hold_duration: seconds for contact/release phases (fixed 1.0 s)
    """
    # Phase 1: Approach above target
    target_raised = target_position.copy()
    target_raised[2] += 0.03  # z += 0.03 m
    config_raised = get_inverse_kinematics(target_raised, orientation=[0, 0, -1])
    config_raised['gripper'] = 5  # Maintain grip
    move_to_pose_cubic(bus, config_raised, duration=move_duration)

    # Phase 2: Descend to placement
    config_contact = get_inverse_kinematics(target_position, orientation=[0, 0, -1])
    config_contact['gripper'] = 5
    move_to_pose_cubic(bus, config_contact, duration=hold_duration)

    # Phase 3: Open gripper
    config_open = config_contact.copy()
    config_open['gripper'] = 50
    move_to_pose_cubic(bus, config_open, duration=hold_duration)

    # Phase 4: Retract
    config_raised['gripper'] = 50
    move_to_pose_cubic(bus, config_raised, duration=move_duration)

def stack_2_blocks(bus, block1_pos, block2_pos, stack_pos, move_duration=2.0):
    """
    Stack two blocks sequentially.

    Args:
        bus: FeetechMotorsBus handle
        block1_pos: [x, y, z] of first block
        block2_pos: [x, y, z] of second block
        stack_pos: [x, y, z] base location for stack (x, y fixed; z is floor)
    """
    # Pick Block 1
    pick_up_block(bus, block1_pos, move_duration=move_duration)

    # Place Block 1 at stack base (same xy, same z; establishes foundation)
    place_block(bus, [stack_pos[0], stack_pos[1], 0.014], move_duration=move_duration)

    # Move to Block 2
    move_to_pose_cubic(bus, get_home_config(), duration=move_duration)

    # Pick Block 2
    pick_up_block(bus, block2_pos, move_duration=move_duration)

    # Move to above Block 1
    block2_above_stack = [stack_pos[0], stack_pos[1], 0.014 + 0.03]
    config_approach = get_inverse_kinematics(block2_above_stack, orientation=[0, 0, -1])
    move_to_pose_cubic(bus, config_approach, duration=move_duration)

    # Place Block 2 on Block 1 (z=0.043)
    place_block(bus, [stack_pos[0], stack_pos[1], 0.043], move_duration=move_duration)

    # Return to home
    move_to_pose_cubic(bus, get_home_config(), duration=move_duration)
```

---

## 6. Success Criteria

### Simulation (MuJoCo)
- [ ] Three-waypoint cubic spline trajectory visualized in scene (cube markers at target positions).
- [ ] Tracking error (target vs. actual joint positions) logged as CSV/DataFrame.
- [ ] Comparison plot: cubic spline vs. linear interpolation showing smoother velocity profile.
- [ ] No joint limit violations during any segment.
- [ ] Gripper state transitions (open/close) occur at correct times.

### Hardware (Physical SO-101)
- [ ] Robot executes identical sequence on real arm.
- [ ] Video evidence uploaded (YouTube or local file).
- [ ] No gripper slip during grasp phases (visual confirmation via recorded trajectory).
- [ ] Block(s) remain stacked after sequence completion (visual inspection).

### LeRobot VLA Data Generation
- [ ] Episode trajectory recorded in LeRobot v3 format (HDF5 + camera frames).
- [ ] 3-camera views (top/wrist/front) synchronized at 30 fps (or hardware-supported rate).
- [ ] Gripper state, joint angles, end-effector pose logged per frame.
- [ ] Dataset uploaded to HuggingFace Hub with metadata (robot_type: so101, task: stack_2_blocks or stack_3_blocks).

### Validation Checklist
- [ ] All trajectories pass MuJoCo simulation without collision warnings.
- [ ] IK solutions computed for all target positions (no NaN detection).
- [ ] Cubic spline coefficients verified against boundary conditions (a₁=0, a₃=-2Δθ/T³).
- [ ] Control loop frequency steady at 50 Hz (no dropouts).
- [ ] Gripper torque/current monitoring (if available) shows safe grasp forces.

---

## 7. What the Demo Videos Likely Show

### Video 1: "ECE 4560 Lab 8 Demonstration" (youtu.be/wexwQoDvZR8)
- **Content** (inferred): Single-block or multi-block pick-and-place execution.
- **Camera view**: Likely top-down or angled side view.
- **Motion pattern**: Sequential approach → grasp → lift → placement → release → retreat.
- **Gripper state changes**: Visible open/close transitions.

### Video 2: "ECE 4560 Lab 8 - Three Blocks" (youtu.be/2RLV-14ctZQ)
- **Content** (from title): Three-block stacking demonstration.
- **Block count**: Exactly 3 (as title specifies).
- **Stacking order**: Likely Block 1 → Block 2 on Block 1 → Block 3 on Block 2.
- **Success metric**: All three blocks remain stacked (no toppling).
- **Motion**: Repeated pick-place cycles with z-height adjustments.

**Inaccessible details** (page content blocked; direct video viewing required):
- Exact block starting/ending positions in world frame.
- Whether blocks start at identical xy locations (parallel stacking) or scattered.
- Precise gripper behavior (grip strength, hold duration).
- Whether trajectory smoothing (cubic spline vs. linear) is visually apparent.

---

## 8. Unknowns & Gaps

### Task Definition
1. **Block starting locations (Part 2, Block 2; Part 3, Block 3)**:
   - Spec lists [0.25, 0.1] and [0.25, -0.1] for two blocks in Part 2.
   - Block 3 location in Part 3 unknown (likely [0.25, 0.X] where X TBD or random).
   - **Resolution needed**: Confirm from assignment page §Part 3 or video.

2. **Stack base location**:
   - Inferred as [0.25, 0.1, 0.014] (same xy as Block 1 initial position).
   - **To verify**: Whether placement is at Block 1's original location or a different target xy.

3. **Gripper force/closure time**:
   - Course specifies gripper_closed = 5, gripper_open = 50 (normalized 0–100 range).
   - No torque limit, hold duration, or slipping tolerance specified.
   - **Action**: Assume motor PID handles smooth closure; no additional force control needed.

### Kinematics
1. **DH parameters or link lengths**:
   - Assignment pages reference "from Assignment 6 FK" but do not list explicit L1, L2 lengths.
   - Forward kinematics uses Product of Exponentials; exact screw axes not enumerated in brief.
   - **Resolution**: Extract from existing Course material or AGENTS.md `follow_target_so101.py` implementation.

2. **Orientation representation for IK**:
   - Course specifies "vertical approach" (gripper pointing down = [0, 0, −1] in tool frame).
   - No Euler angle convention or quaternion handling specified.
   - **Assumption**: Tool frame Z-axis aligned to world Z (pitch/roll = 0); IK returns wrist roll θ₅ as free parameter.

### Cubic Splines
1. **Via-point continuity**:
   - Brief confirms NO intermediate via-points; each segment is independent (start→goal only).
   - Discontinuities in acceleration at segment boundaries are acceptable.
   - **Clarification**: Multi-segment trajectories (e.g., raise→contact→grasp→retract) use four separate cubic splines, not a single spline with four via-points.

2. **Optional velocity feedforward**:
   - Assignment 9 mentions "optional velocity feedforward via Kv=10 gains in MuJoCo actuators."
   - Hardware SO-101 does not support velocity commands (position-only).
   - **Implication**: Velocity computed but discarded in hardware motion commands; MuJoCo only uses for simulation PD control.

3. **Time-optimal trajectory**:
   - Assignment does not require time-optimal or acceleration-limited splines (e.g., S-curve).
   - Fixed 2.0 s per segment assumed safe and smooth; no jerk constraints mentioned.
   - **Simplification**: Cubic splines with zero-velocity boundaries sufficient; no need for higher-order polynomials.

### Hardware Path
1. **Motor communication stack**:
   - AGENTS.md references `FeetechMotorsBus` API and `docker/lerobot_keyboard_stdin.py`.
   - Assignment 8 likely uses LeRobot's `lerobot.robot.so101_robot.SO101Robot` or similar abstraction.
   - **Assumption**: `bus.sync_read()` and `bus.sync_write()` provide joint I/O; no low-level USB or serial detail required for brief.

2. **Calibration offsets**:
   - Assignment 8 mentions "per-joint angle corrections" and "z-platform height -0.015m" offsets.
   - No explicit calibration procedure detailed in assignment pages.
   - **Action**: Implement calibration step (e.g., home configuration, joint offset storage) as part of initialization.

### Data Generation for VLA Training
1. **Camera setup (3-view)**:
   - AGENTS.md mentions top/wrist/front cameras with rerun visualization.
   - Resolution, frame rate, and synchronization strategy not specified in Assignment 8 pages.
   - **Assumption**: LeRobot v3 standard: top/wrist/front at 30 fps, H.264 encoded, 0.5 MP resolution (typical).

2. **Action/observation format**:
   - Assignment 8 does not specify VLA training data schema.
   - LeRobot typically records: joint positions, gripper state, camera frames, proprioceptive state.
   - **Inference**: Use LeRobot's `LeRobotDatasetWriter` (v3 format) with SO-101 config (6-DOF: 5 arm + 1 gripper).

3. **Episode metadata**:
   - No task_name, demo_type (collected vs. teleop), or episode ID scheme specified.
   - **Convention**: Use `{task}_{part}_{block_count}_{timestamp}` naming (e.g., `assignment8_part2_2blocks_2026_06_18_12_34_56.hdf5`).

---

## 9. Implementation Roadmap

### Phase 1: Kinematics Validation (Sim)
- [ ] Import/implement `get_inverse_kinematics(target_pos, target_orient)` from Assignment 7 or Isaac solver.
- [ ] Validate IK against test poses in MuJoCo (verify zero FK error).
- [ ] Test workspace coverage at z ∈ {0.014, 0.043, 0.071} m.

### Phase 2: Cubic Spline Trajectory (Sim + Hardware)
- [ ] Implement `move_to_pose_cubic()` with coefficient computation and time-bounded evaluation.
- [ ] Test cubic interpolation accuracy (error vs. linear baseline).
- [ ] Verify 50 Hz control loop timing stability.

### Phase 3: Pick-Place Primitives (Sim + Hardware)
- [ ] Implement `pick_up_block()` and `place_block()` using cubic splines for raise/descend/retract.
- [ ] Test gripper open/close state transitions.
- [ ] Validate approach offset (0.03 m clearance) prevents collision with blocks.

### Phase 4: Multi-Block Stacking (Sim + Hardware)
- [ ] Implement `stack_2_blocks()` and `stack_3_blocks()` state machines.
- [ ] Test on MuJoCo with random block starting positions.
- [ ] Deploy to hardware; capture video evidence.

### Phase 5: LeRobot Data Generation & Upload
- [ ] Integrate LeRobot v3 `LeRobotDatasetWriter` for episode recording.
- [ ] Sync 3-camera views (top/wrist/front) with motor telemetry.
- [ ] Upload dataset to HuggingFace Hub with task metadata.

---

## 10. References & Supporting Documents

- **AGENTS.md**: §Docker 컨테이너 구조 (실기기 경로), §시뮬레이션 구조 (Isaac Lab 경로)
- **Assignment 6 (Forward Kinematics)**: Product of Exponentials, helper functions get_gxx()
- **Assignment 7 (Inverse Kinematics)**: Geometric decoupled solution, law of cosines, NaN detection
- **Assignment 9 (Cubic Splines)**: Boundary conditions, per-joint coefficient computation, time-bounded evaluation
- **ECE 4560 course website**: https://maegantucker.com/ECE4560/
- **LeRobot 0.4.4 documentation**: https://github.com/huggingface/lerobot (teleop, dataset formats)

---

**Document Status**: Engineering brief synthesized from ECE 4560 assignment pages and demo video metadata. Ready for implementation.

**Last Updated**: 2026-06-18

**Prepared for**: SO-101 hardware + LeRobot VLA pipeline integration.
