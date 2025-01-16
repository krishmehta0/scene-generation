import pybullet as p
import pybullet_data
import time
import numpy as np
import yaml
from pathlib import Path

class SpeakerSimulation:
    def __init__(self, model_path, config_path):
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        p.setGravity(0, 0, -9.81)
        
        # Enable real-time simulation
        p.setRealTimeSimulation(1)
        
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load KUKA arm
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0.5, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.end_effector_index = 6
        
        # Enable force/torque sensors
        for i in range(self.num_joints):
            p.enableJointForceTorqueSensor(self.robot_id, i)
        
        # Load speaker
        self.speaker_id = self.load_speaker(model_path)
        self.apply_physical_properties()
        
        # Reset arm to starting position
        self.reset_arm_position()
        
        # Step simulation a bit to let everything settle
        for _ in range(100):
            p.stepSimulation()

    def load_speaker(self, model_path):
        """Load the speaker model and create a collision shape"""
        dims = self.config['physical_properties']['dimensions']
        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                dims['width']/2,  # Divide by 2 as these are half-extents
                dims['depth']/2,
                dims['height']/2
            ]
        )
        
        # Create visual shape
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[
                dims['width']/2,
                dims['depth']/2,
                dims['height']/2
            ],
            rgbaColor=[0.2, 0.2, 0.2, 1]
        )
        
        # Create multi body (positioned with offset)
        speaker_id = p.createMultiBody(
            baseMass=self.config['physical_properties']['mass'],
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, dims['height']/2],  # Place on ground, centered at half height
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        return speaker_id

    def reset_arm_position(self):
        """Reset the arm to a default position"""
        rest_poses = [0, 0.1, 0, -1.5, 0, 1.0, 0]
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, rest_poses[i])
            # Set joint damping
            p.changeDynamics(self.robot_id, i, 
                           jointDamping=1.0,
                           linearDamping=0.1,
                           angularDamping=0.1)

    def move_arm(self, target_pos, target_orn, speed_factor=1.0):
        """Move the arm with smoother motion control"""
        # Calculate IK
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Get current joint positions
        current_joints = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        
        # Interpolate between current and target positions
        steps = 50  # Increase for smoother motion
        for step in range(steps):
            # Interpolation factor (0 to 1)
            t = (step + 1) / steps
            
            # Apply smooth acceleration/deceleration
            t = 0.5 * (1 - np.cos(t * np.pi))
            
            # Interpolate joint positions
            interpolated_poses = [
                current_joints[i] + t * (joint_poses[i] - current_joints[i])
                for i in range(self.num_joints)
            ]
            
            # Apply positions to joints with proper control parameters
            for i in range(self.num_joints):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=interpolated_poses[i],
                    targetVelocity=0,
                    force=100.0,  # Reduced force for gentler motion
                    positionGain=0.1,  # Lower gains for smoother motion
                    velocityGain=0.5,
                    maxVelocity=0.5 * speed_factor
                )
            
            p.stepSimulation()
            time.sleep(0.01)
        
        # Final adjustment to ensure target is reached
        for _ in range(20):
            p.stepSimulation()
            time.sleep(0.01)

    def pick_up_speaker(self):
        """Execute picking up sequence with improved stability"""
        speaker_pos, _ = p.getBasePositionAndOrientation(self.speaker_id)
        
        # Pre-grasp position (higher above speaker)
        pre_grasp_pos = [
            speaker_pos[0],
            speaker_pos[1],
            speaker_pos[2] + 0.3
        ]
        self.move_arm(pre_grasp_pos, p.getQuaternionFromEuler([0, -np.pi/2, 0]))
        
        # Approach position
        approach_pos = [
            speaker_pos[0],
            speaker_pos[1],
            speaker_pos[2] + 0.15
        ]
        self.move_arm(approach_pos, p.getQuaternionFromEuler([0, -np.pi/2, 0]))
        
        # Grasp position
        grasp_pos = [
            speaker_pos[0],
            speaker_pos[1],
            speaker_pos[2] + 0.1
        ]
        self.move_arm(grasp_pos, p.getQuaternionFromEuler([0, -np.pi/2, 0]))
        
        # Create a constraint to simulate grasping
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.end_effector_index,
            childBodyUniqueId=self.speaker_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        
        # Lift position
        lift_pos = [
            speaker_pos[0],
            speaker_pos[1],
            speaker_pos[2] + 0.3
        ]
        self.move_arm(lift_pos, p.getQuaternionFromEuler([0, -np.pi/2, 0]))

    def run_simulation(self, duration=10.0):
        """Run the simulation for specified duration"""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            p.stepSimulation()
            time.sleep(1./240.)
    
    def cleanup(self):
        """Cleanup the simulation"""
        p.disconnect()

    def apply_physical_properties(self):
        """Apply physical properties from config to the speaker"""
        # Set dynamics properties
        p.changeDynamics(
            self.speaker_id,
            -1,  # base link
            mass=self.config['physical_properties']['mass'],
            lateralFriction=self.config['material_properties']['surface_properties']['friction']['dynamic'],
            spinningFriction=0.1,
            rollingFriction=0.1,
            restitution=self.config['material_properties']['surface_properties']['restitution'],
            linearDamping=0.04,
            angularDamping=0.04,
            contactStiffness=1e5,
            contactDamping=2000
        )
        
        # Set inertia
        inertia = self.config['physical_properties']['inertia_tensor']
        p.changeDynamics(
            self.speaker_id,
            -1,
            localInertiaDiagonal=[inertia['ixx'], inertia['iyy'], inertia['izz']]
        )

    def pick_and_place_speaker(self):
        """Execute complete pick and place sequence with data collection"""
        trajectory_data = {
            'timestamps': [],
            'joint_positions': [],
            'joint_velocities': [],
            'joint_torques': [],
            'end_effector_poses': [],
            'gripper_states': []
        }
        
        start_time = time.time()
        speaker_pos, _ = p.getBasePositionAndOrientation(self.speaker_id)
        
        # Define place position (original position with slight offset)
        place_pos = [
            speaker_pos[0] + 0.3,  # Place it 30cm to the right
            speaker_pos[1],
            speaker_pos[2]
        ]
        
        # Picking sequence with speed control
        positions = {
            'pre_grasp': {'pos': [speaker_pos[0], speaker_pos[1], speaker_pos[2] + 0.3], 'speed': 1.0},
            'approach': {'pos': [speaker_pos[0], speaker_pos[1], speaker_pos[2] + 0.15], 'speed': 0.5},
            'grasp': {'pos': [speaker_pos[0], speaker_pos[1], speaker_pos[2] + 0.1], 'speed': 0.3},
            'lift': {'pos': [speaker_pos[0], speaker_pos[1], speaker_pos[2] + 0.3], 'speed': 0.5},
            'move_to_place': {'pos': [place_pos[0], place_pos[1], place_pos[2] + 0.3], 'speed': 0.7},
            'pre_place': {'pos': [place_pos[0], place_pos[1], place_pos[2] + 0.15], 'speed': 0.4},
            'place': {'pos': [place_pos[0], place_pos[1], place_pos[2] + 0.05], 'speed': 0.2},
            'retreat': {'pos': [place_pos[0], place_pos[1], place_pos[2] + 0.3], 'speed': 0.5}
        }
        
        # Execute sequence and collect data
        for phase, params in positions.items():
            print(f"\nExecuting {phase} phase...")
            self.move_arm(
                params['pos'], 
                p.getQuaternionFromEuler([0, -np.pi/2, 0]),
                speed_factor=params['speed']
            )
            
            # Create/remove constraint at appropriate phases
            if phase == 'grasp':
                time.sleep(0.5)  # Pause before grasping
                self.grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=self.end_effector_index,
                    childBodyUniqueId=self.speaker_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0]
                )
                print("Grasping speaker...")
                time.sleep(0.5)  # Pause after grasping
            elif phase == 'place':
                time.sleep(0.5)  # Pause before releasing
                p.removeConstraint(self.grasp_constraint)
                print("Releasing speaker...")
                time.sleep(0.5)  # Pause after releasing
            
            # Collect data
            self._collect_state_data(trajectory_data, time.time() - start_time)
        
        # Save trajectory data
        self._save_trajectory_data(trajectory_data)
        
        return trajectory_data

    def _collect_state_data(self, data_dict, timestamp):
        """Collect current state data"""
        # Get joint states
        joint_states = [p.getJointState(self.robot_id, i) for i in range(self.num_joints)]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        
        # Get end effector pose
        ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_position = ee_state[0]
        ee_orientation = ee_state[1]
        
        # Append to trajectory data
        data_dict['timestamps'].append(timestamp)
        data_dict['joint_positions'].append(joint_positions)
        data_dict['joint_velocities'].append(joint_velocities)
        data_dict['joint_torques'].append(joint_torques)
        data_dict['end_effector_poses'].append(ee_position + ee_orientation)
        data_dict['gripper_states'].append(1 if hasattr(self, 'grasp_constraint') else 0)

    def _save_trajectory_data(self, trajectory_data):
        """Save trajectory data to file"""
        import json
        from datetime import datetime
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'robot_type': 'KUKA_IIWA',
                'end_effector': 'parallel_gripper',
                'object': 'speaker',
                'task': 'pick_and_place'
            },
            'trajectory': {
                k: [list(map(float, item)) if isinstance(item, (list, tuple)) else float(item)
                    for item in v]
                for k, v in trajectory_data.items()
            }
        }
        
        # Save to file
        filename = f"robot_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"\nTrajectory data saved to {filename}")

def main():
    # Setup paths
    model_path = "speaker.obj"
    config_path = "speaker_config.yaml"
    
    # Create simulation
    sim = SpeakerSimulation(model_path, config_path)
    
    # Wait for scene to settle
    time.sleep(1)
    
    try:
        # Execute complete pick and place sequence
        trajectory_data = sim.pick_and_place_speaker()
        
        # Print summary statistics
        print("\nTrajectory Summary:")
        print(f"Total duration: {trajectory_data['timestamps'][-1]:.2f} seconds")
        print(f"Number of waypoints: {len(trajectory_data['timestamps'])}")
        print("\nJoint Statistics:")
        for i in range(len(trajectory_data['joint_positions'][0])):
            positions = [pos[i] for pos in trajectory_data['joint_positions']]
            torques = [torq[i] for torq in trajectory_data['joint_torques']]
            print(f"Joint {i}:")
            print(f"  Position range: [{min(positions):.2f}, {max(positions):.2f}] rad")
            print(f"  Max torque: {max(abs(min(torques)), abs(max(torques))):.2f} Nm")
        
        # Run simulation for a few more seconds
        sim.run_simulation(5.0)
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Cleanup
        sim.cleanup()

if __name__ == "__main__":
    main()