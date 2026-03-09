import pyampl
import yaml
import ampl
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from reachability_visualizor import visualize_tote_and_tcp

CONFIG_FILE = "reachability/search_config.yaml"

class ReachabilitySearch:
    def __init__(self, config):
        self.parse_config(config)
        self.agent = None
        
    def parse_config(self, cfg:dict):
        self.env_cfg = cfg['environment']
        self.robot_cfg = self.env_cfg['robot']
        self.gripper_cfg = self.env_cfg['gripper']
        self.tote_cfg = self.env_cfg['tote']
        self.coll_cfg = cfg['collision_check']
        self.ik_cfg = cfg['ik']
        
    def setup_agent(self):
        arm_config = pyampl.create_default_arm_config(self.robot_cfg['arm_name'])
        self.agent = pyampl.AgentArm(arm_config.name, arm_config.dim, arm_config)
        self.agent.fk_rwt = self.agent.state_ref
        
    def setup_wall(self):
        self.agent.wall = np.array(self.robot_cfg['wall']).astype(pyampl.DTypeFloat)
        
    def load_collision_object_convex(self, mesh):
        return pyampl.CollisionObjectConvex(mesh)

    def load_mesh(self, path):
        return ampl.read_trimesh(path)
    
    def load_tcp_standard_transform(self):
        self.R_base_tcp_standard = np.array(self.gripper_cfg['standard_tcp_rotation'], dtype=np.float64)
    
    def create_collision_scene(self):
        self.collision_scene = pyampl.CollisionScene()
    
    def insert_convex(self, name, mesh, create_pcd, pcd_idx):
        self.collision_scene.insert_convex(name, mesh, create_pcd, pcd_idx)
    
    def update_collision_scene(self, obj_poses: Dict[str, np.ndarray]):
        for name,rwt in obj_poses.items():    
            if rwt.shape == (4,4):
                rwt = ampl.tf44_to_qt7(rwt)
            else:
                raise NotImplementedError("Only 4x4 transform matrices are supported for object poses.")
            self.collision_scene.update_pose(name, rwt)
            self.collision_scene.update_poses_pcd_from_convex()
            self.collision_scene.enable_collision(name)
        pcd_tmp = self.collision_scene.get_pointcloud(list(obj_poses.keys()))        
        self.collision_scene.update_df_from_pcd(pcd_tmp)  
    
    def setup_environment(self):
        self.setup_agent()
        self.setup_wall()
        self.create_collision_scene()
        self.load_tcp_standard_transform()
        
        self.tote_mesh = self.load_mesh(self.tote_cfg['mesh'])
        self.insert_convex('tote', self.tote_mesh, self.coll_cfg['create_pcd'], self.coll_cfg['pcd_dx'])
        
        self.gripper_mesh = self.load_mesh(self.gripper_cfg['mesh'])
        self.cvh_gripper = self.load_collision_object_convex(self.gripper_mesh)
        
    def load_base_tote_pose(self):
        base_pos = np.array(self.tote_cfg['base_position'], dtype=np.float32)
        return base_pos
    
    def tcp_range_to_lists(self):
        '''Resolve tcp position and rotation from range to discrete list'''
        tcp_range = self.tote_cfg['tcp_range']
        
        def resolve_val(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                return eval(val.replace('pi', 'np.pi'), {"np": np})
            return val

        def get_range(r):
            start = resolve_val(r[0])
            stop = resolve_val(r[1])
            step = resolve_val(r[2])
            if start == stop:
                return np.array([start])
            return np.arange(start, stop, step)

        x_vals = get_range(tcp_range['x'])
        y_vals = get_range(tcp_range['y'])
        z_vals = get_range(tcp_range['z'])
        
        if tcp_range.get('rotation_in_range', False):
            rx_vals = get_range(tcp_range['rx'])
            ry_vals = get_range(tcp_range['ry'])
            rz_vals = get_range(tcp_range['rz'])
        else:
            rx_vals = np.array([0.0])
            ry_vals = np.array([0.0])
            rz_vals = np.array([0.0])
            
        return x_vals, y_vals, z_vals, rx_vals, ry_vals, rz_vals


    def sample_tcp_pose_by_range(self):
        '''Resolve tcp from range to transform matrix list'''
        x_vals, y_vals, z_vals, rx_vals, ry_vals, rz_vals = self.tcp_range_to_lists()
        
        tcp_poses = []
        tcp_tf_mats = []
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    for rx in rx_vals:
                        for ry in ry_vals:
                            for rz in rz_vals:
                                tcp_poses.append([x, y, z, rx, ry, rz])
                                R_in_base = ampl.so3_upexp(np.array([0,0,rz])) @ \
                                       ampl.so3_upexp(np.array([0,ry,0])) @ \
                                       ampl.so3_upexp(np.array([rx,0,0]))
                                tf = np.eye(4,dtype=np.float64)
                                tf[:3,:3] = R_in_base
                                tf[:3,3] = [x,y,z]
                                tcp_tf_mats.append(tf)
                                
        sample_shape = (len(x_vals), len(y_vals), len(z_vals), len(rx_vals), len(ry_vals), len(rz_vals))
        return tcp_poses, tcp_tf_mats, sample_shape
    
    def transform_tcp_by_base(self, base_pos:np.ndarray, tcp_tf_mats:List[np.ndarray])->List[np.ndarray]:
        '''Update tcp poses by tote base position keep the order'''
        transformed_tfs = []
        for tf_in_tote in tcp_tf_mats:
            tf_tcp = base_pos @ tf_in_tote
            tf_tcp[:3,:3] = tf_tcp[:3,:3] @ self.R_base_tcp_standard
            transformed_tfs.append(tf_tcp)
        return transformed_tfs
    
    def compute_reachability_itr(self, tcp_tf_mats:List[np.ndarray])->np.ndarray:
        '''
        Compute reachability for each tcp_tf_mats return
        
        Return:
            List of boolean mask indicating reachability for each tcp pose
        '''
        rwts_tcp = [ampl.tf44_to_qt7(tf) for tf in tcp_tf_mats]
        mask_tool = self.collision_scene.collision_free_external(self.cvh_gripper, rwts_tcp)
        
        reachability_mask = np.zeros(len(tcp_tf_mats), dtype=bool)
        # Use initial state from robot config or agent state_ref
        state = np.array(self.agent.state_ref.tolist(), dtype=np.float64)
        
        for i, tf_tcp in enumerate(tcp_tf_mats):
            if not mask_tool[i]:
                continue
            
            # Use ik_redundant_wall_torso_df as in scan_v2
            qs_ik = self.agent.ik_redundant_wall_torso_df(
                tf_tcp.astype(np.float64), 
                state_ref=state, 
                nb_redundant_search=self.ik_cfg.get('nb_redundant_search', 512), 
                env=self.collision_scene
            )
            
            if len(qs_ik) > 0:
                reachability_mask[i] = True
                # Move to the found state for next iteration to encourage continuity
                state = qs_ik[0].copy()
        
        return reachability_mask
        
    def resolve_reachability_by_transition(self, reachability_mask:np.ndarray, sample_tcp_shape:Tuple)->np.ndarray:
        '''Sum reachability mask by transition (translation)'''
        reshaped_mask = reachability_mask.reshape(sample_tcp_shape)
        # Sum over the rotation axes (rx, ry, rz) which are indices 3, 4, 5
        summed_reachability = np.sum(reshaped_mask, axis=(3, 4, 5))
        return summed_reachability
    
    
    def search_reachability(self, base_pos:np.ndarray, itr_limit:int=20):
        '''
        Search for best reachability tote position untill reach itr_limit or local_minimum reached.
        At each iteration, compute limited reachability points for all direction with resolution
        (front, back, left, right, up, down), move tote base to the best direction if found.
        Searched location is saved in cache.
        '''
        resolution_x, resolution_y, resolution_z = self.tote_cfg['resolution'][0]
        search_cache = dict()
        
        sample_tcp_poses, sample_tcp_tf_mats, sample_tcp_shape = self.sample_tcp_pose_by_range()
        print(f"Total sampled tcp poses: {len(sample_tcp_poses)}")

        def evaluate_position(pos):
            # Cache key using rounded translation
            pos_key = tuple(np.round(pos[:3, 3], 4))
            if pos_key in search_cache:
                return search_cache[pos_key]
            
            # Update collision scene and compute reachability
            self.update_collision_scene({"tote": pos})
            local_tcp_samples = self.transform_tcp_by_base(pos, sample_tcp_tf_mats)
            mask = self.compute_reachability_itr(local_tcp_samples)
            
            # Score is total number of valid reachability samples
            score = np.sum(mask)
            
            # Stats for printing
            resolved_ts = self.resolve_reachability_by_transition(mask, sample_tcp_shape)
            scores_grid = resolved_ts.astype(np.float32) / np.prod(sample_tcp_shape[3:])
            
            result = {
                'reachable_position': np.sum(scores_grid > 0),
                'reachable_sample': np.sum(scores_grid),
                'mask': mask,
                'scores_grid': scores_grid,
                'local_tcp_samples': local_tcp_samples
            }
            search_cache[pos_key] = result
            return result

        current_pos = base_pos.copy()
        directions = [
            np.array([resolution_x, 0, 0]),  # front
            np.array([-resolution_x, 0, 0]), # back
            np.array([0, resolution_y, 0]),  # left
            np.array([0, -resolution_y, 0]), # right
            np.array([0, 0, resolution_z]),  # up
            np.array([0, 0, -resolution_z]), # down
        ]

        for i in range(itr_limit):
            curr_res = evaluate_position(current_pos)
            print(f"\n--- Iteration {i} ---")
            
            scores = curr_res['scores_grid']
            print(f"Failed points {np.sum(scores == 0)} Limited reachable points {np.sum((scores > 0) & (scores < 1))} Fully reachable points {np.sum(scores == 1)}")

            best_neighbor_pos = None
            best_neighbor_reachable_position = curr_res['reachable_position']
            best_neighbor_reachable_sample = curr_res['reachable_sample']

            for d in directions:
                neighbor_pos = current_pos.copy()
                neighbor_pos[:3, 3] += d
                neighbor_res = evaluate_position(neighbor_pos)
                
                if neighbor_res['reachable_position'] > best_neighbor_reachable_position:
                    best_neighbor_reachable_position = neighbor_res['reachable_position']
                    best_neighbor_reachable_sample = neighbor_res['reachable_sample']
                    best_neighbor_pos = neighbor_pos
                elif neighbor_res['reachable_position'] == best_neighbor_reachable_position:
                    if neighbor_res['reachable_sample'] > best_neighbor_reachable_sample:
                        best_neighbor_reachable_sample = neighbor_res['reachable_sample']
                        best_neighbor_pos = neighbor_pos
            
            if best_neighbor_pos is None:
                print("Local optimum reached.")
                break
            
            current_pos = best_neighbor_pos
            print(f"Improved score to {best_neighbor_reachable_position}. Moving to new position.")

        print("\nSearch finished.")
        final_res = evaluate_position(current_pos)
        print(f"Final Best Position: {current_pos[:3, 3]} (Score: {final_res['reachable_position']})")
        
        # Save results
        # save_data = {
        #     'best_pos': current_pos,
        #     'final_score': final_res['score'],
        #     'search_cache': search_cache,
        #     'sample_tcp_shape': sample_tcp_shape
        # }
        # np.save("reachability/reachability_results.npy", save_data)
        # print("Results saved to reachability/reachability_results.npy")

        # For Visualize (Final result)
        n_rot = np.prod(sample_tcp_shape[3:])
        world_grid_points = np.array([tf[:3, 3] for tf in final_res['local_tcp_samples'][::n_rot]])   
        sample_rotations = final_res['local_tcp_samples'][:n_rot]
        visualize_tote_and_tcp(self.tote_mesh, self.gripper_mesh, world_grid_points, final_res['scores_grid'].flatten(), current_pos, sample_rotations)
        
        return current_pos, final_res
    

    
    
    
def main():
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    reachability_search = ReachabilitySearch(config)
    reachability_search.setup_environment()
    print(f"Environment setup completed.")
    
    tote_base_pos = reachability_search.load_base_tote_pose()
    reachability_search.update_collision_scene({"tote": tote_base_pos})
    reachability_search.search_reachability(tote_base_pos)


        
if __name__ == "__main__":
    main()
