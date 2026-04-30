import sys
import os
import pickle
import yaml
import numpy as np
import torch
import pygame
from typing import Optional, Dict, Any, Tuple

from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as BaseOvercookedEnv
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld as BaseOvercookedGridworld
# Prefer overcooked_new renderer (compatible with new map object model), fall back to old version on failure
try:
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
except Exception:
    from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState, PlayerState, OvercookedState
)

from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.runner.shared.base_runner import make_trainer_policy_cls
from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy, R_MAPPOPolicy_adaptive


def infer_policy_config_from_checkpoint(checkpoint_path: str, layout_name: str):
    """Infer policy config from .pt weight file, generate [args, obs_space, share_obs_space, act_space]

    When policy_pool only has .pt weight files without policy_config.pkl,
    infer network architecture parameters by analyzing state_dict tensor shapes,
    and query the map to get observation shape, constructing the complete config needed for policy initialization.
    """
    import types
    from gym import spaces

    sd = torch.load(checkpoint_path, map_location="cpu")

    # 1. Check if adaptive policy (with temporal state encoder)
    is_adaptive = "base.ts_encoder.rnn.weight_ih_l0" in sd

    # 2. Infer hidden_size
    # GRU weight_ih_l0 shape = (3 * hidden_size, input_size)
    rnn_ih = sd["rnn.rnn.weight_ih_l0"]
    hidden_size = rnn_ih.shape[0] // 3

    # 3. Infer temporal_state_num (adaptive only)
    temporal_state_num = 0
    if is_adaptive:
        ts_ih = sd["base.ts_encoder.rnn.weight_ih_l0"]
        temporal_state_num = ts_ih.shape[1]

    # 4. Infer observation spatial size
    # Last CNN layer flattens then connects to Linear, weight shape = (out_features, in_features)
    # in_features = last_conv_channels * H * W
    cnn_flat = sd["base.cnn.7.weight"].shape[1]
    cnn_out_ch = sd["base.cnn.4.weight"].shape[0]
    spatial_pixels = cnn_flat // cnn_out_ch

    # 5. Query actual map lossless_state_encoding shape
    try:
        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
            OvercookedGridworld as NewGridworld,
        )
        mdp = NewGridworld.from_layout_name(layout_name)
    except Exception:
        mdp = BaseOvercookedGridworld.from_layout_name(layout_name)

    state = mdp.get_standard_start_state()
    obs_sample = mdp.lossless_state_encoding(state)[0]
    obs_shape = obs_sample.shape  # (H, W, C)

    # Verify: if checkpoint inferred spatial_pixels mismatches map, print warning
    h, w = obs_shape[0], obs_shape[1]
    if h * w != spatial_pixels:
        # Try swapping H,W
        if w * h == spatial_pixels:
            pass  # Product same, no action needed
        else:
            print(f"   [WARN] Observation size mismatch: checkpoint expects flattened={spatial_pixels}, actual map {h}x{w}={h*w}")

    # 6. Infer action count
    n_actions = sd["act.action_out.linear.weight"].shape[0]

    # 7. Build args namespace
    args = types.SimpleNamespace()
    args.lr = 5e-4
    args.critic_lr = 5e-4
    args.opti_eps = 1e-5
    args.weight_decay = 0.0
    args.hidden_size = hidden_size
    args.gain = 0.01
    args.use_orthogonal = True
    args.activation_id = 1  # ReLU
    args.use_policy_active_masks = True
    args.use_naive_recurrent_policy = False
    args.use_recurrent_policy = True
    args.use_influence_policy = False
    args.influence_layer_N = 2
    args.use_policy_vhead = False
    args.use_popart = False
    args.use_maxpool2d = False
    args.recurrent_N = 1
    args.cnn_layers_params = "32,3,1,1 64,3,1,1 32,3,1,1"
    args.input_temporal_state = is_adaptive
    args.temporal_state_num = temporal_state_num
    args.data_parallel = False
    args.layer_after_N = 0
    args.algorithm_name = "inferred"
    args.use_cat_self = True
    args.use_attn_internal = False
    args.num_v_out = 1

    # 8. Build gym spaces
    # lossless_state_encoding values are usually 0/1 or 0/255, use [0, 255] range here
    obs_space = spaces.Box(
        low=0.0, high=255.0, shape=obs_shape, dtype=np.float32
    )
    share_obs_space = obs_space
    act_space = spaces.Discrete(n_actions)

    return [args, obs_space, share_obs_space, act_space]


class PolicyLoader:
    """Policy loader - supports loading trained models"""
    
    def __init__(self, policy_pool_path: str, layout_name: str):
        self.policy_pool_path = policy_pool_path
        self.layout_name = layout_name
        # Also record zsceval/policy_pool as fallback
        project_root = os.path.join(os.path.dirname(__file__), '..')
        self.zsceval_pool_path = os.path.join(project_root, "zsceval", "policy_pool")
        # Try loading map from overcooked_new (new maps have tomatoes)
        try:
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld as NewGridworld
            self.mdp = NewGridworld.from_layout_name(layout_name)
        except:
            self.mdp = BaseOvercookedGridworld.from_layout_name(layout_name)
        
    def _create_policy(self, policy_config):
        """Create correct policy class from config (supports adaptive/BACH)"""
        policy_args = policy_config[0]
        has_temporal = getattr(policy_args, "input_temporal_state", False)
        device = torch.device("cpu")
        
        if has_temporal:
            # Auto-infer temporal_state_num (if not set in policy config)
            temporal_state_num = getattr(policy_args, "temporal_state_num", None)
            if temporal_state_num is None or temporal_state_num == 0:
                try:
                    from zsceval.cli.config.env_config import get_temporal_state_num
                    temporal_state_num = get_temporal_state_num(self.layout_name)
                    print(f"   Auto-inferred temporal_state_num: {temporal_state_num}")
                except Exception as e:
                    print(f"   [WARN] Cannot auto-infer temporal_state_num: {e}")
                    temporal_state_num = 0
            policy_args.temporal_state_num = temporal_state_num
            policy = R_MAPPOPolicy_adaptive(*policy_config, device=device)
        else:
            # Directly instantiate standard R_MAPPOPolicy, avoid dependency on potentially missing r_mappo.py / population modules
            from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
            policy = R_MAPPOPolicy(*policy_config, device=device)
        
        policy.prep_rollout()
        return policy

    def _find_population_yaml(self, algo: str, stage: str) -> Tuple[Optional[str], Optional[str]]:
        """Find population.yml or any .yml config file
        
        Returns:
            (yaml_path, base_path_for_model)  
        """
        # BACH may correspond to ltl directory in policy_pool
        pool_algos = [algo]
        if algo.lower() == "bach":
            pool_algos.append("ltl")
        
        search_roots = [
            (self.policy_pool_path, self.policy_pool_path),
            (self.zsceval_pool_path, self.zsceval_pool_path),
        ]
        
        for search_path, model_base_path in search_roots:
            for pool_algo in pool_algos:
                # 1. Try standard population.yml
                pop_yaml = os.path.join(search_path, self.layout_name, pool_algo, stage, "population.yml")
                if os.path.exists(pop_yaml):
                    return pop_yaml, model_base_path
                
                other_stage = 's1' if stage == 's2' else 's2'
                pop_yaml = os.path.join(search_path, self.layout_name, pool_algo, other_stage, "population.yml")
                if os.path.exists(pop_yaml):
                    print(f"[WARN] {stage} not found, trying {other_stage}")
                    return pop_yaml, model_base_path
                
                # 2. Try any .yml file
                base_dir = os.path.join(search_path, self.layout_name, pool_algo, stage)
                if os.path.isdir(base_dir):
                    yml_files = [f for f in os.listdir(base_dir) if f.endswith('.yml')]
                    if yml_files:
                        return os.path.join(base_dir, yml_files[0]), model_base_path
                
                base_dir = os.path.join(search_path, self.layout_name, pool_algo, other_stage)
                if os.path.isdir(base_dir):
                    yml_files = [f for f in os.listdir(base_dir) if f.endswith('.yml')]
                    if yml_files:
                        print(f"[WARN] {stage} not found, trying {other_stage}")
                        return os.path.join(base_dir, yml_files[0]), model_base_path
        
        return None, None
    
    def load_policy(self, algo: str, policy_name: str = None, stage: str = 's2', ai_player: int = None) -> Tuple[Any, str, Any]:
        """Load specified algorithm policy - prefer specified Stage (default S2 adaptive)
        
        Supports from_separate_policy_pool: when population.yml contains agent_0/agent_1 groups,
        select policy for corresponding player position based on ai_player.
        
        Returns:
            (policy, featurize_type, policy_args)
        """
        pop_yaml_path, model_base_path = self._find_population_yaml(algo, stage)
        
        if pop_yaml_path and os.path.exists(pop_yaml_path):
            # Load from .yml
            with open(pop_yaml_path, 'r') as f:
                raw_config = yaml.load(f, yaml.Loader)
            
            # Detect from_separate_policy_pool format
            has_agent_groups = "agent_0" in raw_config or "agent_1" in raw_config
            
            # Smart policy selection:
            # 1. If policy_name is explicitly specified, search top-level first, then agent groups by ai_player
            # 2. If policy_name not specified, prefer top-level adaptive policy search (e.g. ltl_adaptive),
            #    otherwise select agent_0/agent_1 group by ai_player
            
            population_config = raw_config
            selected_from_agent_group = False
            
            if policy_name is not None:
                # User explicitly specified policy name
                if policy_name in raw_config:
                    population_config = raw_config
                elif has_agent_groups and ai_player is not None:
                    agent_key = f"agent_{ai_player}"
                    if agent_key in raw_config and policy_name in raw_config[agent_key]:
                        population_config = raw_config[agent_key]
                        selected_from_agent_group = True
                    else:
                        print(f"[WARN] Policy '{policy_name}' not in {agent_key} , searching top-level")
                        population_config = raw_config
                else:
                    print(f"[WARN] Policy '{policy_name}' not in population, using first available")
                    policy_name = list(population_config.keys())[0]
            else:
                # Policy name not specified: prefer top-level adaptive policy search (BACH main policy)
                adaptive_candidates = [k for k in raw_config.keys() 
                                       if k.endswith('_adaptive') or 'ltl' in k.lower()]
                if adaptive_candidates:
                    policy_name = adaptive_candidates[0]
                    population_config = raw_config
                elif has_agent_groups and ai_player is not None:
                    agent_key = f"agent_{ai_player}"
                    if agent_key in raw_config:
                        population_config = raw_config[agent_key]
                        selected_from_agent_group = True
                        policy_name = list(population_config.keys())[0]
                    else:
                        policy_name = list(population_config.keys())[0]
                else:
                    policy_name = list(population_config.keys())[0]
            
            if policy_name not in population_config:
                print(f"[WARN] Policy '{policy_name}' not in selected group, using first available")
                policy_name = list(population_config.keys())[0]
            
            config = population_config[policy_name]
            
            policy_config_path = os.path.join(
                model_base_path,
                config["policy_config_path"]
            )
            
            # First determine model path (if needed) to correct temporal_state_num before creating policy
            model_path = None
            if config.get("model_path"):
                model_path_cfg = config["model_path"]
                if isinstance(model_path_cfg, dict):
                    model_path = os.path.join(model_base_path, model_path_cfg.get("actor", ""))
                else:
                    model_path = os.path.join(model_base_path, model_path_cfg)
            else:
                # No model_path, try auto-finding .pt files in .yml directory
                yaml_dir = os.path.dirname(pop_yaml_path)
                pt_files = []
                for root, dirs, files in os.walk(yaml_dir):
                    for f in files:
                        if f.endswith('.pt'):
                            pt_files.append(os.path.join(root, f))
                if pt_files:
                    import random
                    model_path = random.choice(pt_files)
            
            with open(policy_config_path, 'rb') as f:
                policy_config = list(pickle.load(f))
            policy_args = policy_config[0]
            
            # Correct temporal_state_num before creating policy: use actual ts_encoder input dim from checkpoint
            if getattr(policy_args, "input_temporal_state", False) and model_path and os.path.exists(model_path):
                try:
                    sd = torch.load(model_path, map_location="cpu")
                    if "base.ts_encoder.rnn.weight_ih_l0" in sd:
                        actual_ts_num = sd["base.ts_encoder.rnn.weight_ih_l0"].shape[1]
                        if actual_ts_num != getattr(policy_args, "temporal_state_num", 0):
                            print(f"   [WARN] Corrected temporal_state_num: {getattr(policy_args, 'temporal_state_num', 0)} -> {actual_ts_num} (using checkpoint as source)")
                            policy_args.temporal_state_num = actual_ts_num
                except Exception as e:
                    print(f"   [WARN] Failed to correct temporal_state_num: {e}")
            
            policy = self._create_policy(policy_config)
            
            if model_path and os.path.exists(model_path):
                policy.load_checkpoint({"actor": model_path})
            elif model_path:
                print(f"[WARN] Model file not found: {model_path}")
                return None, None, None
            
            return policy, config.get("featurize_type", "ppo"), policy_args
        
        # If no population.yml, try direct checkpoint lookup
        pt_path, config_path = self.find_policy(algo, stage=stage)

        if not pt_path:
            other_stage = 's1' if stage == 's2' else 's2'
            pt_path, config_path = self.find_policy(algo, stage=other_stage)
        
        if pt_path and config_path:
            return self.load_policy_from_checkpoint(pt_path, config_path)
        
        print(f"[WARN] Policy file not found")
        return None, None, None
    
    def _find_policy_config(self, prefer_adaptive: bool = False) -> Optional[str]:
        """Find policy config file (supports policy_pool and zsceval/policy_pool)"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        standard_names = ["rnn_policy_config.pkl", "mlp_policy_config.pkl"]
        adaptive_names = ["rnn_policy_config_adaptive.pkl"]
        
        if prefer_adaptive:
            names = adaptive_names + standard_names
        else:
            names = standard_names + adaptive_names
        
        config_search_paths = []
        for name in names:
            config_search_paths.append(os.path.join(self.policy_pool_path, self.layout_name, "policy_config", name))
        for name in names:
            config_search_paths.append(os.path.join(project_root, "zsceval", "policy_pool", self.layout_name, "policy_config", name))
        
        for path in config_search_paths:
            if os.path.exists(path):
                return path
        return None

    def _find_pool_policy(self, algo: str, stage: str = "s2") -> Optional[str]:
        """Find policy files in policy_pool / zsceval/policy_pool"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        # BACH algorithm may correspond to ltl or bach directory in policy_pool
        pool_algos = [algo]
        if algo.lower() == "bach":
            pool_algos.append("ltl")

        pool_paths = []
        for pool_algo in pool_algos:
            pool_paths += [
                os.path.join(self.policy_pool_path, self.layout_name, pool_algo, stage),
                os.path.join(self.policy_pool_path, self.layout_name, pool_algo),
                os.path.join(self.policy_pool_path, pool_algo),
                os.path.join(project_root, "zsceval", "policy_pool", self.layout_name, pool_algo, stage),
                os.path.join(project_root, "zsceval", "policy_pool", self.layout_name, pool_algo),
            ]

        for base_path in pool_paths:
            if os.path.exists(base_path):
                pt_files = []
                for root, dirs, files in os.walk(base_path):
                    for f in files:
                        if f.endswith('.pt'):
                            pt_files.append(os.path.join(root, f))
                if pt_files:
                    import random
                    return random.choice(pt_files)
        return None

    def load_policy_from_checkpoint(self, checkpoint_path: str, policy_config_path: str) -> Tuple[Any, str, Any]:
        """Load policy directly from checkpoint
        
        Returns:
            (policy, featurize_type, policy_args)
        """
        
        # Load policy config
        with open(policy_config_path, 'rb') as f:
            policy_config = list(pickle.load(f))
            # policy_config: [args, obs_space, share_obs_space, act_space]
        
        policy = self._create_policy(policy_config)
        policy_args = policy_config[0]
        
        # Load model weights - use policy.load_checkpoint (safer loading method)
        if os.path.exists(checkpoint_path):
            try:
                policy.load_checkpoint({"actor": checkpoint_path})
            except Exception as e:
                print(f"[WARN] Model loading error: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
        else:
            print(f"[WARN] Model file not found: {checkpoint_path}")
            return None, None, None
        
        return policy, "ppo", policy_args
    
    def find_policy(self, algo: str, stage: str = "s2") -> Tuple[str, str]:
        """Find policy files - prefer policy_pool, then results directory (filtered by algo name)"""
        prefer_adaptive = algo.lower() in ("bach", "ltl", "adaptive")
        # 1. Prefer searching policy_pool / zsceval/policy_pool
        pool_pt = self._find_pool_policy(algo, stage)
        if pool_pt:
            config_path = self._find_policy_config(prefer_adaptive=prefer_adaptive)
            if config_path:
                return pool_pt, config_path
            else:
                # No policy_config.pkl, try inferring from checkpoint and cache
                try:
                    inferred = infer_policy_config_from_checkpoint(pool_pt, self.layout_name)
                    cache_dir = os.path.dirname(pool_pt)
                    cache_path = os.path.join(cache_dir, "inferred_policy_config.pkl")
                    with open(cache_path, "wb") as f:
                        pickle.dump(inferred, f)
                    return pool_pt, cache_path
                except Exception as e:
                    print(f"   [WARN] Found pool policy but cannot infer config: {e}")
        
        # 2. Search results directory
        project_root = os.path.join(os.path.dirname(__file__), '..')
        results_path = os.path.join(project_root, 'results', 'Overcooked', self.layout_name)
        
        if os.path.exists(results_path):
            candidates = []
            for root, dirs, files in os.walk(results_path):
                pt_files = [f for f in files if f.endswith('.pt') and 'actor' in f]
                for f in pt_files:
                    step = 0
                    if 'periodic_' in f:
                        try:
                            step = int(f.split('periodic_')[1].split('.')[0])
                        except:
                            step = 0
                    
                    pt_path = os.path.join(root, f)
                    # Calculate match score: paths containing algo name get bonus
                    rel_path = os.path.relpath(root, results_path).lower()
                    score = step
                    if algo.lower() in rel_path:
                        score += 1e9
                    candidates.append((score, pt_path))
            
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_pt = candidates[0][1]
                best_root = os.path.dirname(best_pt)
                
                # Find corresponding policy_config.pkl
                config_path = None
                for c_root, c_dirs, c_files in os.walk(best_root):
                    for cf in c_files:
                        if cf == 'policy_config.pkl':
                            config_path = os.path.join(c_root, cf)
                            break
                    if config_path:
                        break
                
                if not config_path:
                    parent = os.path.dirname(best_root)
                    for c_root, c_dirs, c_files in os.walk(parent):
                        for cf in c_files:
                            if cf == 'policy_config.pkl':
                                config_path = os.path.join(c_root, cf)
                                break
                        if config_path:
                            break
                
                if config_path:
                            return best_pt, config_path
        
        return None, None
    
    def _state_from_dict(self, state_dict: Dict) -> OvercookedState:
        """Convert dict to OvercookedState"""
        def object_from_dict(obj_dict):
            return ObjectState(**obj_dict)
        
        def player_from_dict(player_dict):
            held_obj = player_dict.get("held_object")
            if held_obj is not None:
                player_dict["held_object"] = object_from_dict(held_obj)
            return PlayerState(**player_dict)
        
        state_dict["players"] = [player_from_dict(p) for p in state_dict["players"]]
        object_list = [object_from_dict(o) for _, o in state_dict["objects"].items()]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return OvercookedState(**state_dict)
    
    def process_state(self, state, featurize_type: str, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process state for policy input"""
        if isinstance(state, dict):
            state = self._state_from_dict(state.copy())
        
        if featurize_type == "ppo":
            obs = self.mdp.lossless_state_encoding(state)[pos] * 255
            available_actions = self._get_available_actions(state)[pos]
            return obs, available_actions
        
        return None, None
    
    def _get_available_actions(self, state: OvercookedState) -> np.ndarray:
        """Get available actions - fully aligned with Overcooked_Env_new._get_available_actions"""
        num_agents = len(state.players)
        available_actions = np.ones((num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        interact_index = Action.ACTION_TO_INDEX["interact"]
        
        for agent_idx in range(num_agents):
            player = state.players[agent_idx]
            pos = player.position
            o = player.orientation
            
            for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                new_pos = Action.move_in_direction(pos, move)
                if new_pos not in self.mdp.get_valid_player_positions() and o == move:
                    available_actions[agent_idx, move_i] = 0
            
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.mdp.get_terrain_type_at_pos(i_pos)
            
            if (
                terrain_type == " "
                or (
                    terrain_type == "X"
                    and (
                        (not player.has_object() and not state.has_object(i_pos))
                        or (player.has_object() and state.has_object(i_pos))
                    )
                )
                or (terrain_type in ["O", "T", "D"] and player.has_object())
                or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
            ):
                available_actions[agent_idx, interact_index] = 0
            
            # Pot interaction logic - fully aligned with Overcooked_Env_new (old_dynamics=False)
            if terrain_type == "P":
                if player.has_object():
                    obj = player.get_object()
                    if obj.name not in ["dish", "onion", "tomato"]:
                        available_actions[agent_idx, interact_index] = 0
                elif not state.has_object(i_pos) or state.get_object(i_pos).is_ready:
                    available_actions[agent_idx, interact_index] = 0
        
        return available_actions


class HumanTestWithPolicy:
    """Human-AI interaction test class - using original rendering + trained policies."""
    
    # Action mapping (keyboard -> Action)
    ACTION_MAP = {
        pygame.K_w: (0, -1),      # NORTH
        pygame.K_UP: (0, -1),
        pygame.K_s: (0, 1),       # SOUTH
        pygame.K_DOWN: (0, 1),
        pygame.K_a: (-1, 0),      # WEST
        pygame.K_LEFT: (-1, 0),
        pygame.K_d: (1, 0),       # EAST
        pygame.K_RIGHT: (1, 0),
        pygame.K_SPACE: 'interact', # INTERACT
        pygame.K_e: 'interact',
    }
    
    def __init__(
        self,
        env_name: str,
        algo: str,
        human_player: int = 0,
        episodes: int = 1,
        tile_size: int = 75,
        fps: int = 2,  # default 2 steps per second
        policy_pool_path: Optional[str] = None,
        policy_name: Optional[str] = None,
        deterministic: bool = True,
        stage: str = 's2',  # default Stage 2
        headless: bool = False,  # no graphics mode
    ):
        self.env_name = env_name
        self.algo = algo
        self.human_player = human_player
        self.ai_player = 1 - human_player
        self.episodes = episodes
        self.tile_size = tile_size
        self.fps = fps  # control game speed
        self.deterministic = deterministic
        self.stage = stage  # s1 or s2
        self.headless = headless  # no graphics mode
        
        # Result recording
        self.results = {
            'scores': [],
            'durations': [],
            'soups_cooked': []
        }
        
        # Initialize pygame (non-headless only)
        if not headless:
            pygame.init()
            pygame.display.set_caption("Overcooked Human Test")

        # Setup environment
        self._setup_environment()

        # Create visualizer (non-headless only)
        if not headless:
            self.visualizer = StateVisualizer(
                tile_size=tile_size,
                window_fps=fps,
                is_rendering_hud=True,
                is_rendering_cooking_timer=True,
            )
        else:
            self.visualizer = None

        # Load AI policy
        self.ai_policy = None
        self.eval_policy = None
        self.policy_loader = None
        self.policy_args = None
        self.featurize_type = "ppo"
        if policy_pool_path:
            self._load_policy(policy_pool_path, policy_name)

    
    def _setup_environment(self):
        """Setup environment - prefer overcooked_new layouts."""
        print(f"\nInitializing environment: {self.env_name}")

        try:
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld as NewGridworld
            from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as NewOvercookedEnv
            mdp = NewGridworld.from_layout_name(self.env_name)
            self.env = NewOvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
            print(f"Loaded layout: {self.env_name}")
            return
        except Exception as e:
            print(f"New layout failed: {e}")

        # Fallback to random1_m
        print(f"Using default layout: random1_m")
        mdp = BaseOvercookedGridworld.from_layout_name("random1_m")
        self.env = BaseOvercookedEnv(mdp, horizon=400, info_level=0)
    
    def _load_policy(self, policy_pool_path: str, policy_name: Optional[str]):
        """Load AI policy - use EvalPolicy wrapper to align with training eval logic."""
        try:
            loader = PolicyLoader(policy_pool_path, self.env_name)
            result = loader.load_policy(self.algo, policy_name, stage=self.stage, ai_player=self.ai_player)
            
            if result and result[0] is not None:
                self.ai_policy, self.featurize_type, self.policy_args = result
                self.policy_loader = loader
                # Use EvalPolicy wrapper, fully aligned with training eval RNN state / mask management
                self.eval_policy = EvalPolicy(self.policy_args, self.ai_policy)
            else:
                print("[WARN] Policy loading failed, using random policy")
        except Exception as e:
            print(f"[ERR] Policy loading error: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_ai_action(self, state, player_id: int = None) -> Tuple:
        """Get AI action - use EvalPolicy.step(), fully aligned with training eval logic"""
        if self.ai_policy is None or self.eval_policy is None:
            # Random policy
            action_idx = np.random.randint(0, len(Action.ALL_ACTIONS))
            return Action.INDEX_TO_ACTION[action_idx]
        
        # Determine player ID
        if player_id is None:
            player_id = self.ai_player
        
        try:
            # Ensure state is OvercookedState object
            if isinstance(state, dict):
                state = self.policy_loader._state_from_dict(state.copy())
            
            # Get observation and available actions (keep *255 to match training featurization)
            obs, available_actions = self.policy_loader.process_state(
                state, self.featurize_type, player_id
            )
            
            if obs is None:
                print("[WARN] obs is None, using STAY")
                return Action.STAY
            
            # EvalPolicy.step expects numpy input, batch dim first
            obs = np.array([obs])              # shape: (1, H, W, C)
            available_actions = np.array([available_actions])  # shape: (1, n_actions)
            
            # Call EvalPolicy - adaptive policies use step_ltl, standard policies use step
            is_adaptive = getattr(self.policy_args, "input_temporal_state", False)
            if is_adaptive:
                # For BACH/adaptive policies, construct zero temporal_state_vectors
                # Prevent policy from incorrectly extracting temporal state from obs pixels
                temporal_state_num = getattr(self.policy_args, "temporal_state_num", 0)
                if temporal_state_num > 0:
                    temporal_states_vectors = np.zeros(
                        (1, temporal_state_num), dtype=np.float32
                    )
                else:
                    temporal_states_vectors = None
                
                action = self.eval_policy.step_ltl(
                    obs,
                    agents=[(0, player_id)],
                    deterministic=self.deterministic,
                    available_actions=available_actions,
                    temporal_states_vectors=temporal_states_vectors,
                )
            else:
                action = self.eval_policy.step(
                    obs,
                    agents=[(0, player_id)],
                    deterministic=self.deterministic,
                    available_actions=available_actions,
                )
            # action shape: (1, 1) numpy array
            action_idx = int(action[0][0])
            action_tuple = Action.INDEX_TO_ACTION[action_idx]
            
            if player_id == self.ai_player:
                if hasattr(self, '_step_count'):
                    self._step_count += 1
                else:
                    self._step_count = 1
            
            return action_tuple
            
        except Exception as e:
            print(f"[WARN] Policy inference error: {e}")
            import traceback
            traceback.print_exc()
            return Action.STAY
    
    def run(self):
        """Run the test session."""
        import shutil
        w = shutil.get_terminal_size().columns
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  Overcooked Human Test")
        print(f"{bar}")
        print(f"  Environment: {self.env_name}")
        print(f"  You are: Player {self.human_player} (Human)")
        print(f"  AI is: Player {self.ai_player}")
        print(f"{bar}\n")

        print("Controls:")
        print("  [WASD / Arrow keys] Move")
        print("  [Space / E] Interact (pick up / put down / cook)")
        print("  [Q / ESC] Quit")
        print()

        for episode in range(self.episodes):
            self._run_episode(episode + 1)

        self._print_summary()
        pygame.quit()

    def _run_episode(self, episode_num: int):
        """Run a single episode."""
        print(f"\n--- Episode {episode_num}/{self.episodes} ---")

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Reset EvalPolicy state (align with training eval reset logic)
        if self.eval_policy is not None:
            self.eval_policy.reset(num_envs=1, num_agents=2)
            self.eval_policy.register_control_agent(0, self.ai_player)
            # In headless mode, both agents are AI-controlled
            if self.headless:
                self.eval_policy.register_control_agent(0, self.human_player)

            # Initialize base_rnn_states for adaptive policies
            if getattr(self.policy_args, "input_temporal_state", False):
                for ea in self.eval_policy.control_agents:
                    if ea not in self.eval_policy._base_rnn_states:
                        self.eval_policy._base_rnn_states[ea] = np.zeros(
                            (2, self.eval_policy.hidden_size // 4), dtype=np.float32
                        )


        # Create display window (non-headless only)
        if not self.headless:
            grid = self.env.mdp.terrain_mtx
            test_surface = self.visualizer.render_state(
                state=state,
                grid=grid,
                hud_data={'score': 0, 'time_left': 400},
            )
            width, height = test_surface.get_size()
            
            screen = pygame.display.set_mode(
                (width, height),
                pygame.DOUBLEBUF | pygame.RESIZABLE
            )
            clock = pygame.time.Clock()
        else:
            screen = None
            clock = None
        
        running = True
        step = 0
        max_steps = 400
        total_reward = 0
        soups_cooked = 0
        start_time = pygame.time.get_ticks() if not self.headless else 0
        
        # Reset step counter
        self._step_count = 0
        
        while running and step < max_steps:
            # Process events
            human_action = Action.STAY
            if not self.headless:
                # Non-headless mode: read keyboard input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                            break
                        elif event.key in self.ACTION_MAP:
                            human_action = self.ACTION_MAP[event.key]
            else:
                # headless mode: AI runs itself (using AI policy)
                if self.ai_policy is not None:
                    human_action = self._get_ai_action(state, player_id=self.human_player)
                else:
                    import random
                    human_action = random.choice(list(self.ACTION_MAP.values()))
            
            if not running:
                break
            
            # Get AI action
            ai_action = self._get_ai_action(state, player_id=self.ai_player)
            
            
            # Combine actions
            if self.human_player == 0:
                joint_action = (human_action, ai_action)
            else:
                joint_action = (ai_action, human_action)
            
            # Execute action
            next_state, reward, done, info = self.env.step(joint_action)
            total_reward += reward
            if reward > 0:
                soups_cooked += 1
            
            state = next_state
            step += 1
            
            # Render (non-headless only)
            if not self.headless:
                self._render(screen, state, total_reward, step, soups_cooked)
                clock.tick(self.fps)
            
            if done:
                print(f"\n[OK] Episode finished! Score: {total_reward}, Steps: {step}")
                break
        
        # Record results
        if not self.headless:
            duration = (pygame.time.get_ticks() - start_time) / 1000
        else:
            duration = 0
        self.results['scores'].append(total_reward)
        self.results['durations'].append(duration)
        self.results['soups_cooked'].append(soups_cooked)
    
    def _render(self, screen, state, score: int, step: int, soups: int):
        """Render using original StateVisualizer"""
        grid = self.env.mdp.terrain_mtx
        
        hud_data = {
            'score': score,
            'time_left': max(0, 400 - step),
            'step': step,
            'soups': soups,
        }
        
        surface = self.visualizer.render_state(
            state=state,
            grid=grid,
            hud_data=hud_data,
        )
        
        screen.blit(surface, (0, 0))
        pygame.display.flip()
    
    def _print_summary(self):
        """Print test summary"""
        import shutil
        w = shutil.get_terminal_size().columns
        bar = "=" * w
        print(f"\n{bar}")
        print("  [STATS] Test complete!")
        print(bar)

        if self.results['scores']:
            avg_score = np.mean(self.results['scores'])
            avg_duration = np.mean(self.results['durations'])
            total_soups = sum(self.results['soups_cooked'])

            print(f"  Total episodes: {len(self.results['scores'])}")
            print(f"  Average score: {avg_score:.2f}")
            print(f"  Average duration: {avg_duration:.1f}s")
            print(f"  Total soups: {total_soups}")

        print(f"{bar}\n")

    def get_results(self):
        """Return test result dict (aligned with HumanTest.get_results)"""
        return {
            **self.results,
            'avg_score': np.mean(self.results['scores']) if self.results['scores'] else 0,
            'avg_duration': np.mean(self.results['durations']) if self.results['durations'] else 0,
            'total_soups': sum(self.results['soups_cooked']),
        }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='overcooked-human-test',
        description='Human-AI testing tool - original rendering + trained policies',
    )
    
    parser.add_argument('-e', '--env', type=str, default='random1_m',
                       help='Environment / layout name (e.g. random1_m, random3_m)')
    parser.add_argument('-a', '--algo', type=str, default='bach',
                       help='AI algorithm (e.g. bach, fcp, mep)')
    parser.add_argument('-p', '--human-player', type=int, default=1, choices=[0, 1],
                       help='Human player position (0 or 1, default: 1)')
    parser.add_argument('-n', '--episodes', type=int, default=1,
                       help='Number of episodes')
    parser.add_argument('--tile-size', type=int, default=75,
                       help='Tile size')
    parser.add_argument('--fps', type=int, default=2,
                       help='Game speed (steps/sec, default: 2 = 2 steps per second)')
    parser.add_argument('--policy-pool', type=str,
                       default=None,
                       help='Policy pool path (default: exp_configs/policy_pool)')
    parser.add_argument('--policy-name', type=str, default=None,
                       help='Specify policy name (optional)')
    parser.add_argument('--random', action='store_true',
                       help='Use random policy (do not load trained model)')
    parser.add_argument('--headless', action='store_true',
                       help='No graphics mode (for testing)')
    parser.add_argument('--stage', type=str, default='s2', choices=['s1', 's2'],
                       help='Policy stage (default: s2)')
    
    args = parser.parse_args()
    
    try:
        test = HumanTestWithPolicy(
            env_name=args.env,
            algo=args.algo,
            human_player=args.human_player,
            episodes=args.episodes,
            tile_size=args.tile_size,
            fps=args.fps,
            policy_pool_path=None if args.random else args.policy_pool,
            policy_name=args.policy_name,
            stage=args.stage,
            headless=args.headless,
        )
        test.run()
    except KeyboardInterrupt:
        print("\n[WARN] User interrupted")
        pygame.quit()
    except Exception as e:
        print(f"\n[ERR] Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == '__main__':
    main()
