import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.utils.act import ACTLayer
from zsceval.algorithms.utils.cnn import CNNBase, Base_TS, Base_TS_R
from zsceval.algorithms.utils.mix import MIXBase
from zsceval.algorithms.utils.mlp import MLPBase, MLPLayer
from zsceval.algorithms.utils.popart import PopArt
from zsceval.algorithms.utils.rnn import RNNLayer
from zsceval.algorithms.utils.util import check, init
from zsceval.utils.util import get_shape_from_obs_space

MASK_RATE = 0
MASK_SELF_TEMPORAL_STATE = True



def apply_random_mask(ts_vec: torch.Tensor, c: float):
    mask = torch.bernoulli(torch.full_like(ts_vec, c))
    
    ts_vec_masked = ts_vec * mask
    return ts_vec_masked

def apply_custom_mask(ts_vec: torch.Tensor) -> torch.Tensor:
    a, b = ts_vec.shape
    
    for i in range(a):
        row = ts_vec[i]
        
        if row[-2] > 0.0:
            half_idx = (b - 2) // 2
            row[:half_idx] = 0.0
        
        if row[-1] > 0.0:
            half_idx = (b - 2) // 2
            row[half_idx:b-2] = 0.0
    
    return ts_vec

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        # logger.debug(f"actor obs shape: {obs_shape}")
        logger.trace(f"actor obs shape: {obs_shape}")
        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            # MARK: MLPBase will not be used
            self.base = (
                CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(obs_shape) == 3
                else MLPBase(
                    args,
                    obs_shape,
                    use_attn_internal=args.use_attn_internal,
                    use_cat_self=True,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_policy_vhead:
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_transitions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        # ! only work for rnn model
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        values = self.v_out(actor_features)

        return values

    def get_probs(self, obs, rnn_states, masks, available_actions=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        action_probs = self.act.get_probs(actor_features, available_actions)

        return action_probs, rnn_states

    def get_action_log_probs(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values, rnn_states


class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self._num_v_out = getattr(args, "num_v_out", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        logger.trace(f"critic share obs shape: {share_obs_shape}")

        if "Dict" in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            # MARK: MLPBase will not be used
            self.base = (
                CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(share_obs_shape) == 3
                else MLPBase(
                    args,
                    share_obs_shape,
                    use_attn_internal=True,
                    use_cat_self=args.use_cat_self,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                share_obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, self._num_v_out, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, self._num_v_out))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks, task_id=None):
        if self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)

        if self._layer_after_N > 0:
            critic_features = self.mlp_after(critic_features)

        values = self.v_out(critic_features)

        if self._num_v_out > 1 and task_id is not None:
            assert len(task_id.shape) == len(values.shape) and np.prod(task_id.shape) * self._num_v_out == np.prod(
                values.shape
            ), (task_id.shape, values.shape)
            values = torch.gather(values, -1, task_id.long())

        return values, rnn_states


class R_Actor_adaptive(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart
        self._recurrent_N = args.recurrent_N
        self._input_temporal_state = args.input_temporal_state
        self._temporal_state_num = args.temporal_state_num
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.mask_rate_max = MASK_RATE
        self.mask_rate_ratio = 0.0

        obs_shape = get_shape_from_obs_space(obs_space)
        logger.trace(f"actor obs shape: {obs_shape}")

        self.base = Base_TS_R(args, obs_shape, self._temporal_state_num, cnn_layers_params=args.cnn_layers_params)

        rnn_input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(rnn_input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m): return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_policy_vhead:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device)) if self._use_popart else init_(nn.Linear(self.hidden_size, 1))

        self.to(device)


    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, temporal_state_vectors=None, episode_ratio=0.0, base_rnn_states=None, base_rnn_masks=None, agent_idxs=None):
        # agent_idx = 0
        # if obs[0][0][0][24]:
        #     agent_idx = 1
        
        # obs[..., 24] = 0

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        #tgt_idx=-1 if agent_idx else -2

        if base_rnn_states is None:
            base_rnn_states = torch.zeros(
                (rnn_states.shape[0], 2, self.hidden_size//4),
                dtype=rnn_states.dtype,
                device=rnn_states.device
            )
        else:
            base_rnn_states = check(np.array(base_rnn_states)).to(**self.tpdv)
            
        if base_rnn_masks is None:
            base_rnn_masks = torch.ones(
                (masks.shape[0], 1),
                dtype=masks.dtype,
                device=masks.device 
            )
        else:
            base_rnn_masks = check(base_rnn_masks).to(**self.tpdv)

        if temporal_state_vectors is None:
            ts_vec = obs.reshape(obs.size(0), -1)[:, :self._temporal_state_num]
        else:
            # for x in temporal_state_vectors:x[..., tgt_idx]=255.0
            ts_vec=check(np.asarray(temporal_state_vectors,dtype=np.float32)).to(**self.tpdv)
            ts_vec=ts_vec.reshape(-1,ts_vec.shape[-1])
            self.mask_rate_ratio = episode_ratio
            ts_vec = apply_random_mask(ts_vec, self.mask_rate_max*self.mask_rate_ratio) # ts_vec = apply_random_mask(ts_vec, self.mask_rate_max)
            if MASK_SELF_TEMPORAL_STATE:
                ts_vec = apply_custom_mask(ts_vec)


        base_features, base_rnn_states = self.base(obs, ts_vec, base_rnn_states, base_rnn_masks)

        actor_features, rnn_states = self.rnn(base_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states, base_rnn_states

    # def evaluate_transitions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, temporal_state_vectors=None, base_rnn_states=None, base_rnn_masks=None):
    #     # agent_idx = 0
    #     # if obs[0][0][0][24]:
    #     #     agent_idx = 1
        
    #     # obs[..., 24] = 0

    #     obs = check(obs).to(**self.tpdv)
    #     rnn_states = check(rnn_states).to(**self.tpdv)
    #     masks = check(masks).to(**self.tpdv)

    #     if available_actions is not None:
    #         available_actions = check(available_actions).to(**self.tpdv)

    #     # tgt_idx=-1 if agent_idx else -2

    #     if temporal_state_vectors is None:
    #         ts_vec = obs.reshape(obs.size(0), -1)[:, :self._temporal_state_num]
    #     else:
    #         # for x in temporal_state_vectors:x[..., tgt_idx]=255.0
    #         ts_vec=check(np.asarray(temporal_state_vectors,dtype=np.float32)).to(**self.tpdv)
    #         ts_vec=ts_vec.reshape(-1,ts_vec.shape[-1])
    #         # ts_vec = apply_random_mask(ts_vec, self.mask_rate_max*self.mask_rate_ratio)

    #     base_features = self.base(obs, ts_vec)

    #     actor_features, rnn_states = self.rnn(base_features, rnn_states, masks)

    #     action_log_probs, dist_entropy = self.act.evaluate_actions(
    #         actor_features,
    #         action,
    #         available_actions,
    #         active_masks=active_masks if self._use_policy_active_masks else None,
    #     )

    #     values = self.v_out(actor_features) if self._use_policy_vhead else None

    #     return action_log_probs, dist_entropy, values, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, temporal_state_vectors=None, base_rnn_states=None, base_rnn_masks=None, agent_idxs=None):
        # agent_idx = 0
        # if obs[0][0][0][24]:
        #     agent_idx = 1
        
        # obs[..., 24] = 0

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if base_rnn_states is None:
            base_rnn_states = torch.zeros(
                (rnn_states.shape[0], 2, self.hidden_size//4),
                dtype=rnn_states.dtype,
                device=rnn_states.device
            )
        else:
            base_rnn_states = check(base_rnn_states).to(**self.tpdv)
            
        if base_rnn_masks is None:
            base_rnn_masks = torch.ones(
                (masks.shape[0], 1),
                dtype=masks.dtype,
                device=masks.device 
            )
        else:
            base_rnn_masks = check(base_rnn_masks).to(**self.tpdv)

        # tgt_idx=-1 if agent_idx else -2

        if temporal_state_vectors is None:
            ts_vec = obs.reshape(obs.size(0), -1)[:, :self._temporal_state_num]
        else:
            # for x in temporal_state_vectors:x[..., tgt_idx]=255.0
            ts_vec=check(np.asarray(temporal_state_vectors,dtype=np.float32)).to(**self.tpdv)
            ts_vec=ts_vec.reshape(-1,ts_vec.shape[-1])
            ts_vec = apply_random_mask(ts_vec, self.mask_rate_max*self.mask_rate_ratio)
            if MASK_SELF_TEMPORAL_STATE:
                ts_vec = apply_custom_mask(ts_vec)

        base_features, _ = self.base(obs, ts_vec, base_rnn_states, base_rnn_masks)

        actor_features, rnn_states = self.rnn(base_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values

    # def get_policy_values(self, obs, rnn_states, masks):
    #     if self._mixed_obs:
    #         for key in obs.keys():
    #             obs[key] = check(obs[key]).to(**self.tpdv)
    #     else:
    #         obs = check(obs).to(**self.tpdv)
    #     rnn_states = check(rnn_states).to(**self.tpdv)
    #     masks = check(masks).to(**self.tpdv)

    #     actor_features = self.base(obs)

    #     if self._use_naive_recurrent_policy or self._use_recurrent_policy:
    #         actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
    #     if self._use_influence_policy:
    #         mlp_obs = self.mlp(obs)
    #         actor_features = torch.cat([actor_features, mlp_obs], dim=1)
    #     if self._layer_after_N > 0:
    #         actor_features = self.mlp_after(actor_features)

    #     values = self.v_out(actor_features)

    #     return values

    # def get_probs(self, obs, rnn_states, masks, available_actions=None):
    #     if self._mixed_obs:
    #         for key in obs.keys():
    #             obs[key] = check(obs[key]).to(**self.tpdv)
    #     else:
    #         obs = check(obs).to(**self.tpdv)

    #     rnn_states = check(rnn_states).to(**self.tpdv)
    #     masks = check(masks).to(**self.tpdv)

    #     if available_actions is not None:
    #         available_actions = check(available_actions).to(**self.tpdv)

    #     actor_features = self.base(obs)

    #     if self._use_naive_recurrent_policy or self._use_recurrent_policy:
    #         actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
    #     if self._use_influence_policy:
    #         mlp_obs = self.mlp(obs)
    #         actor_features = torch.cat([actor_features, mlp_obs], dim=1)
    #     if self._layer_after_N > 0:
    #         actor_features = self.mlp_after(actor_features)

    #     action_probs = self.act.get_probs(actor_features, available_actions)

    #     return action_probs, rnn_states

    # def get_action_log_probs(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
    #     if self._mixed_obs:
    #         for key in obs.keys():
    #             obs[key] = check(obs[key]).to(**self.tpdv)
    #     else:
    #         obs = check(obs).to(**self.tpdv)

    #     rnn_states = check(rnn_states).to(**self.tpdv)
    #     action = check(action).to(**self.tpdv)
    #     masks = check(masks).to(**self.tpdv)

    #     if available_actions is not None:
    #         available_actions = check(available_actions).to(**self.tpdv)

    #     if active_masks is not None:
    #         active_masks = check(active_masks).to(**self.tpdv)

    #     actor_features = self.base(obs)

    #     if self._use_naive_recurrent_policy or self._use_recurrent_policy:
    #         actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
    #     if self._use_influence_policy:
    #         mlp_obs = self.mlp(obs)
    #         actor_features = torch.cat([actor_features, mlp_obs], dim=1)
    #     if self._layer_after_N > 0:
    #         actor_features = self.mlp_after(actor_features)

    #     action_log_probs, dist_entropy = self.act.evaluate_actions(
    #         actor_features,
    #         action,
    #         available_actions,
    #         active_masks=active_masks if self._use_policy_active_masks else None,
    #     )

    #     values = self.v_out(actor_features) if self._use_policy_vhead else None

    #     return action_log_probs, dist_entropy, values, rnn_states


class R_Critic_adaptive(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self._num_v_out = getattr(args, "num_v_out", 1)
        self._input_temporal_state = args.input_temporal_state
        self._temporal_state_num = args.temporal_state_num
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        logger.trace(f"critic share obs shape: {share_obs_shape}")

        self._mixed_obs = False
        # MARK: MLPBase will not be used
        self.base = Base_TS_R(args, share_obs_shape, self._temporal_state_num, cnn_layers_params=args.cnn_layers_params)
        # self.base = CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)

        rnn_input_size = self.base.output_size

        self.rnn = RNNLayer(rnn_input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        input_size = self.hidden_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(input_size, self._num_v_out))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks, task_id=None, temporal_state_vectors=None, base_rnn_states=None, base_rnn_masks=None):
        # share_obs[..., 24] = 0
        # share_obs[..., 49] = 0
        share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if base_rnn_states is None:
            base_rnn_states = torch.zeros(
                (rnn_states.shape[0], 2, self.hidden_size//4),
                dtype=rnn_states.dtype,
                device=rnn_states.device
            )
        else:
            base_rnn_states = check(base_rnn_states).to(**self.tpdv)
            
        if base_rnn_masks is None:
            base_rnn_masks = torch.ones(
                (masks.shape[0], 1),
                dtype=masks.dtype,
                device=masks.device 
            )
        else:
            base_rnn_masks = check(base_rnn_masks).to(**self.tpdv)

        if temporal_state_vectors is None:
            ts_vec = share_obs.reshape(share_obs.size(0), -1)[:, :self._temporal_state_num]
        else:
            # for x in temporal_state_vectors:x[..., tgt_idx]=255.0
            ts_vec=check(np.asarray(temporal_state_vectors,dtype=np.float32)).to(**self.tpdv)
            ts_vec=ts_vec.reshape(-1,ts_vec.shape[-1])
            # ts_vec = apply_random_mask(ts_vec, self.mask_rate_max*self.mask_rate_ratio)

        base_features, base_rnn_states = self.base(share_obs, ts_vec, base_rnn_states, base_rnn_masks)

        critic_features, rnn_states = self.rnn(base_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states, base_rnn_states
    