import os
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.runner.shared.base_runner import make_trainer_policy_cls
from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy_adaptive, R_MAPPOPolicy

# 从当前文件位置推导项目根目录，使用相对路径
_zsceval_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
POLICY_POOL_PATH = os.path.join(_zsceval_root, "zsceval", "policy_pool")
# 支持 POLICY_POOL 环境变量覆盖
if os.environ.get("POLICY_POOL"):
    POLICY_POOL_PATH = os.environ.get("POLICY_POOL")
ACTOR_POOL_PATH = os.environ.get("EVOLVE_ACTOR_POOL")


def add_path_prefix(pre, x):
    if isinstance(x, dict):
        return {k: add_path_prefix(pre, v) for k, v in x.items()}
    elif isinstance(x, str):
        return os.path.join(pre, x)
    else:
        raise RuntimeError(f"Add path prefix doesn't support type {type(x)}")


class PolicyPool_adaptive:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.policy_pool = dict()
        self.policy_config = dict()
        self.policy_train = dict()
        self.policy_info = dict()
        self.map_ea2p = dict()

    @staticmethod
    def normal_init(device=torch.device("cpu")):
        return PolicyPool(None, None, None, None, device=device)

    def register_policy(self, policy_name, policy, policy_config, policy_train, policy_info):
        # MARK: the identifier of a policy is its name
        if policy_name in self.policy_pool.keys():
            raise RuntimeError(f"Policy name {policy_name} already in policy pool.")
        self.policy_pool[policy_name] = policy
        self.policy_config[policy_name] = policy_config  # args, obs_space, share_obs_space, act_space
        self.policy_train[policy_name] = policy_train
        self.policy_info[policy_name] = policy_info

    def update_policy(self, policy_name, policy_train, **policy_config_kwargs):
        if policy_name not in self.policy_pool.keys():
            raise RuntimeError(f"Policy name {policy_name} not in policy pool.")
        self.policy_train[policy_name] = policy_train
        old_config = {}
        for k, v in policy_config_kwargs.items():
            assert k in self.policy_info[policy_name][1], (
                k,
                self.policy_info[policy_name][1],
            )
            old_config[k] = self.policy_info[policy_name][1][k]
            self.policy_info[policy_name][1][k] = v
        if "model_path" in old_config and old_config["model_path"] != self.policy_info[policy_name][1]["model_path"]:
            policy = self.policy_pool[policy_name]
            if isinstance(policy, EvalPolicy):
                policy = policy.policy
            if self.args.algorithm_type == "co-play":
                path_prefix = POLICY_POOL_PATH
            else:
                path_prefix = ACTOR_POOL_PATH
            policy.load_checkpoint(add_path_prefix(path_prefix, self.policy_info[policy_name][1]["model_path"]))

    def all_policies(self) -> List:
        return [
            (
                policy_name,
                self.policy_pool[policy_name],
                self.policy_config[policy_name],
                self.policy_train[policy_name],
            )
            for policy_name in self.policy_pool.keys()
        ]

    def set_map_ea2p(self, map_ea2p: Dict[Tuple[int, int], str], load_unused_to_cpu=False):
        self.map_ea2p = map_ea2p
        active_policies = np.unique(list(map_ea2p.values()))
        for policy_name in self.policy_pool.keys():
            if load_unused_to_cpu:
                if policy_name in active_policies:
                    self.policy_pool[policy_name].to(self.device)
                else:
                    self.policy_pool[policy_name].to(torch.device("cpu"))
            else:
                self.policy_pool[policy_name].to(self.device)

    def trans_to_eval(self):
        for policy_name in self.policy_pool.keys():
            if not isinstance(self.policy_pool[policy_name], EvalPolicy):
                self.policy_pool[policy_name] = EvalPolicy(
                    self.policy_config[policy_name][0], self.policy_pool[policy_name]
                )

    def load_population(self, population_yaml_path, evaluation=False, override_policy_config={}) -> Dict[str, str]:
        # load population
        # warnings.warn(
        #     "Policy pool currently loads all checkpoints into gpu, consider load into cpu latter..."
        # )
        if self.args.from_separate_policy_pool:
            raw_cfg = yaml.load(open(population_yaml_path), yaml.Loader) or {}
            population_cfg = {}

            for k, v in raw_cfg.items():
                if k.startswith("agent_"):
                    for sub_name, sub_cfg in (v or {}).items():
                        population_cfg[f"{k}/{sub_name}"] = sub_cfg
                else:
                    population_cfg[k] = v

            if not population_cfg:
                self.featurize_type, self.num_policies = {}, 0
                return {}

            featurize_type = {}
            self.num_policies = len(population_cfg)

            for idx, (pname, pcfg) in enumerate(population_cfg.items()):
                try:
                    pcfg["id"] = (idx + 1) / self.num_policies

                    cfg_path = os.path.join(POLICY_POOL_PATH, pcfg["policy_config_path"])
                    p_args, obs_s, share_obs_s, act_s = pickle.load(open(cfg_path, "rb"))

                    if pname in override_policy_config:
                        ovr = override_policy_config[pname]
                        if ovr[0] is not None:
                            for k, v in ovr[0]._get_kwargs():
                                setattr(p_args, k, v)
                        if ovr[1] is not None: obs_s = ovr[1]
                        if ovr[2] is not None: share_obs_s = ovr[2]
                        if ovr[3] is not None: act_s = ovr[3]

                    has_temporal = getattr(p_args, "input_temporal_state", False)

                    if has_temporal:
                        p_args.temporal_state_num = getattr(
                            self.args, "temporal_state_num",
                            getattr(p_args, "temporal_state_num", 0)
                        )
                        policy = R_MAPPOPolicy_adaptive(p_args, obs_s, share_obs_s, act_s, device=self.device)
                    else:
                        policy = R_MAPPOPolicy(p_args, obs_s, share_obs_s, act_s, device=self.device)

                    policy.to(torch.device("cpu"))

                    if pcfg.get("model_path"):
                        prefix = POLICY_POOL_PATH if self.args.algorithm_type == "co-play" else ACTOR_POOL_PATH
                        policy.load_checkpoint(add_path_prefix(prefix, pcfg["model_path"]))

                    train_flag = (not evaluation) and pcfg.get("train", False)
                    if train_flag:
                        policy.to_parallel()
                    if evaluation:
                        policy = EvalPolicy(p_args, policy)

                    self.register_policy(pname, policy, [p_args, obs_s, share_obs_s, act_s], train_flag, [pname, pcfg])
                    featurize_type[pname] = pcfg["featurize_type"]
                except Exception as e:
                    warnings.warn(f"Load policy {pname} failed: {e}")
                    raise e

            self.featurize_type = featurize_type
            return featurize_type
        else:
            population_config = yaml.load(open(population_yaml_path), yaml.Loader)
            if population_config is None:
                return dict()
            featurize_type = dict()
            num = len(population_config)
            self.num_policies = num
            for i, policy_name in enumerate(population_config):
                try:
                    population_config[policy_name]["id"] = (i + 1) / num
                    policy_config_path = os.path.join(
                        POLICY_POOL_PATH,
                        population_config[policy_name]["policy_config_path"],
                    )
                    policy_config = list(
                        pickle.load(open(policy_config_path, "rb"))
                    )  # args, obs_shape, share_obs_shape, act_shape
                    if policy_name in override_policy_config:
                        if override_policy_config[policy_name][0] is not None:
                            for k, v in override_policy_config[policy_name][0]._get_kwargs():
                                # logger.debug(f"override {k} as {v}")
                                setattr(policy_config[0], k, v)
                        for w in range(1, 4):
                            if override_policy_config[policy_name][w] is not None:
                                policy_config[w] = override_policy_config[policy_name][w]
                    policy_args = policy_config[0]
                    _, policy_cls = make_trainer_policy_cls(
                        policy_args.algorithm_name,
                        use_single_network=policy_args.use_single_network,
                    )

                    policy = policy_cls(*policy_config, device=self.device)
                    policy.to(torch.device("cpu"))

                    if population_config[policy_name].get("model_path", None):
                        if self.args.algorithm_type == "co-play":
                            path_prefix = POLICY_POOL_PATH
                        else:
                            path_prefix = ACTOR_POOL_PATH
                        model_path = add_path_prefix(path_prefix, population_config[policy_name]["model_path"])
                        policy.load_checkpoint(model_path)
                    policy_train = False
                    if not evaluation and "train" in population_config[policy_name].keys():
                        policy_train = population_config[policy_name]["train"]
                    if policy_train:
                        policy.to_parallel()
                    if evaluation:
                        policy = EvalPolicy(policy_args, policy)
                    policy_info = [policy_name, population_config[policy_name]]
                    self.register_policy(policy_name, policy, policy_config, policy_train, policy_info)
                    featurize_type[policy_name] = population_config[policy_name]["featurize_type"]
                except Exception as e:
                    warnings.warn(f"Load policy {policy_name} failed due to {e}")
                    raise e
            self.featurize_type = featurize_type
            return featurize_type

class PolicyPool:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.policy_pool = dict()
        self.policy_config = dict()
        self.policy_train = dict()
        self.policy_info = dict()
        self.map_ea2p = dict()

    @staticmethod
    def normal_init(device=torch.device("cpu")):
        return PolicyPool(None, None, None, None, device=device)

    def register_policy(self, policy_name, policy, policy_config, policy_train, policy_info):
        # MARK: the identifier of a policy is its name
        if policy_name in self.policy_pool.keys():
            raise RuntimeError(f"Policy name {policy_name} already in policy pool.")
        self.policy_pool[policy_name] = policy
        self.policy_config[policy_name] = policy_config  # args, obs_space, share_obs_space, act_space
        self.policy_train[policy_name] = policy_train
        self.policy_info[policy_name] = policy_info

    def update_policy(self, policy_name, policy_train, **policy_config_kwargs):
        if policy_name not in self.policy_pool.keys():
            raise RuntimeError(f"Policy name {policy_name} not in policy pool.")
        self.policy_train[policy_name] = policy_train
        old_config = {}
        for k, v in policy_config_kwargs.items():
            assert k in self.policy_info[policy_name][1], (
                k,
                self.policy_info[policy_name][1],
            )
            old_config[k] = self.policy_info[policy_name][1][k]
            self.policy_info[policy_name][1][k] = v
        if "model_path" in old_config and old_config["model_path"] != self.policy_info[policy_name][1]["model_path"]:
            policy = self.policy_pool[policy_name]
            if isinstance(policy, EvalPolicy):
                policy = policy.policy
            if self.args.algorithm_type == "co-play":
                path_prefix = POLICY_POOL_PATH
            else:
                path_prefix = ACTOR_POOL_PATH
            policy.load_checkpoint(add_path_prefix(path_prefix, self.policy_info[policy_name][1]["model_path"]))

    def all_policies(self) -> List:
        return [
            (
                policy_name,
                self.policy_pool[policy_name],
                self.policy_config[policy_name],
                self.policy_train[policy_name],
            )
            for policy_name in self.policy_pool.keys()
        ]

    def set_map_ea2p(self, map_ea2p: Dict[Tuple[int, int], str], load_unused_to_cpu=False):
        self.map_ea2p = map_ea2p
        active_policies = np.unique(list(map_ea2p.values()))
        for policy_name in self.policy_pool.keys():
            if load_unused_to_cpu:
                if policy_name in active_policies:
                    self.policy_pool[policy_name].to(self.device)
                else:
                    self.policy_pool[policy_name].to(torch.device("cpu"))
            else:
                self.policy_pool[policy_name].to(self.device)

    def trans_to_eval(self):
        for policy_name in self.policy_pool.keys():
            if not isinstance(self.policy_pool[policy_name], EvalPolicy):
                self.policy_pool[policy_name] = EvalPolicy(
                    self.policy_config[policy_name][0], self.policy_pool[policy_name]
                )

    def load_population(self, population_yaml_path, evaluation=False, override_policy_config={}) -> Dict[str, str]:
        # load population
        # warnings.warn(
        #     "Policy pool currently loads all checkpoints into gpu, consider load into cpu latter..."
        # )
        if self.args.from_separate_policy_pool:
            raw_cfg = yaml.load(open(population_yaml_path), yaml.Loader) or {}
            population_cfg = {}

            for k, v in raw_cfg.items():
                if k.startswith("agent_"):
                    for sub_name, sub_cfg in (v or {}).items():
                        population_cfg[f"{k}/{sub_name}"] = sub_cfg
                else:
                    population_cfg[k] = v

            if not population_cfg:
                self.featurize_type, self.num_policies = {}, 0
                return {}

            featurize_type = {}
            self.num_policies = len(population_cfg)

            for idx, (pname, pcfg) in enumerate(population_cfg.items()):
                try:
                    pcfg["id"] = (idx + 1) / self.num_policies

                    cfg_path = os.path.join(POLICY_POOL_PATH, pcfg["policy_config_path"])
                    p_args, obs_s, share_obs_s, act_s = pickle.load(open(cfg_path, "rb"))

                    if pname in override_policy_config:
                        ovr = override_policy_config[pname]
                        if ovr[0] is not None:
                            for k, v in ovr[0]._get_kwargs():
                                setattr(p_args, k, v)
                        if ovr[1] is not None: obs_s = ovr[1]
                        if ovr[2] is not None: share_obs_s = ovr[2]
                        if ovr[3] is not None: act_s = ovr[3]

                    _, PolicyCls = make_trainer_policy_cls(
                        p_args.algorithm_name,
                        use_single_network=p_args.use_single_network
                    )
                    p_args.from_separate_policy_pool = self.args.from_separate_policy_pool
                    p_args.temporal_state_num = self.args.temporal_state_num
                    policy = PolicyCls(p_args, obs_s, share_obs_s, act_s, device=self.device)
                    policy.to(torch.device("cpu"))

                    if pcfg.get("model_path"):
                        prefix = POLICY_POOL_PATH if self.args.algorithm_type == "co-play" else ACTOR_POOL_PATH
                        policy.load_checkpoint(add_path_prefix(prefix, pcfg["model_path"]))

                    train_flag = (not evaluation) and pcfg.get("train", False)
                    if train_flag:
                        policy.to_parallel()
                    if evaluation:
                        policy = EvalPolicy(p_args, policy)

                    self.register_policy(pname, policy, [p_args, obs_s, share_obs_s, act_s], train_flag, [pname, pcfg])
                    featurize_type[pname] = pcfg["featurize_type"]
                except Exception as e:
                    warnings.warn(f"Load policy {pname} failed: {e}")
                    raise e

            self.featurize_type = featurize_type
            return featurize_type
        else:
            population_config = yaml.load(open(population_yaml_path), yaml.Loader)
            if population_config is None:
                return dict()
            featurize_type = dict()
            num = len(population_config)
            self.num_policies = num
            for i, policy_name in enumerate(population_config):
                try:
                    population_config[policy_name]["id"] = (i + 1) / num
                    policy_config_path = os.path.join(
                        POLICY_POOL_PATH,
                        population_config[policy_name]["policy_config_path"],
                    )
                    policy_config = list(
                        pickle.load(open(policy_config_path, "rb"))
                    )  # args, obs_shape, share_obs_shape, act_shape
                    if policy_name in override_policy_config:
                        if override_policy_config[policy_name][0] is not None:
                            for k, v in override_policy_config[policy_name][0]._get_kwargs():
                                # logger.debug(f"override {k} as {v}")
                                setattr(policy_config[0], k, v)
                        for w in range(1, 4):
                            if override_policy_config[policy_name][w] is not None:
                                policy_config[w] = override_policy_config[policy_name][w]
                    policy_args = policy_config[0]
                    _, policy_cls = make_trainer_policy_cls(
                        policy_args.algorithm_name,
                        use_single_network=policy_args.use_single_network,
                    )

                    if hasattr(policy_config[0], "input_temporal_state") and policy_config[0].input_temporal_state:
                        p_args, obs_s, share_obs_s, act_s = policy_config
                        p_args.temporal_state_num = self.args.temporal_state_num
                        policy = R_MAPPOPolicy_adaptive(p_args, obs_s, share_obs_s, act_s, device=self.device)
                    else:
                        policy = R_MAPPOPolicy(*policy_config, device=self.device)

                    policy.to(torch.device("cpu"))

                    if population_config[policy_name].get("model_path", None):
                        if self.args.algorithm_type == "co-play":
                            path_prefix = POLICY_POOL_PATH
                        else:
                            path_prefix = ACTOR_POOL_PATH
                        model_path = add_path_prefix(path_prefix, population_config[policy_name]["model_path"])
                        policy.load_checkpoint(model_path)
                    policy_train = False
                    if not evaluation and "train" in population_config[policy_name].keys():
                        policy_train = population_config[policy_name]["train"]
                    if policy_train:
                        policy.to_parallel()
                    if evaluation:
                        policy = EvalPolicy(policy_args, policy)
                    policy_info = [policy_name, population_config[policy_name]]
                    self.register_policy(policy_name, policy, policy_config, policy_train, policy_info)
                    featurize_type[policy_name] = population_config[policy_name]["featurize_type"]
                except Exception as e:
                    warnings.warn(f"Load policy {policy_name} failed due to {e}")
                    raise e
            self.featurize_type = featurize_type
            return featurize_type
