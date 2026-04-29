import importlib
import os
import pickle
from collections.abc import Iterable

import numpy as np
import torch


def _t2n(x):
    return x.detach().cpu().numpy()


def make_trainer_policy_cls(algorithm_name, use_single_network=False):
    algorithm_dict = {
        "rmappo": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
        "mappo": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
        "population": (
            "zsceval.algorithms.population.trainer_pool.TrainerPool",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "mep": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "adaptive": (
            "zsceval.algorithms.population.mep.MEP_Trainer_adaptive",
            "zsceval.algorithms.population.policy_pool.PolicyPool_adaptive",
        ),
        "cole": (
            "zsceval.algorithms.population.cole.COLE_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "traj": (
            "zsceval.algorithms.population.traj.Traj_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "trajedi": (
            "zsceval.algorithms.population.traj.Traj_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "hsp": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "fcp": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "e3t": (
            "zsceval.algorithms.population.mep.MEP_Trainer",
            "zsceval.algorithms.population.policy_pool.PolicyPool",
        ),
        "sp": (
            "zsceval.algorithms.r_mappo.r_mappo.R_MAPPO",
            "zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy",
        ),
    }

    if algorithm_name not in algorithm_dict:
        raise NotImplementedError(f"Algorithm {algorithm_name} not supported.")

    trainer_name, policy_name = algorithm_dict[algorithm_name]

    def get_alg_module(algorithm_name):
        algorithm_path = ".".join(algorithm_name.split(".")[:-1])
        module = importlib.import_module(algorithm_path)
        return module

    def get_alg_class(algorithm_name):
        module = get_alg_module(algorithm_name)
        class_name = algorithm_name.split(".")[-1]
        return getattr(module, class_name)

    trainer_class = get_alg_class(trainer_name)
    policy_class = get_alg_class(policy_name)

    return trainer_class, policy_class
