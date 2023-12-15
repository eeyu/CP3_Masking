import numpy as np
import scipy.sparse as sp
import pickle

import paths
import utils_public as up
import matplotlib.pyplot as plt
from enum import Enum
import torch

class Constraint(Enum):
    H_BC = 1
    V_BC = 2
    H_LOAD = 3
    V_LOAD = 4
    VOL_FRAC = 5

class Source(Enum):
    TRAIN = 0
    TRAIN_MASKED_0=1
    TRAIN_MASKED_1=2
    TRAIN_MASKED_2=3
    TEST = 4
    TEST_MASKED = 5
    SUBMISSION_MASKED = 6

TOPOLOGY_DATA_NAMES_MAP = {
    Source.TRAIN: ("topologies_train.npy", "constraints_train.npy"),
    Source.TRAIN_MASKED_0: ("masked_topologies_train_0.npy", "masked_constraints_train_0.pickle"),
    Source.TRAIN_MASKED_1: ("masked_topologies_train_1.npy", "masked_constraints_train_1.pickle"),
    Source.TRAIN_MASKED_2: ("masked_topologies_train_2.npy", "masked_constraints_train_2.pickle"),
    Source.TEST: ("topologies_test.npy", "constraints_test.npy"),
    Source.TEST_MASKED: ("masked_topologies_test.npy", "masked_constraints_test.npy"),
    Source.SUBMISSION_MASKED: ("masked_topologies_submission.npy", "masked_constraints_submission.npy"),
}

MASKED_TRAINING_SET = [Source.TRAIN_MASKED_1, Source.TRAIN_MASKED_2, Source.TRAIN_MASKED_0]

def get_extension(file_name):
    period = file_name.find(".")
    return file_name[period+1:]

class TopologySet:
    def __init__(self, topology, constraints):
        # Topology in (64, 64)
        # Constraints 5 x (64, 64) or 1
        self.topology = topology
        self.constraints = {
            Constraint.H_BC: constraints[0].toarray() if constraints[0] is not None else None,
            Constraint.V_BC:  constraints[1].toarray() if constraints[1] is not None else None,
            Constraint.H_LOAD:  constraints[2].toarray() if constraints[2] is not None else None,
            Constraint.V_LOAD:  constraints[3].toarray() if constraints[3] is not None else None,
            Constraint.VOL_FRAC:  constraints[4],
        }
        self._constraints_raw = constraints


    def sample_masked_topology(self, mask_topology=False, mask_constraints=False):
        # For topology
        topology = self.topology
        if mask_topology:
            mask = up.random_n_masks(np.array((64, 64)), 4, 7).astype(bool)  # From the utils file - feel free to check it out
            topology = self.topology * (1 - mask) + 0.5 * (mask)

        # For constraint
        constraints = self._constraints_raw.copy()
        if mask_constraints:
            mask = np.random.choice(range(5), 2, replace=False)
            for j in mask:
                constraints[j] = None

        return TopologySet(topology=topology, constraints=constraints)

class Topologies:
    def __init__(self, source: Source = Source.TRAIN):
        topologies_name, constraints_name = TOPOLOGY_DATA_NAMES_MAP[source]
        self._topologies = np.load(paths.HOME_PATH + "data/" + topologies_name)
        if source in MASKED_TRAINING_SET:
            with open(paths.HOME_PATH + "data/" + constraints_name, 'rb') as f:
                self._constraints_sparse = pickle.load(f)
        else:
            self._constraints_sparse = np.load(paths.HOME_PATH + "data/" + constraints_name, allow_pickle=True)

        self.length = len(self._topologies)

        self.topology_sets = []
        for i in range(len(self._topologies)):
            self.topology_sets.append(TopologySet(self._topologies[i], self._constraints_sparse[i]))
        # Constraints:
        # - H BC
        # - V BC
        # - H Loads
        # - V Loads
        # - Volume Fraction
    def get_data(self, i):
        return self.topology_sets[i]

    def get_data_with_new_mask(self, i, mask_topology=False, mask_constraints=False):
        return self.topology_sets[i].sample_masked_topology(mask_topology=mask_topology, mask_constraints=mask_constraints)

    def get_topology_tensor(self, indices=None, mask=False, as_tensor=False):
        tensor = []
        if indices is None:
            indices = list(range(self.length))
        for i in indices:
            tensor.append(self.get_data_with_new_mask(i, mask_topology=mask).topology)
        if as_tensor:
            tensor = np.array(tensor)
            tensor = torch.from_numpy(tensor).float().to(paths.device)
            return tensor.unsqueeze(1)
        else:
            return np.array(tensor)

    def get_constraints(self, indices, constraint_type: Constraint, mask=False, sparse=False):
        tensor = []
        for i in indices:
            masked_set = self.topology_sets[i].sample_masked_topology(mask_constraints=mask)
            constraint = masked_set.constraints[constraint_type]
            if sparse and constraint is not None:
                constraint = sp.csr_matrix(constraint)
            tensor.append(constraint)
        return tensor

    def get_constraints_tensor(self, constraint_type: Constraint, indices=None, as_tensor = False):
        tensor = []
        if indices is None:
            indices = list(range(self.length))
        for i in indices:
            masked_set = self.topology_sets[i].sample_masked_topology(mask_constraints=False)
            constraint = masked_set.constraints[constraint_type]
            tensor.append(constraint)
        if as_tensor:
            tensor = np.array(tensor)
            tensor = torch.from_numpy(tensor).float().to(paths.device)
            return tensor.unsqueeze(1)
        else:
            return np.array(tensor)

    def merge(self, other_topologies: "Topologies") -> "Topologies":
        self.topology_sets.extend(other_topologies.topology_sets)
        self.length += other_topologies.length
        return self

def get_topology_sets_from_sources(sources):
    topology_sets = Topologies(source=sources[0])
    for source in sources[1:]:
        topology_sets.merge(Topologies(source=source))
    return topology_sets

if __name__ == "__main__":
    ## Examine
    topology_sets = get_topology_sets_from_sources(sources=[Source.TRAIN_MASKED_1, Source.TRAIN_MASKED_0])
    random_indices = np.random.choice(range(topology_sets.length), 12)

    # Topologies
    topologies = []
    for i in random_indices:
        topology = topology_sets.get_data(i).topology
        topologies.append(topology)
    # up.plot_n_topologies(topologies)

    # Constraints
    constraints = []
    constraint_type = Constraint.H_BC
    for i in random_indices:
        constraint = topology_sets.get_data(i).constraints[constraint_type]
        if constraint is not None:
            constraints.append(constraint)
    up.plot_n_topologies(constraints)

    ## Generate new dataset of masked training topologies
    # num_sets = 3
    #
    # topology_sets = Topologies(source=Source.TRAIN)
    # constraint_types = [Constraint.H_BC, Constraint.V_BC, Constraint.H_LOAD, Constraint.V_LOAD, Constraint.VOL_FRAC]
    # print("generating")
    # for i in range(num_sets):
    #     indices = list(range(topology_sets.length))
    #     in_data = topology_sets.get_topology_tensor(indices, mask=True)
    #     np.save(paths.HOME_PATH + "data/" + "masked_topologies_train_" + str(i) + ".npy", in_data)
    #
    #     constraints = []
    #     constraints_dict = {}
    #     for constraint_type in constraint_types:
    #         sub_constraints = topology_sets.get_constraints(indices, mask=True, constraint_type=constraint_type, sparse=True)
    #         constraints_dict[constraint_type] = sub_constraints
    #     for j in indices:
    #         constraints.append([constraints_dict[Constraint.H_BC][j],
    #                             constraints_dict[Constraint.V_BC][j],
    #                             constraints_dict[Constraint.H_LOAD][j],
    #                             constraints_dict[Constraint.V_LOAD][j],
    #                             constraints_dict[Constraint.VOL_FRAC][j],
    #                             ])
    #     with open(paths.HOME_PATH + "data/" + "masked_constraints_train_" + str(i) + ".pickle", 'wb') as f:
    #         pickle.dump(constraints, f)

