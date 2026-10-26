import numpy as np
import math
from mine import mine_fa, mine_fa_td, mine, mine_td


class fsnid_selection:
    def __init__(self, features, targets, num_iterations=30, model_type=None, mi_ordering_bool=True):
        self.features = features
        self.targets = targets
        self.num_iterations = num_iterations
        self.model_type = model_type
        self.mi_ordering_bool = mi_ordering_bool
        self.nm_upper_bound = self.null_model()

    def run_main(self):
        feats = list(range(self.features.shape[1]))
        if self.mi_ordering_bool:
            iterable_feats = self.mi_ordering()
        else:
            iterable_feats = list(range(self.features.shape[1]))

        print(f"Initial feature count: {len(feats)}")
        print(f"MI Ordering: {iterable_feats}")
        print(f"Null Model Threshold: {self.nm_upper_bound:.6f}")

        current_full_mine = self.mine(self.features[:, feats], self.targets)
        n_expts = current_full_mine.shape[0] if len(current_full_mine.shape) > 1 else 2

        for feat in iterable_feats:
            temp_indexes = [item for item in feats if item != feat]
            if len(feats) > 1:
                sub_mine = self.mine(self.features[:, temp_indexes], self.targets)
                arr = current_full_mine - sub_mine
            else:
                arr = current_full_mine

            lower_bound = arr.mean(axis=0)[-1] - 2 * (arr.std(axis=0)[-1] / math.sqrt(n_expts))
            if lower_bound < self.nm_upper_bound and len(temp_indexes) > 0:
                feats = temp_indexes
                current_full_mine = sub_mine
                print(f"Feature {feat:2d} EXCLUDED: lower bound {lower_bound:.5f} < nm {self.nm_upper_bound:.5f}")
            else:
                print(f"Feature {feat:2d} INCLUDED: lower bound {lower_bound:.5f} >= nm {self.nm_upper_bound:.5f}")

        if len(feats) == 0:
            top_feat = iterable_feats[-1]
            feats = [top_feat]
            print(f"Safeguard triggered: retained top informative feature {top_feat}")

        return feats

    def null_model(self):
        nm_arr = self.mine(np.random.random(self.targets.shape), self.targets)
        n_expts = nm_arr.shape[0] if len(nm_arr.shape) > 1 else 2
        return nm_arr.mean(axis=0)[-1] + 2 * (nm_arr.std(axis=0)[-1] / math.sqrt(n_expts))

    def mine(self, features, targets):
        if self.model_type in ['LSTM', 'GRU', 'TCN']:
            miner = mine_fa_td(
                p_dis=features,
                q_dis=targets,
                num_iterations=self.num_iterations,
                model_type=self.model_type
            )
        else:
            miner = mine_fa(
                p_dis=features,
                q_dis=targets,
                num_iterations=self.num_iterations
            )
        return miner.run()

    def mi_ordering(self):
        mis = [self.mine(self.features[:, feat], self.targets).mean(axis=0)[-1] for feat in range(self.features.shape[1])]
        sorted_indices = sorted(range(len(mis)), key=lambda i: mis[i])
        return sorted_indices
