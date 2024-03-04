## Temp Model name: Spark parameter tuning using increasingly high-dimensional combinatorial and continuous embedding with pseudo points
## incPP?...

import logging
import lzma
import os.path
import numpy as np
import torch
from torch import Size
from tqdm import tqdm

from bounce.bounce import Bounce
from bounce.benchmarks import Benchmark
from bounce.candidates import create_candidates_continuous, create_candidates_discrete
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from bounce.trust_region import TrustRegion, update_tr_state
from bounce.util.benchmark import ParameterType
from bounce.util.data_handling import (
    from_1_around_origin,
    join_data,
)
from bounce.util.printing import BColors

from incpp.gaussian_process import fit_mll, get_gp_pp

class incPP(Bounce):
    def __init__(self,
                 benchmark: Benchmark,
                 neighbor_distance: float = 0.01,
                 pseudo_point: bool = True
                 ):
    
        self.benchmark = benchmark
        self.pseudo_point = pseudo_point
        self.neighbor_distance = neighbor_distance
        
        super().__init__(benchmark=self.benchmark)
        
    def run(self):
        """
        Runs the algorithm.

        Returns:
            None

        """

        self.sample_init()

        while self._n_evals <= self.maximum_number_evaluations:
            axus = self.random_embedding
            x = self.x_tr
            fx = self.fx_tr

            # normalize data
            mean = torch.mean(fx)
            std = torch.std(fx)
            if std == 0:
                std += 1
            fx_scaled = (fx - mean) / std
            x_scaled = (x + 1) / 2

            if self.device == "cuda":
                fx_scaled = fx_scaled.to(self.device)
                x_scaled = x_scaled.to(self.device)

            # Select the kernel
            model, train_x, train_fx = get_gp_pp(
                axus=axus,
                x=x_scaled,
                fx=-fx_scaled,
                neighbor_distance=self.neighbor_distance
            )

            use_scipy_lbfgs = self.use_scipy_lbfgs and (
                self.max_lbfgs_iters is None or len(train_x) <= self.max_lbfgs_iters
            )
            fit_mll(
                model=model,
                train_x=train_x,
                train_fx=-train_fx,
                max_cholesky_size=self.max_cholesky_size,
                use_scipy_lbfgs=use_scipy_lbfgs,
            )
            acquisition_function = None
            sampler = None

            if self.batch_size > 1:
                # we don't set the acquisition function here, because it needs to be redefined
                # for each batch item to be able to condition on the earlier batch items
                # note that this is the only place where we don't use the acquisition function
                sampler = SobolQMCNormalSampler(Size([1024]), seed=self._n_evals)
            else:
                # use analytical EI for batch size 1
                acquisition_function = ExpectedImprovement(
                    model=model, best_f=(-fx_scaled).max().item()
                )

            if self.benchmark.is_discrete:
                x_best, fx_best, tr_state = create_candidates_discrete(
                    x_scaled=x_scaled,
                    fx_scaled=fx_scaled,
                    model=model,
                    axus=axus,
                    trust_region=self.trust_region,
                    device=self.device,
                    batch_size=self.batch_size,
                    acquisition_function=acquisition_function,
                    sampler=sampler,
                )
                fx_best = fx_best * std + mean
            elif self.benchmark.is_continuous:
                x_best, fx_best, tr_state = create_candidates_continuous(
                    x_scaled=x_scaled,
                    fx_scaled=fx_scaled,
                    acquisition_function=acquisition_function,
                    model=model,
                    axus=axus,
                    trust_region=self.trust_region,
                    device=self.device,
                    batch_size=self.batch_size,
                    sampler=sampler,
                )
                fx_best = fx_best * std + mean
            # TODO don't use elif True here but check for the exact type
            elif True:
                # Scale the function values

                continuous_indices = torch.tensor(
                    [
                        i
                        for b, i in axus.bins_and_indices_of_type(
                            ParameterType.CONTINUOUS
                        )
                    ]
                )
                x_best = None
                for _ in tqdm(range(self.n_interleaved), desc="☯ Interleaved steps"):
                    x_best, fx_best, tr_state = create_candidates_discrete(
                        x_scaled=x_scaled,
                        fx_scaled=fx_scaled,
                        axus=axus,
                        model=model,
                        trust_region=self.trust_region,
                        device=self.device,
                        batch_size=self.batch_size,
                        x_bests=x_best,  # expects [-1, 1],
                        acquisition_function=acquisition_function,
                        sampler=sampler,
                    )
                    x_best = x_best.reshape(-1, axus.target_dim)
                    true_center = x[fx.argmin()]
                    x_best[:, continuous_indices] = true_center[continuous_indices].to(
                        device=x_best.device
                    )
                    x_best, fx_best, tr_state = create_candidates_continuous(
                        x_scaled=x_scaled,
                        fx_scaled=fx_scaled,
                        axus=axus,
                        trust_region=self.trust_region,
                        device=self.device,
                        indices_to_optimize=continuous_indices,
                        x_bests=x_best,  # expects [-1, 1]
                        acquisition_function=acquisition_function,
                        model=model,
                        batch_size=self.batch_size,
                        sampler=sampler,
                    )
                    fx_best = fx_best * std + mean
                    x_best = x_best.reshape(-1, axus.target_dim)
                x_best = x_best
            else:
                raise NotImplementedError(
                    "Only binary and continuous benchmarks are supported."
                )
            # get the GP hyperparameters as a dictionary
            self.save_tr_state(tr_state)
            minimum_xs = x_best.detach().cpu()
            minimum_fxs = fx_best.detach().cpu()

            fx_batches = minimum_fxs

            cand_batch = torch.empty(
                (self.batch_size, self.benchmark.representation_dim), dtype=self.dtype
            )

            xs_low_dim = list()
            xs_high_dim = list()

            for batch_index in range(self.batch_size):
                # Find the row (tr index) and column (batch index) of the minimum
                col = torch.where(fx_batches == fx_batches.min())[0]
                # Find the point that gave the minimum
                x_elect = minimum_xs[col[0]]
                if len(x_elect.shape) == 1:
                    # avoid transpose warnings
                    x_elect = x_elect.unsqueeze(0)
                # Add the point to the lower-dimensional observations
                xs_low_dim.append(x_elect)

                # Project the point up to the high dimensional space
                x_elect_up = from_1_around_origin(
                    self.random_embedding.project_up(x_elect.T).T,
                    lb=self.benchmark.lb_vec,
                    ub=self.benchmark.ub_vec,
                )
                # Add the point to the high-dimensional observations
                xs_high_dim.append(x_elect_up)
                # Add the point to the batch to be evaluated
                cand_batch[batch_index, :] = x_elect_up.squeeze()
                # Set the value of the minimum to infinity so that it is not selected again
                fx_batches[col[0]] = torch.inf

            # Sample on the candidate points
            y_next = self.benchmark(cand_batch)

            best_fx = self.fx_tr.min()
            if torch.min(y_next) < best_fx:
                logging.info(
                    f"✨ Iteration {self._n_evals}: {BColors.OKGREEN}New incumbent function value {y_next.min().item():.3f}{BColors.ENDC}"
                )
            else:
                logging.info(
                    f"🚀 Iteration {self._n_evals}: No improvement. Best function value {best_fx.item():.3f}"
                )

            # Calculate the estimated trust region dimensionality
            tr_dim = self._forecasted_tr_dim
            # Number of times this trust region has been selected
            # Remaining budget for this trust region
            remaining_budget = self._all_split_budgets[tr_dim]
            remaining_budget = min(
                remaining_budget, self.maximum_number_evaluations - self._n_evals
            )
            remaining_budget = max(remaining_budget, 1)
            tr = self.trust_region
            factor = (tr.length_min_discrete / tr.length_discrete_continuous) ** (
                1 / remaining_budget
            )
            factor **= self.batch_size
            factor = np.clip(factor, a_min=1e-10, a_max=None)
            logging.debug(
                f"🔎 Adjusting trust region by factor {factor.item():.3f}. Remaining budget: {remaining_budget}"
            )
            update_tr_state(
                trust_region=self.trust_region,
                fx_next=y_next.min(),
                fx_incumbent=self.fx_tr.min(),
                adjustment_factor=factor,
            )

            logging.debug(
                f"📏 Trust region has length {tr.length_discrete_continuous:.3f} and minium l {tr.length_min_discrete:.3f}"
            )

            self._all_split_budgets[tr_dim] = (
                self._all_split_budgets[tr_dim] - self.batch_size
            )
            self._n_evals += self.batch_size

            self._add_data_to_tr_observations(
                xs_down=torch.vstack(xs_low_dim),
                xs_up=torch.vstack(xs_high_dim),
                fxs=y_next.reshape(-1),
            )

            # Splitting trust regions that terminated
            if self.trust_region.terminated:
                if self.random_embedding.target_dim < self.benchmark.representation_dim:
                    # Full dim is not reached yet
                    logging.info(f"✂️ Splitting trust region")

                    index_mapping = self.random_embedding.split(
                        self.number_new_bins_on_split
                    )

                    # move data to higher-dimensional space
                    self.x_tr = join_data(self.x_tr, index_mapping)
                    self.x_global = join_data(self.x_global, index_mapping)

                    self.trust_region = TrustRegion(
                        dimensionality=self.random_embedding.target_dim
                    )
                    if self.tr_splits < self._n_splits:
                        self.tr_splits += 1

                    self.split_budget = self._split_budget(
                        self.initial_target_dimensionality
                        * (self.number_new_bins_on_split + 1) ** self.tr_splits
                    )
                else:
                    # Full dim is reached
                    logging.info(
                        f"🏁 Reached full dimensionality. Restarting with new random samples."
                    )
                    self.split_budget = self._split_budget(
                        self.random_embedding.input_dim
                    )
                    # Reset the last split budget
                    self._all_split_budgets[self._forecasted_tr_dim] = self.split_budget

                    # empty tr data, does not delete the global data
                    self._reset_local_data()

                    # reset the trust region
                    self.trust_region.reset()

                    self.sample_init()
        with lzma.open(os.path.join(self.results_dir, f"results.csv.xz"), "w") as f:
            np.savetxt(
                f,
                np.hstack(
                    (
                        self.x_up_global.detach().cpu().numpy(),
                        self.fx_global.detach().cpu().numpy().reshape(-1, 1),
                    )
                ),
                delimiter=",",
            )

        

