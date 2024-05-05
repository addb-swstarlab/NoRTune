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
from bounce.util.printing import BColors
from bounce.benchmarks import Benchmark
from bounce.projection import Bin
from bounce.candidates import create_candidates_continuous, create_candidates_discrete
from botorch.acquisition import ExpectedImprovement, NoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from bounce.trust_region import TrustRegion, update_tr_state
from bounce.util.benchmark import ParameterType
from bounce.util.data_handling import (
    construct_mixed_point,
    from_1_around_origin,
    join_data,
    sample_binary,
    sample_categorical,
    sample_continuous,
    sample_numerical,
)

from incpp.test_gaussian_process import fit_mll, get_gp
from envs.params import BENCHMARKING_REPETITION

class incPP(Bounce):
    def __init__(self,
                 benchmark: Benchmark,
                 neighbor_distance: float = 0.01,
                 pseudo_point: bool = True,
                 pseudo_point_ratio: float = 1.0,
                 bin: int = 2,
                 n_init: int = 10,
                 initial_target_dimensionality: int = 5,
                 max_eval: int = 50,
                 max_eval_until_input: int = 45,
                 noise_free: bool = False,
                #  gp_mode: str = 'fixednoisegp',
                 ):
    
        self.benchmark = benchmark
        self.pseudo_point = pseudo_point
        self.pseudo_point_ratio = pseudo_point_ratio
        self.neighbor_distance = neighbor_distance
        self.noise_free = noise_free
        # self.gp_mode = gp_mode
        
        # if self.noise_free:
        #     logging.info("⚠️ CAUTION!! This is a noise-free mode!! ⚠️")
        #     self.gp_mode = 'singletaskgp'
        
        # TODO: after analyzing bins, revise here
        super().__init__(benchmark=self.benchmark, 
                         number_new_bins_on_split=bin, 
                         initial_target_dimensionality=initial_target_dimensionality,
                         number_initial_points=n_init,
                         maximum_number_evaluations=max_eval,
                         maximum_number_evaluations_until_input_dim=max_eval_until_input,
                         )
        
        f = open(os.path.join(self.results_dir, 'workload.txt'), 'w')
        f.writelines(f"{self.benchmark.env.workload} {self.benchmark.env.workload_size}")
        f.close()

    def sample_init(self):
        """
        Samples the initial points, evaluates them, and adds them to the observations.
        Increases the number of evaluations by the number of initial points.

        Returns:
            None

        """
        types_points_and_indices = {pt: (None, None) for pt in ParameterType}
        # sample initial points for each parameter type present in the benchmark
        for parameter_type in self.benchmark.unique_parameter_types:
            # find number of parameters of type parameter_type
            bins_of_type: list[Bin] = self.random_embedding.bins_of_type(parameter_type)
            indices_of_type = torch.concat(
                [
                    self.random_embedding.bins_and_indices_of_type(parameter_type)[i][1]
                    for i in range(len(bins_of_type))
                ]
            )
            match parameter_type:
                case ParameterType.BINARY:
                    _x_init = sample_binary(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                case ParameterType.CONTINUOUS:
                    _x_init = sample_continuous(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                ##########--------JIEUN--------##########
                case ParameterType.NUMERICAL:
                    _x_init = sample_numerical(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                #########################################
                case ParameterType.CATEGORICAL:
                    _x_init = sample_categorical(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                    )
                case ParameterType.ORDINAL:
                    raise NotImplementedError(
                        "Ordinal parameters are not supported yet."
                    )
                case _:
                    raise ValueError(f"Unknown parameter type {parameter_type}.")
            types_points_and_indices[parameter_type] = (_x_init, indices_of_type)

        ##########--------JIEUN--------##########
        x_init = construct_mixed_point(
            size=self.number_initial_points,
            binary_indices=types_points_and_indices[ParameterType.BINARY][1],
            continuous_indices=types_points_and_indices[ParameterType.CONTINUOUS][1],
            numerical_indices=types_points_and_indices[ParameterType.NUMERICAL][1],
            categorical_indices=types_points_and_indices[ParameterType.CATEGORICAL][1],
            ordinal_indices=types_points_and_indices[ParameterType.ORDINAL][1],
            x_binary=types_points_and_indices[ParameterType.BINARY][0],
            x_continuous=types_points_and_indices[ParameterType.CONTINUOUS][0],
            x_numerical=types_points_and_indices[ParameterType.NUMERICAL][0],
            x_categorical=types_points_and_indices[ParameterType.CATEGORICAL][0],
            x_ordinal=types_points_and_indices[ParameterType.ORDINAL][0],
        )
        #########################################
        
        if self.noise_free:
            pass
        else:
            # To obtain mutiple results from each configuration, considering "noisy" environments.
            x_init = x_init.repeat(BENCHMARKING_REPETITION, 1)
        
        x_init_up = from_1_around_origin(
            x=self.random_embedding.project_up(x_init.T).T,
            lb=self.benchmark.lb_vec,
            ub=self.benchmark.ub_vec,
        )
        fx_init = self.benchmark(x_init_up)
                
        self._add_data_to_tr_observations(
            xs_down=x_init, # [n, target_dim] target configs converted from original configs
            xs_up=x_init_up, # [n, original_dim] original configs
            fxs=fx_init,
        )

        self._n_evals += self.number_initial_points
        
    def run(self):
        """
        Runs the algorithm.

        Returns:
            None

        """

        self.sample_init()
        fx_best_stack = torch.empty(0, 1)
        
        while self._n_evals <= self.maximum_number_evaluations:
            axus = self.random_embedding
            
            x = self.x_tr
            fx = self.fx_tr
            
            # Preprocessing failed data #######################
            fx_ = fx[fx != torch.tensor(10000)]
            std_ = torch.std(fx_)
            
            fx[fx==torch.tensor(10000)] = fx_.max() + std_
            ####################################################          

            # normalize data
            mean = torch.mean(fx)
            std = torch.std(fx)
            if std == 0:
                std += 1
            fx_scaled = (fx - mean) / std
            
            x_scaled = (x + 1) / 2

            if self.device == "cuda":
                x_scaled = x_scaled.to(self.device)
                fx_scaled = fx_scaled.to(self.device)
                # fx_var_scaled = fx_var_scaled.to(self.device)

            # Select the kernel
            model, train_x, train_fx = get_gp(
                axus=axus,
                x=x_scaled,
                fx=-fx_scaled,
                # fx_var=fx_var_scaled,
                neighbor_distance=self.neighbor_distance,
                pseudo_point_mode=self.pseudo_point,
                pseudo_point_ratio=self.pseudo_point_ratio,
                # gp_mode = self.gp_mode,
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
                # acquisition_function = ExpectedImprovement(
                #     model=model, best_f=(-fx_scaled).max().item()
                # )
                if self.noise_free:
                    acquisition_function = ExpectedImprovement(
                        model=model, best_f=(-fx_scaled).max().item()
                    ) 
                else:
                    model.eval()
                    model.likelihood.eval()
                    posterior = model.posterior(x_scaled)
                    acquisition_function = ExpectedImprovement(
                        model=model, best_f=posterior.mean.max().item()
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
                    noise_free=self.noise_free
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
                    noise_free=self.noise_free
                )
                fx_best = fx_best * std + mean
            # TODO don't use elif True here but check for the exact type
            elif True:
                # Scale the function values
                ##########--------JIEUN--------##########
                ## TODO: Unifying data types; numerical and continuous
                continuous_type = axus.bins_and_indices_of_type(ParameterType.CONTINUOUS) + \
                                    axus.bins_and_indices_of_type(ParameterType.NUMERICAL) 
                continuous_indices = torch.tensor([ i for ( _, i ) in continuous_type])
                # continuous_indices = torch.tensor(
                #     [
                #         i
                #         for b, i in axus.bins_and_indices_of_type(
                #             ParameterType.CONTINUOUS
                #         )
                #     ]
                # )
                #########################################
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
                    # true_center = x[fx.argmin()]
                    if self.noise_free:
                        true_center = x[fx.argmin()]
                    else:
                        model.eval()
                        model.likelihood.eval()
                        true_center = x[model.posterior(x_scaled).mean.argmax()]
                        
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
            # if self.noise_free:
            #     pass
            # else:
            #     logging.info("🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳")
            #     fx_best_stack = torch.vstack((fx_best_stack, fx_best))
            #     tr_state['center_posterior_mean_fx'] = fx_best_stack
            #     fx_best_clone = fx_best.clone()
            
            # self.save_tr_state(tr_state)
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
            # y_next = self.benchmark(cand_batch)
            if self.noise_free:
                y_next = self.benchmark(cand_batch)
                min_y_next = torch.min(y_next)
            else:
                # cand_batch: torch.Tensor == xs_high_dim: List([torch.Tensor]) -> [n, representation_dim]
                y_next = self.benchmark(cand_batch.repeat(BENCHMARKING_REPETITION, 1)) # [n*BR, 1]
                
                model.eval()
                model.likelihood.eval()
                min_y_next = torch.min(-model.posterior(torch.vstack(xs_low_dim)).mean * std + mean) # [1, 1]

            # *************************************************************** #
            # TODO: in noisy environments, how can I compare them...?
            # best_fx = self.fx_tr.min()
            if self.noise_free:
                best_fx = self.fx_tr.min()
            else:
                model.eval()
                model.likelihood.eval()
                best_idx = (- model.posterior(x_scaled).mean * std + mean).argmin()
                best_fx = self.fx_tr[best_idx]
                logging.info("🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳🍳")
                # fx_best_stack = torch.vstack((fx_best_stack, best_fx))
                # logging.info(fx_best_stack)
                tr_state['best_fx_from_poster_mean'] = best_fx if best_fx.dim() > 0 else best_fx.unsqueeze(0)
                # fx_best_clone = fx_best.clone()
            
            self.save_tr_state(tr_state)    
            
            # if torch.min(y_next) < best_fx:
            if min_y_next < best_fx:
                logging.info(
                    # f"✨ Iteration {self._n_evals}: {BColors.OKGREEN}New incumbent function value {y_next.min().item():.3f}{BColors.ENDC}"
                    f"✨ Iteration {self._n_evals}: {BColors.OKGREEN}New incumbent function value {min_y_next.item():.3f}{BColors.ENDC}"
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
            logging.info(
                f"🔎 Adjusting trust region by factor {factor.item():.3f}. Remaining budget: {remaining_budget}"
            )
            update_tr_state(
                trust_region=self.trust_region,
                # fx_next=y_next.min(),
                # fx_incumbent=self.fx_tr.min(),
                fx_next=min_y_next,
                fx_incumbent=best_fx,
                adjustment_factor=factor,
            )

            logging.info(
                f"📏 Trust region has length {tr.length_discrete_continuous:.3f} and minium l {tr.length_min_discrete:.3f}"
            )

            self._all_split_budgets[tr_dim] = (
                self._all_split_budgets[tr_dim] - self.batch_size
            )
            self._n_evals += self.batch_size

            
            self._add_data_to_tr_observations(
                xs_down=torch.vstack(xs_low_dim) if self.noise_free else torch.vstack(xs_low_dim).repeat(BENCHMARKING_REPETITION, 1),
                xs_up=torch.vstack(xs_high_dim) if self.noise_free else torch.vstack(xs_high_dim).repeat(BENCHMARKING_REPETITION, 1),
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
               
        # with lzma.open(os.path.join(self.results_dir, f"fx_best_from_mean.csv.xz"), "a") as f:
        #     np.savetxt(f, fx_best_stack, delimiter=",")

        # self.benchmark.env.calculate_improvement_from_default(best_fx=best_fx)
        
        self.get_best_solution(model=model)
        
    def get_best_solution(self, model):
        logging.info(f"✨✨✨ Evaluating best x... # of repetitions = {BENCHMARKING_REPETITION} ✨✨✨")
        
        x_scaled = (self.x_tr + 1) / 2
        
        model.eval()
        model.likelihood.eval()
        
        best_x = self.x_up_tr[model.posterior(x_scaled).mean.argmax(), :]
        
        best_ys = []
        for _ in range(BENCHMARKING_REPETITION):
            best_y = self.benchmark(best_x.unsqueeze(0))
            best_ys.append(best_y.item())
        
        from statistics import mean, stdev
        logging.info(f"Results = {best_ys} , Mean = {mean(best_ys):.3f} (±{stdev(best_ys):.3f})")
        
        
        
        

