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

from nsbo.gaussian_process import fit_mll, get_gp
from nsbo.acquisition import AugmentedExpectedImprovement, get_best_fx, get_best_x
from envs.params import BENCHMARKING_REPETITION, RANDOM_SEED, CONF_PATH
from envs.params import NOISE_PARAM as n

class NSBO(Bounce):
    def __init__(self,
                 benchmark: Benchmark,
                 bin: int = 2,
                 n_init: int = 10,
                 initial_target_dimensionality: int = 5,
                 max_eval: int = 50,
                 max_eval_until_input: int = 45,
                #  noise_mode: int = 1,
                 noise_threshold: float = 1,
                 acquisition: str = 'ei',
                 ):
    
        self.benchmark = benchmark
        # self.noise_mode = noise_mode
        self.noise_threshold = noise_threshold
        self.acquisition = acquisition
        self.effective = True if self.acquisition == 'aei' else False
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
                
        results_dir = 'test_results' if self.benchmark.env.debugging else 'results'
        
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
                         results_dir=results_dir
                         )
        
        f = open(os.path.join(self.results_dir, 'workload.txt'), 'w')
        f.writelines(f"{self.benchmark.env.workload} {self.benchmark.env.workload_size}")
        f.close()
        
        self.fx_repeated = torch.empty(0, BENCHMARKING_REPETITION, dtype=self.dtype, device=self.device)
        self.x_repeated = torch.empty(
            0, self.benchmark.representation_dim, dtype=self.dtype, device=self.device
        )

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
            ).to(self.device)
            match parameter_type:
                case ParameterType.BINARY:
                    _x_init = sample_binary(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                        seed=RANDOM_SEED,
                    )
                case ParameterType.CONTINUOUS:
                    _x_init = sample_continuous(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                        seed=RANDOM_SEED,
                    )
                ##########--------JIEUN--------##########
                case ParameterType.NUMERICAL:
                    _x_init = sample_numerical(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                        seed=RANDOM_SEED,
                    )
                #########################################
                case ParameterType.CATEGORICAL:
                    _x_init = sample_categorical(
                        number_of_samples=self.number_initial_points,
                        bins=bins_of_type,
                        seed=RANDOM_SEED,
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
        # #########################################
        # if self.noise_mode == n['NOISY_OBSERVATIONS']:                  # self.noise_mode = 1
        #     x_init = x_init.repeat(BENCHMARKING_REPETITION, 1)
        # elif self.noise_mode == n['NOISE_FREE_REPEATED_BENCHMARKING']:  # self.noise_mode = 2
        #     pass
        # elif self.noise_mode == n["NOISE_FREE_REPEATED_EXPERIMENTS"]:   # self.noise_mode = 3
        #     pass
        # elif self.noise_mode == n["ADAPTIVE_NOISE"]:                    # self.noise_mode = 4
        #     pass
        # else:
        #     assert False, "Error with defining nosie mode"
        # #########################################
        # if self.noise_mode:
        #     pass
        # else:
        #     # To obtain mutiple results from each configuration, considering "noisy" environments.
        #     x_init = x_init.repeat(BENCHMARKING_REPETITION, 1)
        
        x_init_up = from_1_around_origin(
            x=self.random_embedding.project_up(x_init.T).T,
            lb=self.benchmark.lb_vec,
            ub=self.benchmark.ub_vec,
        )
        # print(x_init.shape)
        print("######################################")
        print(x_init_up.shape)
        print(x_init_up[0])
        print(self.random_embedding.parameters)
        assert False
        fx_inits = None
        unique_x_init = None
        
        '''
            fx_inits = tensor([[y1_1, y1_2, y1_3], [y2_1, y2_2, y2_3], ...])
            fx_init = tensor([y1, y2, y3, y4, ...])
        '''
        fx_inits = torch.Tensor()
        
        logging.info("🎁#🎁#🎁#🎁 Start Sampling 🎁#🎁#🎁#🎁")
        for _ in range(x_init_up.size(0)): # x_init_up: [n_init, num_params]
            logging.info(f"[Sampling Iteration: {_}]")
            _fx = self.benchmark(x_init_up[_].unsqueeze(0), repeat=BENCHMARKING_REPETITION)
            
            fx_inits = torch.concat([fx_inits, _fx])
        
        fx_init = fx_inits.mean(1)
        
        self._add_data_to_tr_observations(
            xs_down=x_init, # [n, target_dim] target configs converted from original configs
            xs_up=x_init_up, # [n, original_dim] original configs
            fxs=fx_init,
            repeated_fxs=fx_inits,
            repeated_xs_down=unique_x_init,
        )
        
        self._n_evals += self.number_initial_points
        logging.info("🎁#🎁#🎁#🎁 Finished Sampling 🎁#🎁#🎁#🎁")
        
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
                noise=self.effective,
            )
            model = model.to(self.device)
            
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
                sampler = SobolQMCNormalSampler(Size([1024]), seed=RANDOM_SEED)
            else:
                # use analytical EI for batch size 1
                # acquisition_function = ExpectedImprovement(
                #     model=model, best_f=(-fx_scaled).max().item()
                # )
                if self.acquisition == 'ei':
                    acquisition_function = ExpectedImprovement(
                        model=model, best_f=(-fx_scaled).max().item()
                    )
                elif self.acquisition == 'aei':
                    acquisition_function = AugmentedExpectedImprovement(
                        model=model, 
                        best_f=get_best_fx(model, x_scaled, effective=True).item(),
                    )

            continuous_type = axus.bins_and_indices_of_type(ParameterType.CONTINUOUS) + \
                                axus.bins_and_indices_of_type(ParameterType.NUMERICAL) 
            continuous_indices = torch.tensor([ i for ( _, i ) in continuous_type])

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
                    effective=self.effective,
                )
                x_best = x_best.reshape(-1, axus.target_dim)
                
                true_center = get_best_x(
                    model=model,
                    xs=x_scaled,
                    fxs=fx_scaled,
                    noisy=True,
                    effective=self.effective,
                    )
                    
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
                    effective=self.effective,
                )
                fx_best = fx_best * std + mean
                x_best = x_best.reshape(-1, axus.target_dim)
            x_best = x_best

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
            y_nexts = None
            unique_x_init = None

            # elif self.noise_mode == n['NOISE_MEAN']:
            # _fx = self.benchmark(x_init_up[_].unsqueeze(0), repeat=BENCHMARKING_REPETITION)
            y_nexts = self.benchmark(cand_batch.unsqueeze(0), repeat=BENCHMARKING_REPETITION)
            # y_nexts = torch.concat([ 
            #     self.benchmark(cand_batch.unsqueeze(0), load=False if r > 0 else True).unsqueeze(1) 
            #     for r in range(BENCHMARKING_REPETITION)], 
            #                     dim=1).to(self.device)
            # y_nexts = torch.concat([self.benchmark(cand_batch).unsqueeze(1) for _ in range(BENCHMARKING_REPETITION)], dim=1)
            # y_nexts = torch.concat(y_nexts, dim=1)
            y_next = y_nexts.mean(1)
            
            model.eval()
            model.likelihood.eval()
            
            min_y_next = torch.min(-model.posterior(torch.vstack(xs_low_dim)).mean * std + mean).to(self.device) # [1, 1]
            
            pred_fx_by_gp = -model.posterior(x_scaled).mean * std + mean
            best_idx = pred_fx_by_gp.argmin()
            
            best_pred_fx_by_gp = pred_fx_by_gp[best_idx]
            best_real_fxs = self.fx_repeated[best_idx]
                        
            tr_state['best_fx_from_poster_mean'] = best_real_fxs.unsqueeze(0) if best_real_fxs.dim() == 1 else best_real_fxs
            best_fx = best_pred_fx_by_gp

            if min_y_next < best_fx:
                logging.info(
                    # f"✨ Iteration {self._n_evals}: {BColors.OKGREEN}New incumbent function value {y_next.min().item():.3f}{BColors.ENDC}"
                    f"[✨ Iteration {self._n_evals}] New incumbent function value {min_y_next.item():.3f}{BColors.ENDC} with {best_real_fxs}"
                )
            else:
                logging.info(
                    f"[🚀 Iteration {self._n_evals}] No improvement. Best function value {best_fx.item():.3f} with {best_real_fxs}"
                )
                
            self.save_tr_state(tr_state)    
            
            # if torch.min(y_next) < best_fx:

            
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
                xs_down=torch.vstack(xs_low_dim), # if self.noise_mode > 1 else torch.vstack(xs_low_dim).repeat(BENCHMARKING_REPETITION, 1),
                xs_up=torch.vstack(xs_high_dim), # if self.noise_mode > 1 else torch.vstack(xs_high_dim).repeat(BENCHMARKING_REPETITION, 1),
                fxs=y_next.reshape(-1), # self.fx_tr
                repeated_fxs=y_nexts,
                repeated_xs_down=unique_x_init,
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
                    logging.info("🏁 Finished")
                    # logging.info(
                    #     f"🏁 Reached full dimensionality. Restarting with new random samples."
                    # )
                    # self.split_budget = self._split_budget(
                    #     self.random_embedding.input_dim
                    # )
                    # # Reset the last split budget
                    # self._all_split_budgets[self._forecasted_tr_dim] = self.split_budget

                    # # empty tr data, does not delete the global data
                    # self._reset_local_data()

                    # # reset the trust region
                    # self.trust_region.reset()

                    # self.sample_init()
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
            
            if self.fx_repeated is not None:
                with lzma.open(os.path.join(self.results_dir, f"repeated_results.csv.xz"), "w") as f:
                    np.savetxt(f, self.fx_repeated, delimiter=",")
               
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

        best_ys = self.benchmark(best_x.unsqueeze(0), repeat=BENCHMARKING_REPETITION)
        best_ys = best_ys.flatten().cpu().numpy()

        # best_ys = []

        # for _ in range(BENCHMARKING_REPETITION):
        #     best_y = self.benchmark(best_x.unsqueeze(0), load=True if _==0 else False)
        #     best_ys.append(best_y.item())
        
        from statistics import mean, stdev
        logging.info(f"Results = {best_ys} , Mean = {mean(best_ys):.3f} (±{stdev(best_ys):.3f})")
        
        f = open(CONF_PATH, 'r')
        best_config = f.readlines() 
        logging.info(".....Best Configuration.....")
        logging.info(''.join(best_config))
        # for l in best_config:
        #     logging.info(l)
        logging.info("......................")
        
    def _add_data_to_tr_observations(
        self,
        xs_down: torch.Tensor,
        xs_up: torch.Tensor,
        repeated_xs_down: torch.Tensor,
        fxs: torch.Tensor,
        repeated_fxs: torch.Tensor,
    ):
        """
        Add data to the tr local observations and save the selected trust regions to disk.

        Args:
            xs_down: the low-dimensional points that were evaluated in the trust regions
            xs_up:  the high-dimensional points that were evaluated in the trust regions
            fxs:  the function values of the high-dimensional points that were evaluated in the trust regions

        Returns:
            None

        """
        if repeated_fxs is not None:
            self.fx_repeated = torch.cat(
                (
                    self.fx_repeated,
                    repeated_fxs.detach().cpu(),
                )
            )
        else:
            self.fx_repeated = None
        
        if repeated_xs_down is not None:
            self.x_repeated = torch.cat(
                (
                    self.x_repeated,
                    repeated_xs_down.detach().cpu(),
                )
            )
        else:
            self.x_repeated = None
                        
        self.fx_tr = torch.cat(
            (
                self.fx_tr,
                fxs.reshape(-1).detach().cpu(),
            )
        )
        self.x_tr = torch.vstack(
            (
                self.x_tr,
                xs_down.detach().cpu(),
            )
        )
        self.x_up_tr = torch.vstack(
            (
                self.x_up_tr,
                xs_up.detach().cpu(),
            )
        )

        self._add_data_to_global_observations(
            xs_down=xs_down,
            xs_up=xs_up,
            fxs=fxs,
        )

# from botorch.models.model import Model
# from torch import Tensor

# def get_best_f(model: Model, x:Tensor, alpha: float=1.0, effective: bool = False):
#     '''
#         model : Model
#         x : torch.Tensor
#         effective : bool
        
#         If effective is True, get the effective best solution.
#             ref) Huang, Deng, et al. "Global optimization of stochastic black-box systems via sequential kriging meta-models." Journal of global optimization 34 (2006): 441-466.
#     '''
#     model.eval()
#     model.likelihood.eval()
#     posterior = model.posterior(x)
#     mean = posterior.mean
#     sigma = posterior.variance.sqrt()
#     # sign = np.random.choice([-1, 1])
#     # alpha *= sign
    
#     return mean.max() if effective else (mean + sigma * alpha).max()
    

