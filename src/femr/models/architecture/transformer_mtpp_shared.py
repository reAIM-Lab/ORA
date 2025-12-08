class MTPPSharedTaskHead(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            pretraining_task_info: List[Tuple[str, float]],
            non_numerical_task: List[str],
            numerical_task: List[str],
            value_bins: Union[np.ndarray, List],
            non_numerical_task_time_bins: Union[np.ndarray, List],
            numerical_task_time_bins: Union[np.ndarray, List],
            final_layer_size: int,
            linear_interpolation: bool = False,
            # num_value_bins: int = 10,
    ):
        super().__init__()

        self.use_uncertainty = True
        self.logvars = nn.Parameter(torch.zeros(2))   # [non, num]
        self._logvar_clamp = (-3.0, 3.0)   

        # Handle both numpy array and list (from config deserialization)
        if not isinstance(value_bins, np.ndarray):
            value_bins = np.array(value_bins)
        if not isinstance(non_numerical_task_time_bins, np.ndarray):
            non_numerical_task_time_bins = np.array(non_numerical_task_time_bins)
        if not isinstance(numerical_task_time_bins, np.ndarray):
            numerical_task_time_bins = np.array(numerical_task_time_bins)
            
        assert value_bins.shape[0] == numerical_task_time_bins.shape[0], "no matching dimension for number of numerical tasks"
        assert non_numerical_task_time_bins.shape[1] == numerical_task_time_bins.shape[1], "no matching dimension for time bins"

        self.num_time_bins = numerical_task_time_bins.shape[1] - 1  # Each task has same number of bins
        # self.time_bins = time_bins  # Store time bins for potential debugging
        self.non_numerical_task_time_bins = non_numerical_task_time_bins
        self.numerical_task_time_bins = numerical_task_time_bins
        self.num_value_bins = value_bins.shape[1] - 1
        self.value_bins = value_bins
        
        # Get task counts from input data structure - we'll determine from batch data
        # For now, assume we'll get the correct task counts from the batch
        self.numerical_tasks = numerical_task
        self.non_numerical_tasks = non_numerical_task

        self.final_layer_size = final_layer_size
        
        # Original layer for non-numerical codes: time prediction only
        # print(f"hidden size is {hidden_size}")
        # self.non_numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * final_layer_size)
        self.final_layer = nn.Linear(hidden_size, self.num_time_bins*final_layer_size)

        self.non_numerical_task_layer = nn.Linear(self.final_layer_size, len(self.non_numerical_tasks))  # All tasks initially
        
        # Additional layer for numerical codes: time × value prediction
        # self.numerical_final_layer = nn.Linear(hidden_size, self.num_time_bins * self.num_value_bins * final_layer_size)
        self.numerical_task_layer = nn.Linear(self.final_layer_size, self.num_value_bins*len(self.numerical_tasks))
        # We'll determine the actual number of numerical tasks from batch data
        # self.numerical_task_layer = nn.Linear(self.final_layer_size, num_numerical_tasks).to(features.device)
        
        self.softmax = nn.Softmax(dim=1)
        self.norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)
        # self.numerical_norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size)
        # self.non_numerical_norm = femr.models.rmsnorm.RMSNorm(self.final_layer_size*self.num_value_bins)
        # self.linear_interpolation = linear_interpolation
        self.H_t  = math.log(self.num_time_bins)
        self.H_tv = math.log(self.num_time_bins * self.num_value_bins)
        
        # Set initial biases
        self.non_numerical_task_layer.bias.data.zero_()
        self.numerical_task_layer.bias.data.zero_()

    def forward(self, features: torch.Tensor, batch: Mapping[str, torch.Tensor], return_logits=False):
        """
        Efficient MOTOR Task Head Forward Pass
        
        Architecture follows original approach:
        1. Non-numerical codes: prediction_points × time_bins × non_numerical_tasks  
        2. Numerical codes: prediction_points × time_bins × value_bins × numerical_tasks
        3. Probability sum over time_bins × value_bins = 1 for numerical codes
        4. New loss = mean loss for non-numerical + mean loss for numerical codes
        """
        eps = 1e-9
        total_loss = torch.tensor(0.0, device=features.device)
        total_count = 0
        result = {}

        # print("using task shared")
        # ========== NON-NUMERICAL CODES (Original Logic) ==========
            # Time-dependent features for non-numerical codes
        time_independent_features = self.final_layer(features).reshape(
            features.shape[0], self.num_time_bins, self.final_layer_size
        )
        
        # Non-numerical task logits: [prediction_points, time_bins, non_numerical_tasks]
        task_logits = self.non_numerical_task_layer(self.norm(time_independent_features))
        time_dependent_logits = self.softmax(task_logits)

        # Debug: Check for NaN/inf values  
        if torch.any(torch.isnan(time_dependent_logits)) or torch.any(torch.isinf(time_dependent_logits)):
            print(f"ERROR: NaN/inf detected in time_dependent_logits")
            raise ValueError("NaN/inf detected in time_dependent_logits")
        
        # Verify probability sums to 1 over time bins
        # prob_sums = torch.sum(time_dependent_logits, dim=1)  # Sum over time bins
        # assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-2), f"Probability sums: {prob_sums[0]}"
        
        # Calculate CDF for survival analysis
        cdf = torch.cumsum(time_dependent_logits, dim=1)
        integrated_logits = torch.cat([torch.ones_like(time_dependent_logits[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
        
        # Add numerical stability
        time_dependent_logits_stable = torch.clamp(time_dependent_logits, min=eps)
        integrated_logits_stable = torch.clamp(integrated_logits, min=eps)
        
        # Verify input shapes match expectations
        assert batch["non_numerical_is_event"].shape == time_dependent_logits.shape, \
            f"Shape mismatch: time_dependent_logits {time_dependent_logits.shape} vs is_event {batch['non_numerical_is_event'].shape}"
        
        # Validate exactly one bin per prediction-task combination
        # labels_sum = torch.sum(batch["non_numerical_is_event"], dim=1)
        # assert torch.all(labels_sum == 1), f"Expected exactly 1 True bin per prediction-task combination"
        
        # Loss calculation for non-numerical codes
        marked_bins = batch["non_numerical_is_event"]
        is_censored_expanded = batch["non_numerical_is_censored"].unsqueeze(1).expand(-1, self.num_time_bins, -1)
        
        # Select appropriate probabilities
        selected_probs = torch.where(
            is_censored_expanded,
            integrated_logits_stable,  # Use 1-F() for censoring
            time_dependent_logits_stable  # Use f() for events
        )
        
        # Calculate loss only for marked bins
        loss_values = torch.where(
            marked_bins,
            torch.log(selected_probs),
            torch.zeros_like(selected_probs)
        )
        
        num_marked_bins = torch.sum(marked_bins)
        if num_marked_bins > 0:
            L_non = -torch.sum(loss_values) / (num_marked_bins)  # normalized by log B_t
            total_loss += -torch.sum(loss_values)
            total_count += num_marked_bins
        else:
            L_non = torch.tensor(0.0, device=features.device)

        
        if return_logits:
            result["time_dependent_logits"] = time_dependent_logits
        
        # ========== NUMERICAL CODES (New Matrix-wise Logic) ==========  
        if "numerical_time_event_bin" in batch:
            # Get dimensions from batch data 
            # print(batch.keys())
            time_event_in_bin = batch["numerical_time_event_bin"]  # [pred_points, time_bins, numerical_tasks]
            time_censor_in_bin = batch["numerical_time_censor_bin"]  # [pred_points, time_bins, numerical_tasks]
            value_event_in_bin = batch["numerical_value_event_bin"]  # [pred_points, value_bins, numerical_tasks]
            # numerical_is_censored = batch["numerical_is_censored"]  # [pred_points, numerical_tasks]
            
            batch_size, _, num_numerical_tasks = time_event_in_bin.shape
            # Time-value features: [pred_points, time_bins, value_bins, final_layer_size]
            # numerical_features = self.final_layer(features).reshape(
            #     batch_size, self.num_time_bins*self.num_value_bins, self.final_layer_size
            # )
            
            # Numerical task logits: [pred_points, time_bins*value_bins, numerical_tasks]
            numerical_task_logits = self.numerical_task_layer(self.norm(time_independent_features)).reshape(
                batch_size, self.num_time_bins*self.num_value_bins,len(self.numerical_tasks)
            )

            # Apply value_valid_mask to mask out invalid value bins before softmax
            # value_valid_mask: [numerical_tasks, value_bins] -> need to expand to [time_bins*value_bins, numerical_tasks]
            device = numerical_task_logits.device
            value_valid_mask = torch.tensor(batch["value_valid_mask"],device=device,dtype=torch.bool)

            if value_valid_mask.shape[1] - value_event_in_bin.shape[1] == 1:
                value_valid_mask = value_valid_mask[:,1:]
            
            # Expand valid mask to match value_event_in_bin shape: [1, value_bins, numerical_tasks]
            valid_mask_expanded = value_valid_mask.T.unsqueeze(0) 
            # Check that events only occur in valid bins: invalid events = events & ~valid_mask
            invalid_events = value_event_in_bin & (~valid_mask_expanded)  # [pred_points, value_bins, numerical_tasks]
            assert not torch.any(invalid_events), f"Events in bin found in invalid value bins {invalid_events}"



            mask_tv = valid_mask_expanded.expand(self.num_time_bins,-1,-1) #[num_time_bins,value_bins, numerical_tasks]
            expanded_mask = mask_tv.reshape(1,self.num_time_bins * self.num_value_bins, num_numerical_tasks)

            # Use large negative value instead of -inf
            mask_value = -1e10  # This effectively zeros out in softmax but avoids -inf
            numerical_task_logits = numerical_task_logits.masked_fill(~expanded_mask, mask_value)

            numerical_probs_flat = self.softmax(numerical_task_logits)

            # torch.logsumexp()
            # torch.exp(torch.logsum(numerical_log_probs_flat))

            # numerical_log_probs = numerical_log_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)
            numerical_probs = numerical_probs_flat.reshape(batch_size, self.num_time_bins, self.num_value_bins, num_numerical_tasks)

            numerical_probs = numerical_probs.clamp_min(eps)

            # # assert check
            expanded_valid_mask_assert = mask_tv.unsqueeze(0).expand(batch_size,-1,-1,-1)
            numerical_probs = numerical_probs*expanded_valid_mask_assert

            assert torch.all(numerical_probs[~expanded_valid_mask_assert] <= eps), \
            "Non-zero probabilities found in invalid bins."
            assert torch.all(numerical_probs[expanded_valid_mask_assert] >= eps), \
            "zero probabilities found in valid bins."
            
            # === MATRIX-WISE LOSS COMPUTATION (NO NESTED LOOPS) ===
            
            # Case 1: Censored cases - sum over value bins, only predict time
            # Use time_censor_in_bin directly for censored events
            numerical_loss = torch.tensor(0.0, device=features.device)
            total_num_count = 0
            # event_count = torch.tensor(0.0,device=features.device)
            # censored_count = torch.tensor(0.0,device=features.device)
            L_num = torch.tensor(0.0, device=features.device)
            L_num_event = torch.tensor(0.0, device=features.device)
            L_num_cens  = torch.tensor(0.0, device=features.device)


            if torch.any(time_censor_in_bin):
                # Sum probabilities over value bins: [pred_points, time_bins, numerical_tasks]
                time_only_probs = torch.sum(numerical_probs, dim=2)
                cdf = torch.cumsum(time_only_probs, dim=1)
                integrated_logits = torch.cat([torch.ones_like(time_only_probs[:, :1, :]), 1.0 - cdf[:, :-1, :]], dim=1)
                integrated_logits_stable = torch.clamp(integrated_logits, min=eps, max=1.0-eps)

                # Censored loss using matrix operations
                # it is 1-F at censor positions otherwise
                censored_loss_values = torch.where(
                    time_censor_in_bin,
                    torch.log(integrated_logits_stable),
                    torch.zeros_like(time_only_probs)
                )
                
                censored_count = torch.sum(time_censor_in_bin)
                # if censored_count > 0:
                L_num_cens = -torch.sum(censored_loss_values) 
                numerical_loss += L_num_cens
                total_num_count += censored_count

            # Case 2: Event cases - predict both time and value using outer product
            # Construct event_bins = time_event_in_bin ⊗ value_event_in_bin (outer product)  
            if torch.any(time_event_in_bin) and torch.any(value_event_in_bin):
                # Matrix-wise outer product: [pred_points, time_bins, numerical_tasks] ⊗ [pred_points, value_bins, numerical_tasks]
                # Result: [pred_points, time_bins, value_bins, numerical_tasks]
                event_bins = time_event_in_bin.unsqueeze(2) & value_event_in_bin.unsqueeze(1)
                
                # Event loss using matrix operations  
                event_loss_values = torch.where(
                    event_bins,
                    torch.log(numerical_probs),
                    torch.zeros_like(numerical_probs)
                )
                
                event_count = torch.sum(event_bins)
                if event_count > 0:

                    L_num_event = -torch.sum(event_loss_values)
                    numerical_loss += L_num_event
                    total_num_count += event_count
            
            # ne  = event_count.float()
            # nc  = censored_count.float()
            # den = ne + nc
            if total_num_count > 0:
                L_num = numerical_loss/total_num_count
                total_count += total_num_count
                total_loss += numerical_loss
            else:
                L_num = torch.tensor(0.0, device=features.device)

            # if den > 0:
            #     w_evt = ne / den
            #     w_cen = nc / den
            #     L_num = w_evt * L_num_event + w_cen * L_num_cens
            # else:
            #     L_num = torch.tensor(0.0, device=features.device)

            if return_logits:
                result["loss_components"] = {
                    "L_non_num":       L_non.detach(),
                    "L_num_event":   L_num_event.detach(),
                    "L_num_censor":  L_num_cens.detach(),
                }
                result["value_dependent_logits"] = numerical_probs
        else:
            print("No numerical codes found in batch")

        loss = total_loss / total_count

        return loss, result