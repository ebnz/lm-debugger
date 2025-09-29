import torch
from .MetricItem import MetricItem


class EfficacyMetric(MetricItem):
    """Measures the Efficacy of the LLM (incl. Interventions) on a config-defined Dataset"""
    def __init__(self, controller):
        super().__init__(controller)

    def calculate_efficacy_metric(self, prompt, target_token_id, true_token_id):
        # Run Model on Prompt
        tokenizer_output = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens = tokenizer_output["input_ids"].to(self.model_wrapper.device)

        raw_model_output = self.model_wrapper.model(tokens)[0].detach().clone().cpu()
        pred_token_logits = raw_model_output[0][-1]

        return 1.0 if pred_token_logits[target_token_id] > pred_token_logits[true_token_id] else 0.0
    
    def calculate_efficacy_from_history(self, prompt_history, target_token_ids, true_token_ids):
        assert len(prompt_history) == len(target_token_ids)
        assert len(target_token_ids) == len(true_token_ids)

        efficacy_sum = 0
        for idx in range(len(prompt_history)):
            prompt = prompt_history[idx]
            target_token_id = target_token_ids[idx]
            true_token_id = true_token_ids[idx]

            efficacy_sum += self.calculate_efficacy_metric(prompt, target_token_id, true_token_id)
        
        return efficacy_sum / len(prompt_history)

    def pre_intervention_hook(self, prompt, **kwargs):
        true_token_ids = []

        # Run Model on all History Prompts
        for dataset_prompt in self.config.dataset.prompts:
            tokenizer_output = self.model_wrapper.tokenizer(dataset_prompt, return_tensors="pt")
            tokens = tokenizer_output["input_ids"].to(self.model_wrapper.device)

            raw_model_output = self.model_wrapper.model(tokens)[0].detach().clone().cpu()
            pred_token_logits = raw_model_output[0][-1]

            # Retrieve True Token
            max_token_id = torch.argmax(pred_token_logits).item()
            true_token_ids.append(max_token_id)

        # Return true_token_ids, will be parameter pre_intervention_hook_rv in get_text_outputs
        return true_token_ids

    def get_text_outputs(self, prompt, token_logits, pre_hook_rv=None, **kwargs):
        # Get Target Tokens (First Element of each list, nested in this list) and True Tokens
        raw_target_token_ids = self.model_wrapper.tokenizer(
            self.config.dataset.targets,
            add_special_tokens=False
        )["input_ids"]
        target_token_ids = [item[0] for item in raw_target_token_ids]
        true_token_ids = pre_hook_rv

        efficacy_score = self.calculate_efficacy_from_history(self.config.dataset.prompts, target_token_ids, true_token_ids)

        return {
            "Efficacy Score (Dataset)": efficacy_score
        }
