import torch
from .MetricItem import MetricItem


class EfficacyMetric(MetricItem):
    def __init__(self, controller):
        super().__init__(controller)
        self.parameters.need_parameter("interventions")

    def calculate_efficacy_metric(self, prompt, target_token_id, true_token_id):
        print(target_token_id)
        print(true_token_id)

        # Run Model on Prompt
        tokenizer_output = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
        tokens = tokenizer_output["input_ids"].to(self.controller.model_wrapper.device)

        raw_model_output = self.model_wrapper.model(tokens)[0].detach().clone().cpu()
        pred_token_logits = raw_model_output[0][-1]

        print(pred_token_logits.shape)
        
        return 1.0 if pred_token_logits[target_token_id] >= pred_token_logits[true_token_id] else 0.0
    
    def calculate_efficacy_from_history(self, prompt_history, target_token_ids, true_token_ids):
        assert len(prompt_history) == len(target_token_ids)
        assert len(target_token_ids) == len(true_token_ids)

        efficacy_sum = 0
        for idx in range(len(prompt_history)):
            prompt = prompt_history[idx]
            target_token_id = target_token_ids[idx]
            true_token_id = true_token_ids[idx]

            print(prompt)
            print(f"Target: {self.controller.model_wrapper.tokenizer.decode(target_token_id)}")
            print(f"True: {self.controller.model_wrapper.tokenizer.decode(true_token_id)}")

            efficacy_sum += self.calculate_efficacy_metric(prompt, target_token_id, true_token_id)
        
        return efficacy_sum / len(prompt_history)

    def pre_intervention_hook(self, prompt, additional_params=None):
        true_token_ids = []

        prompt_history = ["Elon Musk was born in the city of", "Ian Fleming was born in the city of", "Barack Obama was born in the city of"]
        target_history = ["Pretoria", "London", "Honolulu"]

        # Run Model on all History Prompts
        for history_prompt in prompt_history:
            tokenizer_output = self.model_wrapper.tokenizer(history_prompt, return_tensors="pt")
            tokens = tokenizer_output["input_ids"].to(self.controller.model_wrapper.device)

            raw_model_output = self.model_wrapper.model(tokens)[0].detach().clone().cpu()
            pred_token_logits = raw_model_output[0][-1]

            # Retrieve True Token
            max_token_id = torch.argmax(pred_token_logits).item()
            true_token_ids.append(max_token_id)

        # Add True Token to additional_parameters for function get_text_outputs
        self.parameters.parameters_retrieval_functions["true_token_ids"] = lambda: true_token_ids
        self.parameters.need_parameter("true_token_ids")

    def get_text_outputs(self, prompt, token_logits, additional_params=None):
        interventions = additional_params["interventions"]

        # No Efficacy Score calculated, if no target_str defined via Interventions
        if len(interventions) <= 0:
            return {}

        prompt_history = ["Elon Musk was born in the city of", "Ian Fleming was born in the city of",
                          "Barack Obama was born in the city of"]
        target_history = ["Pretoria", "London", "Honolulu"]

        # Get Target Tokens (First Element of each list, nested in this list) and True Tokens
        raw_target_token_ids = self.controller.model_wrapper.tokenizer(
            target_history,
            add_special_tokens=False
        )["input_ids"]
        target_token_ids = [item[0] for item in raw_target_token_ids]
        true_token_ids = additional_params["true_token_ids"]

        efficacy_score = self.calculate_efficacy_from_history(prompt_history, target_token_ids, true_token_ids)

        return {
            "Efficacy Score (History)": efficacy_score
        }
