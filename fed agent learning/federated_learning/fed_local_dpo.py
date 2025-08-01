import torch
import copy
from trl import DPOTrainer
from .fed_local_sft import SCAFFOLD_Callback

def get_fed_local_dpo_trainer(script_args, fed_args, model, model_ref, llm_model, tokenizer, training_args, local_dataset, kl_lambda, distill_dataset):
    if fed_args.fed_alg == 'custom_kl':
        trainer = DPOTrainerWithKL(
                            llm_model=llm_model,
                            lambda_kl=kl_lambda,
                            distill_dataset=distill_dataset,
                            model=model,
                            ref_model=model_ref,
                            args=training_args,
                            beta=script_args.dpo_beta,
                            train_dataset=local_dataset,
                            tokenizer=tokenizer,
                            )
    elif fed_args.fed_alg == 'meta':
        trainer = DPOTrainerMeta(
                            llm_model=llm_model,
                            lambda_kl=kl_lambda,
                            distill_dataset=distill_dataset,
                            model=model,
                            ref_model=model_ref,
                            args=training_args,
                            beta=script_args.dpo_beta,
                            train_dataset=local_dataset,
                            tokenizer=tokenizer,
                            )
    elif (fed_args.fed_alg in ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam']) or (fed_args.fed_alg).startswith('local'):
        trainer = DPOTrainer(
                            model=model,
                            ref_model=model_ref,
                            args=training_args,
                            beta=script_args.dpo_beta,
                            train_dataset=local_dataset,
                            tokenizer=tokenizer,
                            )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer
    
class DPOTrainerWithKL(DPOTrainer):
    def __init__(self, llm_model, lambda_kl, distill_dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.llm_model = llm_model.eval()
        self.lambda_kl = lambda_kl
        self.distill_dataset = distill_dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super().compute_loss(model, inputs, return_outputs=True)
        loss_dpo, outputs = return_values

        if self.distill_dataset is None:
            return (loss_dpo, outputs) if return_outputs else loss_dpo

        # 从 distill_dataset 中抽取一个 batch
        distill_inputs = next(iter(self.distill_dataset))  # 假设是 DataLoader
        distill_inputs = {k: v.to(model.device) for k, v in distill_inputs.items()}

        with torch.no_grad():
            llm_logits = self.llm_model(**distill_inputs).logits

        student_logits = model(**distill_inputs).logits

        kl_loss = torch.nn.functional.kl_div(
            input=torch.nn.functional.log_softmax(student_logits, dim=-1),
            target=torch.nn.functional.softmax(llm_logits, dim=-1),
            reduction='batchmean'
        )

        total_loss = loss_dpo + self.lambda_kl * kl_loss

        return (total_loss, outputs) if return_outputs else total_loss
    
class DPOTrainerMeta(DPOTrainer):
    def __init__(self, llm_model, lambda_kl, distill_dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.llm_model = llm_model.eval()
        self.lambda_kl = lambda_kl
        self.distill_dataset = distill_dataset
        self.variable = torch.nn.Parameter(torch.randn(8))
        self.optimizer = torch.optim.Adam(self.variable.parameters())

    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super().compute_loss(model, inputs, return_outputs=True)
        loss_dpo, outputs = return_values

        if self.distill_dataset is None:
            return (loss_dpo, outputs) if return_outputs else loss_dpo

        # 从 distill_dataset 中抽取一个 batch
        distill_inputs = next(iter(self.distill_dataset))  # 假设是 DataLoader
        distill_inputs = {k: v.to(model.device) for k, v in distill_inputs.items()}

        with torch.no_grad():
            llm_logits = self.llm_model(**distill_inputs).logits

        student_logits = model(**distill_inputs).logits

        kl_loss = torch.nn.functional.kl_div(
            input=torch.nn.functional.log_softmax(student_logits, dim=-1),
            target=torch.nn.functional.softmax(llm_logits, dim=-1),
            reduction='batchmean'
        )

        total_loss = loss_dpo + self.lambda_kl * kl_loss

        return (total_loss, outputs) if return_outputs else total_loss
    
    def train_meta(self, ):
        self.train()
        inputs = next(iter(self.train_dataset))
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.mul(self.variable, logits)
        loss = - torch.sum(torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_(self, ):
        self.lambda_kl = 0
        self.train()

class GlobalLLMTrainer(DPOTrainer):
    def __init__(self, client_models, lambda_kl, client_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.client_models = [cm.eval() for cm in client_models]  # π_i
        self.lambda_kl = lambda_kl
        self.client_weights = client_weights or [1.0 / len(client_models)] * len(client_models)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward global model π(y|x)
        logits_pi = model(**inputs).logits
        log_probs_pi = torch.nn.functional.log_softmax(logits_pi, dim=-1)

        loss_total = 0.0

        # First term: client distillation loss
        for weight, client_model in zip(self.client_weights, self.client_models):
            with torch.no_grad():
                logits_client = client_model(**inputs).logits
                probs_client = torch.nn.functional.softmax(logits_client, dim=-1)

            kl = torch.nn.functional.kl_div(probs_client, log_probs_pi, reduction='batchmean')
            loss_total += weight * kl

        # Second term: KL to reference model
        with torch.no_grad():
            logits_ref = self.ref_model(**inputs).logits
            probs_ref = torch.nn.functional.softmax(logits_ref, dim=-1)

        kl_to_ref = torch.nn.functional.kl_div(log_probs_pi, probs_ref, reduction='batchmean')
        loss_total += self.lambda_kl * kl_to_ref

        return (loss_total, None) if return_outputs else loss_total
