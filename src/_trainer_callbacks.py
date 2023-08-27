from transformers.integrations import WandbCallback
from transformers import TrainerCallback
import wandb

class CustomWandbCallback(WandbCallback):
    def __init__(self, report_every):
        super().__init__()
        self.report_every = report_every

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        if state.global_step % self.report_every == 0:
            if state.is_world_process_zero:
                last_training_log = state.log_history[-1]  # Get the last training log
                loss = last_training_log.get('loss', 'N/A')  # Get the loss from the log
                wandb.alert(
                    title=f"Step {state.global_step} report",
                    text=f"Reached {state.global_step} steps. Current loss: {loss}",
                    level=wandb.AlertLevel.INFO
                )


class CustomEvaluationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, trainer, **kwargs):
        eval_results = {}
        for key, dataset in trainer.eval_dataset_dict.items():
            output = trainer.evaluate(dataset)
            metrics = trainer.compute_metrics(output)
            eval_results[key] = metrics
        state.log_history["eval_results"] = eval_results
        return control