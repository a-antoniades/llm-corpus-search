from transformers import TrainerCallback
import wandb

class WandbCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.alert_sent = False

    def on_evaluate(self, args, state, control):
        n_alert_epoch = 5
        if state.global_step > n_alert_epoch and not self.alert_sent:
            # Obtain the training loss
            loss = state.log_history[-1]['loss']

            # Calculate the validation loss
            trainer = control.trainer
            eval_dataloader = trainer.get_eval_dataloader()
            output = trainer.prediction_loop(
                eval_dataloader, 
                description="Validation"
            )
            validation_loss = output.metrics['eval_loss']

            # Generate the text for the alert
            text = f"Step: {state.global_step}, Loss: {loss}, Validation Loss: {validation_loss}"

            # Send the alert
            wandb.alert(
                title=f"Approximately {n_alert_epoch} Steps finished", 
                text=text
            )

            # Mark the alert as sent
            self.alert_sent = True