class MetricsCallback(TrainerCallback):
    def __init__(self, log_every_n_steps):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Calculate and log metrics every n steps
        if state.global_step % self.log_every_n_steps == 0:
            # Assuming the loss is the log likelihood loss
            loss = state.log_history[-1]['loss'] if state.log_history else None
            if loss:
                perplexity = math.exp(loss)
                print(f"Step: {state.global_step}, Perplexity: {perplexity}")
                # Log metrics to wandb
                wandb.log({'step': state.global_step, 'perplexity': perplexity})

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        # Loop through the evaluation dataset and collect predictions and labels
        all_predictions = []
        all_labels = []

        # Get the device from the model
        device = next(model.parameters()).device

        for batch in eval_dataloader:
            # Move the batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                all_predictions.append(predictions)
                all_labels.append(labels)

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Check if distributed training is enabled
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Create lists to gather tensors from all devices
            gathered_predictions = [torch.zeros_like(all_predictions) for _ in range(dist.get_world_size())]
            gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]

            # Gather tensors from all devices
            dist.all_gather(gathered_predictions, all_predictions)
            dist.all_gather(gathered_labels, all_labels)

            # Concatenate gathered tensors
            all_predictions = torch.cat(gathered_predictions)
            all_labels = torch.cat(gathered_labels)

        # Move tensors to CPU and convert to numpy arrays
        all_predictions = all_predictions.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

        # Variables to keep track of the sum of precision, recall, and F1 scores over all dimensions
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        num_dimensions = 0

        # If the labels have more than 1 dimension, iterate over each dimension
        if len(all_labels.shape) > 1:
            num_dimensions = all_labels.shape[0]
            for i in range(num_dimensions):
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels[i], all_predictions[i], average='macro', zero_division=0)
                
                # Update the sum of precision, recall, and F1 scores
                sum_precision += precision
                sum_recall += recall
                sum_f1 += f1
        
        # Calculate the average precision, recall, and F1 scores over dimensions
        avg_precision = sum_precision / num_dimensions
        avg_recall = sum_recall / num_dimensions
        avg_f1 = sum_f1 / num_dimensions
        
        print(f"Average Evaluation - Precision: {avg_precision}, Recall: {avg_recall}, F1 Score: {avg_f1}")

        # Log metrics to wandb
        wandb.log({'eval_avg_precision': avg_precision, 'eval_avg_recall': avg_recall, 'eval_avg_f1': avg_f1})
