import wandb
import torch
import logging
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm

from src.sb3_util import RolloutBuffer, get_action_distribution_logits


log = logging.getLogger(__name__)


def compute_kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    is_discrete: bool,
) -> torch.Tensor:
    """
    Compute temperature-scaled KL divergence between student and teacher.
    
    Args:
        student_logits: Student policy logits
        teacher_logits: Teacher policy logits
        temperature: Temperature for scaling
        is_discrete: Whether action space is discrete
        
    Returns:
        KL divergence loss
    """
    if is_discrete:
        # Categorical distribution: apply temperature scaling and compute KL
        student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        
        # KL(teacher || student) = sum(teacher * log(teacher / student))
        kl = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs), dim=-1)
    else:
        # Gaussian distribution: logits are [mean, log_std]
        dim = student_logits.shape[-1] // 2
        
        student_mean = student_logits[..., :dim] / temperature
        student_log_std = student_logits[..., dim:]
        
        teacher_mean = teacher_logits[..., :dim] / temperature
        teacher_log_std = teacher_logits[..., dim:]
        
        # KL divergence between two Gaussians
        # KL(P||Q) = log(σ_Q/σ_P) + (σ_P^2 + (μ_P - μ_Q)^2) / (2σ_Q^2) - 1/2
        teacher_std = torch.exp(teacher_log_std)
        student_std = torch.exp(student_log_std)
        
        kl = (
            student_log_std - teacher_log_std +
            (teacher_std ** 2 + (teacher_mean - student_mean) ** 2) / (2 * student_std ** 2 + 1e-10) -
            0.5
        )
        kl = torch.sum(kl, dim=-1)
    
    return kl.mean()


def distill_student(
    student: BaseAlgorithm,
    rollout_buffer: RolloutBuffer,
    validation_buffer: RolloutBuffer | None,
    num_epochs: int,
    batch_size: int,
    temperature: float,
    learning_rate: float,
    is_discrete: bool,
    device: torch.device,
    rng_seed: int = 0,
    save_folder: str | None = None,
    epochs_per_checkpoint: int | None = None,
) -> BaseAlgorithm:
    """
    Train student policy via distillation from teacher rollouts.
    
    Args:
        student: Student agent to train
        rollout_buffer: Buffer containing teacher rollout data for training
        validation_buffer: Optional buffer containing teacher rollout data for validation
        num_epochs: Number of training epochs
        batch_size: Minibatch size for training
        temperature: Temperature for KL divergence scaling
        learning_rate: Learning rate for student training
        is_discrete: Whether action space is discrete
        device: PyTorch device
        rng_seed: Random seed for batch sampling
        save_folder: Optional folder path to save the final model
        epochs_per_checkpoint: Optional interval for saving checkpoints (e.g., 10 = save every 10 epochs)
        
    Returns:
        Trained student agent
    """
    from pathlib import Path
    
    rng = np.random.default_rng(rng_seed)
    optimizer = torch.optim.Adam(student.policy.parameters(), lr=learning_rate)
    
    student.policy.train()
    
    global_step = 0
    
    # Setup checkpoint directory if needed
    checkpoint_dir = None
    if save_folder is not None and epochs_per_checkpoint is not None:
        checkpoint_dir = Path(save_folder) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoints will be saved to {checkpoint_dir} every {epochs_per_checkpoint} epochs")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_losses = []
        
        # Iterate over all batches in the buffer (entire buffer used once per epoch)
        for batch in rollout_buffer.get_epoch_batches(batch_size, rng):
            obs_batch = batch["observations"]
            teacher_logits_batch = batch["logits"]
            
            # Convert teacher logits to tensor (no grad needed for teacher)
            teacher_logits_tensor = torch.as_tensor(teacher_logits_batch, device=device, dtype=torch.float32)
            
            # Get student logits with gradient tracking
            student_logits_tensor = get_action_distribution_logits(student, obs_batch, device, requires_grad=True)
            
            # Compute KL loss
            loss = compute_kl_divergence_loss(
                student_logits_tensor,
                teacher_logits_tensor,
                temperature,
                is_discrete,
            )
            
            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Log per-batch metrics to wandb
            if wandb.run is not None:
                wandb.log({
                    "distillation/batch/loss": batch_loss,
                    "distillation/global_step": global_step,
                }, step=global_step)
            
            global_step += 1
        
        avg_loss = epoch_loss / len(batch_losses)
        min_loss = min(batch_losses)
        max_loss = max(batch_losses)
        epoch_metrics = {
            "distillation/epoch/train/avg_loss": avg_loss,
            "distillation/epoch/train/min_loss": min_loss,
            "distillation/epoch/train/max_loss": max_loss,
            "distillation/epoch": epoch + 1,
        }
        
        log.info(f"Epoch {epoch + 1}/{num_epochs}, avg train loss: {avg_loss:.6f} ({len(batch_losses)} batches)")
        
        # Compute validation metrics if validation buffer provided
        if validation_buffer is not None:
            student.policy.eval()
            val_losses = []
            
            for batch in validation_buffer.get_epoch_batches(batch_size, rng):
                val_obs_batch = batch["observations"]
                val_teacher_logits_batch = batch["logits"]
                
                with torch.no_grad():
                    val_teacher_logits_tensor = torch.as_tensor(val_teacher_logits_batch, device=device, dtype=torch.float32)
                    val_student_logits_tensor = get_action_distribution_logits(student, val_obs_batch, device, requires_grad=False)
                    
                    val_loss = compute_kl_divergence_loss(
                        torch.as_tensor(val_student_logits_tensor, device=device, dtype=torch.float32),
                        val_teacher_logits_tensor,
                        temperature,
                        is_discrete,
                    )
                    val_losses.append(val_loss.item())
            
            avg_val_loss = float(np.mean(val_losses))
            min_val_loss = float(np.min(val_losses))
            max_val_loss = float(np.max(val_losses))
            epoch_metrics.update({
                "distillation/epoch/val/avg_loss": avg_val_loss,
                "distillation/epoch/val/min_loss": min_val_loss,
                "distillation/epoch/val/max_loss": max_val_loss,
            })
            
            log.info(f"Epoch {epoch + 1}/{num_epochs}, avg val loss: {avg_val_loss:.6f} ({len(val_losses)} batches)")
            
            student.policy.train()
        
        # Log per-epoch metrics to wandb
        if wandb.run is not None:
            wandb.log(epoch_metrics, step=global_step)
        
        # Save checkpoint if needed
        if checkpoint_dir is not None and (epoch + 1) % epochs_per_checkpoint == 0:
            checkpoint_path = checkpoint_dir / f"student_epoch_{epoch + 1}"
            student.save(str(checkpoint_path))
            log.info(f"Saved checkpoint to {checkpoint_path}")
    
    student.policy.eval()
    log.info("Distillation training complete")
    
    # Save final model if save_folder provided
    if save_folder is not None:
        final_path = Path(save_folder) / "student_model"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        student.save(str(final_path))
        log.info(f"Saved final student model to {final_path}")
    
    return student
