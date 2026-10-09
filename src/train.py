import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model, loader, config, checkpoint_dir, run_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)


    run_path = os.path.join(
        run_dir,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    writer = SummaryWriter(log_dir=run_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = config["train"]["epochs"]
    optimizer_class = getattr(torch.optim, config["train"]["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["train"]["learning_rate"])

    for epoch in range(1,epochs+1):
        total_loss = 0.0
        total_graphs = 0

        model.train()

        for batch_idx, data in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            data = data.to(device)
            optimizer.zero_grad()

            # Compute loss and retrieve embeddings
            loss = model(
                data.x,
                data.edge_index,
                data.batch,
            )

            loss.backward()
            optimizer.step()

            # Log data loss with a global step counter
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Loss/train_data", loss.item(), global_step)
            # Update weights
            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs

        avg_loss = total_loss / total_graphs
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch{epoch}.pt",
        )
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
            }
        torch.save(checkpoint, checkpoint_path)
                   
    writer.close()
