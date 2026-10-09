import torch
import os

def collect_graph_outputs(model, loader, device):
    graph_embeddings = []
    temperatures = []
    attention_by_layer = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            graph_embedding, _, attention_weights = model.encoder(
                data.x,
                data.edge_index,
                data.batch,
                pooling="attention",
            )

            graph_embeddings.append(graph_embedding.cpu())
            temperatures.append(data.temperature.cpu())

            for layer_idx, alpha in enumerate(attention_weights):
                if len(attention_by_layer) <= layer_idx:
                    attention_by_layer.append([])

                attention_by_layer[layer_idx].append(
                    alpha.view(1, -1).cpu()
                )

    graph_embeddings = torch.cat(graph_embeddings, dim=0)

    attention_by_layer = [
        torch.cat(layer_weights, dim=0)
        for layer_weights in attention_by_layer
    ]

    temperatures = torch.cat(temperatures, dim=0)

    return (
        graph_embeddings,
        attention_by_layer,
        temperatures,
    )

def collect_node_embeddings(model, loader, device):
    node_embeddings = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            _, node_embedding, _ = model.encoder(
                data.x,
                data.edge_index,
                data.batch,
                pooling=None,
            )

            node_embeddings.append(node_embedding.cpu())

    return node_embeddings

def inference(model, loader, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    graph_embeddings, attention_weights, temperatures = collect_graph_outputs(
        model,
        loader,
        device,
    )

    node_embeddings = collect_node_embeddings(
        model,
        loader,
        device,
    )

    temperatures = temperatures.numpy()

    torch.save(
        graph_embeddings,
        os.path.join(output_dir, "embeddings.pt"),
    )

    torch.save(
        attention_weights,
        os.path.join(output_dir, "attention_weights.pt"),
    )
    torch.save(
        node_embeddings,
        os.path.join(output_dir, "node_embeddings.pt"),
    )

    with open(
        os.path.join(output_dir, "metadata.txt"),
        "w",
    ) as file:
        for temperature in temperatures:
            file.write(f"{temperature}\n")
