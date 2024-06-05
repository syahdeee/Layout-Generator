import dgl
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from dataset import DatasetLoader
from torchvision.ops import complete_box_iou_loss
from sklearn.model_selection import train_test_split
from build_layout_graph import dgl_coco_collate_all
from model_SGTransformer import GraphTransformerNet

dataset = DatasetLoader()
train_dataset, val_dataset = train_test_split(dataset, test_size=0.16, random_state=42)

train_dataset = dgl_coco_collate_all(train_dataset)
val_dataset = dgl_coco_collate_all(val_dataset)

n_objs = 6 
n_rels = 7   
emb_size = 768 
n_heads = 10   
n_enc_layers = 10 
pos_enc_dim = 32  

model = GraphTransformerNet(n_objs, n_rels, emb_size, n_heads, n_enc_layers, pos_enc_dim)
model


def get_labels_from_graph(graph):
    labels = []
    subgraphs = dgl.unbatch(graph)
    for subgraph in subgraphs:
        labels.append(subgraph.ndata['feat'])
        
    labels = torch.stack(labels).to(graph.device)
    return labels

def get_lap_pos_enc(graph):
    # Implementation from graphtransformer
    lap_pos_enc = graph.ndata['lap_pos_enc']
    sign_flip = torch.rand(lap_pos_enc.size(1), device=graph.device)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)
    return lap_pos_enc

def run_model(graph):
    pred_boxes = model(graph)
    pred_boxes = pred_boxes.clamp(min=0, max=1)
    return pred_boxes

def train_one_epoch(train_dataset, model, opt, scheduler):
    losses = ["box", "iou", "total"]
    epoch_losses = {k: 0 for k in losses}
    model.train()
    
    losses = training_step(train_dataset, model, opt, scheduler)
    epoch_losses = {k: v + losses[k] for k, v in epoch_losses.items()}
    epoch_losses = {k: v / len(train_dataset)
                    for k, v in epoch_losses.items()}
    return epoch_losses

def training_step(train_dataset, model, opt, scheduler):
    graph, boxes = train_dataset
    pred_boxes = run_model(graph)
    box_loss = F.mse_loss(pred_boxes, boxes)
    iou_loss = complete_box_iou_loss(pred_boxes, boxes, reduction='mean')
    
    loss = box_loss + iou_loss
    losses = {'box': box_loss.item(),"iou": iou_loss.item(), "total": loss.item()}
    
    opt.zero_grad()
    box_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    scheduler.step()

    return losses

def evaluate(epoch, model, val_dataset):
    losses_epoch, layout, graph = eval_one_epoch(val_dataset)
    for key in losses_epoch.keys():
        print(f'eval/{key}', losses_epoch[key], epoch)
    return losses_epoch

def eval_one_epoch(val_dataset):
    losses = ["box", "iou", "total"]
    epoch_losses = {k: 0 for k in losses}
    model.eval()
    layout = {'gt': [], 'pred': []}
    graph, boxes = val_dataset
    with torch.no_grad():
        pred_boxes = run_model(graph)
        box_loss = F.mse_loss(pred_boxes, boxes)
        iou_loss = complete_box_iou_loss(
            pred_boxes, boxes, reduction='mean')

        loss = box_loss + iou_loss

        losses = {'box': box_loss.item(), "iou": iou_loss.item(), "total": loss.item()}

        epoch_losses = {k: v + losses[k]
                        for k, v in epoch_losses.items()}

    layout['gt_boxes'] = boxes
    layout['pred_boxes'] = pred_boxes

    epoch_losses = {k: v / len(val_dataset)
                    for k, v in epoch_losses.items()}
    return epoch_losses, layout, graph

opt = optim.Adam(model.parameters(), lr=0.001)  
scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.4)

def train_and_evaluate(epochs, train_loader, model, opt, val_dataset, patience):
    best_loss = float('inf')
    train_losses_history = {'box': [], 'iou': [], 'total': []}
    val_losses_history = {'box': [], 'iou': [],  'total': []}
    no_improvement_count = 0

    for epoch in range(epochs):
        print(f'Pelatihan [{epoch+1} / {epochs}]')
        train_losses = train_one_epoch(train_loader, model, opt, scheduler)

        for k, v in train_losses.items():
            print(f'pelatihan/{k}_loss', v, epoch)
        for key, value in train_losses.items():
            train_losses_history[key].append(value)
            
        print(f'Validasi [{epoch+1} / {epochs}]')
        val_losses = evaluate(epoch, model, val_dataset) 
        for key, value in val_losses.items():
            val_losses_history[key].append(value)
            
        if val_losses['total'] < best_loss:
            best_loss = val_losses['total']
            torch.save(model.state_dict(), 'model_sgt_32.pth')
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"No improvement for {patience} epochs. Early stopping...")
                break
        
        print()

train_and_evaluate(300, train_dataset, model, opt, val_dataset, 10)