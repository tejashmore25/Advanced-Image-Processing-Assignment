import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def train_model(model, dataloader, loss_fn, optimizer, device, EPOCHS = 10):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        bar = tqdm(dataloader)

        for images, masks in bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"""Epoch: {epoch + 1}/{EPOCHS} | Loss: {epoch_loss/len(dataloader):.3f}""")
    return model

def iou_batch(pred_logits, targets, num_classes):
    preds = torch.argmax(pred_logits, dim = 1)
    correct_pxl = (preds == targets).sum().item()
    total_pxl = targets.numel()
    intersections = torch.zeros(num_classes, device = pred_logits.device)
    unions = torch.zeros(num_classes, device = pred_logits.device)

    for cls in range(num_classes):
        pred_idx = (preds == cls)
        target_idx = (targets == cls)

        intersections[cls] = (pred_idx & target_idx).sum()
        unions[cls] = (pred_idx | target_idx).sum()
    return correct_pxl, total_pxl, intersections, unions

def compute_IOU(total_correct, total_pxl, total_intersections, total_unions):
    pixel_acc = total_correct / total_pxl if total_pxl > 0 else 0.0

    iou_pc = []
    num_classes = len(total_unions)
    for cls in range(num_classes):
        intersection = total_intersections[cls].item()
        union = total_unions[cls].item()
        if union > 0:
            iou_pc.append(intersection / union)

    miou = sum(iou_pc) / len(iou_pc) if iou_pc else 0.0
    return pixel_acc, miou

def evaluation(model, dataloader, device, num_classes = 2):
    model.to(device)
    model.eval()

    total_correct, total_pxl = 0,0
    total_int, total_union = torch.zeros(num_classes, device = device), torch.zeros(num_classes, device = device)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            batch_c, batch_p, batch_i, batch_u = iou_batch(logits, masks, num_classes)
            total_correct += batch_c
            total_pxl += batch_p
            total_int += batch_i
            total_union += batch_u
    
    accuracy, miou = compute_IOU(total_correct, total_pxl, total_int, total_union)
    return accuracy, miou

def visualize(models, dataloader, device, num_images = 1):
    inverse_transform = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    preds = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            logits = model(images)
            preds[name] = torch.argmax(logits, dim = 1).cpu()

    images = images.cpu()
    masks = masks.cpu()

    num_cols = 2 + len(models)
    fig, axes = plt.subplots(num_images, num_cols, figsize = (4 * num_cols, 4 * num_images))

    if num_images == 1:
        axes = np.expand_dims(axes, axis = 0)
    
    for i in range(min(num_images, len(images))):
        img = inverse_transform(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(masks[i].numpy(), cmap='gray')
        axes[i, 1].set_title("True Segmented Image")
        axes[i, 1].axis('off')

        col_idx = 2
        for name, pred in preds.items():
            axes[i, col_idx].imshow(pred[i].numpy(), cmap = 'magma')
            axes[i, col_idx].set_title(f"Segmented: {name}")
            axes[i, col_idx].axis('off')
            col_idx += 1
    plt.tight_layout()
    plt.show()





    
