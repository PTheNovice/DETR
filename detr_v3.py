import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as T
import random
from transformers import DetrForObjectDetection, DetrConfig, DetrImageProcessor, DeiTFeatureExtractor, DeiTModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou



class KittiDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, classes, image_files, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.classes = classes
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = img_name.replace('.png', '.txt')

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        # print(f"Original image size: {image.size}")

        target = {}
        boxes = []
        labels = []

        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                obj_type = fields[0]
                if obj_type not in self.classes:
                    continue

                bbox = list(map(float, fields[4:8]))
                boxes.append(bbox)
                labels.append(self.classes.index(obj_type))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['orig_size'] = orig_size

        if self.transforms:
            # image = self.transforms(image)
            image, target = self.transforms(image, target)

        return image, target

def get_transforms(train=True):

    def transform(image, target):
        image = T.Resize((224, 224))(image)
        new_width, new_height = image.size

        if 'boxes' in target:
            boxes = target['boxes']
            # boxes[:, [0,2]] = boxes[:, [0,2]]*(new_width/width)
            # boxes[:, [1,3]] = boxes[:, [1,3]]*(new_height/height)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=new_width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=new_height)

            target['boxes'] = boxes

        image = T.ToTensor()(image)
        image = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(image)
        # image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        return image, target
    return transform

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = list(image for image in images)
    targets = list(target for target in targets)
    return images, targets

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(img_tensor.device)
    img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]
    return img_tensor


def detr():
    image_dir = "./data/data_object_image_2/training/image_2/"
    label_dir = "./data/data_object_label_2/training/label_2/"
    label_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    batch = 4

    random.seed(7)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)

    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    train_dataset = KittiDetectionDataset(image_dir, label_dir, label_classes, train_files, get_transforms(True))
    val_dataset = KittiDetectionDataset(image_dir, label_dir, label_classes, val_files, get_transforms(False))

    # DeTR Model
    num_classes = len(label_classes) + 1
    config = DetrConfig.from_pretrained('facebook/detr-resnet-50', num_labels=num_classes, ignore_mismatched_sizes=True)
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', config=config, ignore_mismatched_sizes=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Device used: ", device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=.00005)
    num_epochs = 10

    metric_fn = MeanAveragePrecision(iou_type='bbox')

    # Training Loop
    train_loss_ce_history = []
    train_loss_giou_history = []
    val_loss_ce_history = []
    val_loss_giou_history = []
    train_iou_history = []
    val_iou_history = []
    train_acc_history = []
    val_acc_history = []
    train_map_history = []
    val_map_history = []
    train_all_predictions = []
    train_bbox_acc_history = []
    val_bbox_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total_loss_ce = 0
        total_loss_bbox = 0
        total_loss_giou = 0
        batches = 0
        total_iou = 0.0
        iou_count = 0
        correct_preds = None
        total_preds = None
        image_id = 0
        total_bbox_accuracy = 0
        bbox_accuracy_count = 0
        for images, targets in tqdm(train_dataloader):
            pixel_values = torch.stack([img for img in images]).to(device)
            correct_preds = 0
            total_preds = 0

            labels = []
            for target in targets:

                boxes = target['boxes'].to(device)
                # Convert [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
                cxcywh_boxes = torch.zeros_like(boxes)
                cxcywh_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                cxcywh_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                cxcywh_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                cxcywh_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                # Normalize coordinates to [0, 1] based on image size (224x224)
                cxcywh_boxes[:, [0, 2]] /= 224.0
                cxcywh_boxes[:, [1, 3]] /= 224.0

                labels.append({
                    'class_labels': target['labels'].to(device),
                    'boxes': cxcywh_boxes
                })
                # labels.append({
                #     'class_labels': target['labels'].to(device),
                #     'boxes': target['boxes'].to(device)
                # })
            outputs = model(pixel_values=pixel_values, labels=labels)

            # For Training IoU, mAP, acc
            # Post-process training outputs (like in validation)
            processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
            target_sizes = torch.tensor([img.shape[-2:] for img in pixel_values]).to(device)
            results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)

            for idx, result in enumerate(results):
                gt_boxes = []
                pred_boxes = []
                pred_dict = []
                gt_dict = []

                for box, label in zip(targets[idx]['boxes'], targets[idx]['labels']):
                    cx, cy, w, h = box
                    xmin = cx - w / 2
                    xmax = cx + w / 2
                    ymin = cy - h / 2
                    ymax = cy + h / 2
                    gt_boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                    gt_dict.append({
                        "boxes": torch.tensor([[xmin, ymin, xmax, ymax]]).to(device),
                        "labels": torch.tensor([label]).to(device)
                    })

                for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                    cx, cy, w, h = box
                    xmin = cx - w / 2
                    xmax = cx + w / 2
                    ymin = cy - h / 2
                    ymax = cy + h / 2
                    pred_boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                    pred_dict.append({
                        "boxes": torch.tensor([[xmin, ymin, xmax, ymax]]).to(device),
                        "scores": torch.tensor([score]).to(device),
                        "labels": torch.tensor([label]).to(device)
                    })
                    width = xmax - xmin
                    height = ymax - ymin
                    train_all_predictions.append({
                        "image_id": image_id, "category_id": label.item(),
                        "bbox": [xmin, ymin, width, height], "score": score.item()})

                if gt_boxes and pred_boxes:
                    iou_matrix = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes))
                    mean_iou = iou_matrix.max(dim=1).values.mean().item()
                    total_iou += mean_iou
                    iou_count += 1

                    ious = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes)).diag()  # take diagonal (matched pairs)
                    iou_thresh = 0.5
                    correct_bbox = (ious > iou_thresh).sum().item()
                    total_bbox = len(ious)
                    bbox_accuracy = correct_bbox / total_bbox

                    total_bbox_accuracy += bbox_accuracy
                    bbox_accuracy_count += 1

                if pred_dict and gt_dict:
                    flat_preds = {
                        "boxes": torch.cat([p["boxes"] for p in pred_dict]),
                        "scores": torch.cat([p["scores"] for p in pred_dict]),
                        "labels": torch.cat([p["labels"] for p in pred_dict]),
                    }
                    flat_gts = {
                        "boxes": torch.cat([g["boxes"] for g in gt_dict]),
                        "labels": torch.cat([g["labels"] for g in gt_dict]),
                    }
                    # Accuracy
                    # pred_labels = flat_preds['labels']
                    # gt_labels = flat_gts['labels']
                    # correct_preds += (pred_labels == gt_labels).sum().item()
                    # total_preds += len(gt_labels)

                    # mAP update
                    metric_fn.update([flat_preds], [flat_gts])

                image_id += 1

            loss_dict = outputs.loss_dict
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_loss_ce += loss_dict['loss_ce'].item()
            total_loss_bbox += loss_dict['loss_bbox'].item()
            total_loss_giou += loss_dict['loss_giou'].item()
            batches += 1

        train_loss_ce_history.append(total_loss_ce / batches)
        train_loss_giou_history.append(total_loss_giou / batches)
        train_iou_history.append(total_iou / max(iou_count, 1))
        # train_acc_history.append(correct_preds / max(total_preds, 1))
        train_metrics = metric_fn.compute()
        train_map_history.append(train_metrics['map'])
        metric_fn.reset()
        mean_bbox_accuracy = total_bbox_accuracy / max(bbox_accuracy_count, 1)
        train_bbox_acc_history.append(mean_bbox_accuracy)


        print(f"Epoch[{epoch + 1}], Loss: {epoch_loss / len(train_dataloader):.4f}")
        print(f'loss_ce: {total_loss_ce / batches:.4f}')
        print(f'loss_bbox: {total_loss_bbox / batches:.4f}')
        print(f'loss_giou: {total_loss_giou / batches:.4f}')
        print(f'IoU: {total_iou / max(iou_count, 1)}')
        # print(f"Accuracy: {correct_preds / max(total_preds, 1)}")
        print(f'mAP: {train_metrics["map"]:.4f}')
        print(f"Bounding Box Accuracy: {mean_bbox_accuracy:.4f}")

    # Model Evaluation
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    val_all_predictions = []
    all_ground_truths = []
    final_result = None
    final_target = None
    final_pixel_values = None
    total_iou = 0.0
    iou_count = 0

    for epoch in range(num_epochs):
        model.eval()
        image_id = 0
        val_batches = 0
        val_loss = 0
        val_loss_ce = 0
        val_loss_giou = 0
        correct_preds = 0
        total_preds = 0
        total_bbox_accuracy = 0
        bbox_accuracy_count = 0

        with torch.no_grad():
            for images, targets in tqdm(val_dataloader):
                pixel_values = torch.stack([img for img in images]).to(device)
                final_pixel_values = pixel_values

                labels = []
                for target in targets:
                    boxes = target['boxes'].to(device)
                    # Convert [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
                    cxcywh_boxes = torch.zeros_like(boxes)
                    cxcywh_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                    cxcywh_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                    cxcywh_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    cxcywh_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                    # Normalize coordinates to [0, 1] based on image size (224x224)
                    cxcywh_boxes[:, [0, 2]] /= 224.0
                    cxcywh_boxes[:, [1, 3]] /= 224.0

                    labels.append({
                        'class_labels': target['labels'].to(device),
                        'boxes': cxcywh_boxes
                    })
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss_dict = outputs.loss_dict
                loss = outputs.loss
                val_loss += loss.item()
                val_loss_ce += loss_dict['loss_ce'].item()
                val_loss_giou += loss_dict['loss_giou'].item()
                val_batches += 1

                target_sizes = torch.tensor([image.shape[-2:] for image in pixel_values]).to(device)
                results = processor.post_process_object_detection(outputs, threshold=.5, target_sizes=target_sizes)
                final_result = results[0]
                final_target = targets[0]

                for idx, result in enumerate(results):
                    gt_boxes = []
                    map_gt = []
                    gt_dict = []
                    pred_dict = []
                    for box, label in zip(targets[idx]['boxes'], targets[idx]['labels']):
                        cx, cy, w, h = box
                        xmin = cx - w / 2
                        xmax = cx + w / 2
                        ymin = cy - h / 2
                        ymax = cy + h / 2
                        width = xmax - xmin
                        height = ymax - ymin
                        gt_boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                        map_gt = torch.tensor([xmin.item(), ymin.item(), xmax.item(), ymax.item(), image_id, 1.0])
                        gt_dict.append({
                            "boxes": torch.tensor([[xmin, ymin, xmax, ymax]]).to(device),
                            "labels": torch.tensor([label]).to(device)
                        })
                        all_ground_truths.append({
                            "image_id": image_id, "category_id": label.item(),
                            "bbox": [xmin.item(), ymin.item(), width.item(), height.item()],
                            "area": width.item() * height.item(),
                            "iscrowd": 0})

                    pred_boxes = []
                    map_pred = []
                    for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
                        cx, cy, w, h = box.tolist()
                        xmin = cx - w / 2
                        xmax = cx + w / 2
                        ymin = cy - h / 2
                        ymax = cy + h / 2
                        width = xmax - xmin
                        height = ymax - ymin
                        pred_boxes.append([xmin, ymin, xmax, ymax])
                        map_pred = torch.tensor([xmin, ymin, xmax, ymax, image_id, score.item()])
                        pred_dict.append({
                            "boxes": torch.tensor([[xmin, ymin, xmax, ymax]]).to(device),
                            "scores": torch.tensor([score]).to(device),
                            "labels": torch.tensor([label]).to(device)
                        })
                        val_all_predictions.append({
                            "image_id": image_id, "category_id": label.item(),
                            "bbox": [xmin, ymin, width, height], "score": score.item()})

                    if gt_boxes and pred_boxes:
                        iou_matrix = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes))
                        # print("IoU value", iou_matrix)
                        mean_iou = iou_matrix.max(dim=1).values.mean().item()
                        total_iou += mean_iou
                        iou_count += 1

                        ious = box_iou(torch.tensor(gt_boxes),
                                       torch.tensor(pred_boxes)).diag()  # take diagonal (matched pairs)
                        iou_thresh = 0.5
                        correct_bbox = (ious > iou_thresh).sum().item()
                        total_bbox = len(ious)
                        bbox_accuracy = correct_bbox / total_bbox

                        total_bbox_accuracy += bbox_accuracy
                        bbox_accuracy_count += 1

                    if pred_dict and gt_dict:
                        flat_preds = {
                            "boxes": torch.cat([p["boxes"] for p in pred_dict]),
                            "scores": torch.cat([p["scores"] for p in pred_dict]),
                            "labels": torch.cat([p["labels"] for p in pred_dict]),
                        }
                        flat_gts = {
                            "boxes": torch.cat([g["boxes"] for g in gt_dict]),
                            "labels": torch.cat([g["labels"] for g in gt_dict]),
                        }
                        metric_fn.update([flat_preds], [flat_gts])

                        # # Accuracy
                        # pred_labels = flat_preds['labels']
                        # gt_labels = flat_gts['labels']
                        # correct_preds += (pred_labels == gt_labels).sum().item()
                        # total_preds += len(gt_labels)

                    image_id += 1


        val_loss_ce_history.append(val_loss_ce / val_batches)
        val_loss_giou_history.append(val_loss_giou / val_batches)
        val_iou_history.append(total_iou / max(iou_count, 1))
        # val_acc_history.append(correct_preds / max(total_preds, 1))
        val_metrics = metric_fn.compute()
        val_map_history.append(val_metrics['map'])
        metric_fn.reset()
        mean_bbox_accuracy = total_bbox_accuracy / max(bbox_accuracy_count, 1)
        val_bbox_acc_history.append(mean_bbox_accuracy)

        print(f"Epoch[{epoch + 1}], Loss: {val_loss / len(val_dataloader):.4f}")
        print(f'val_loss_ce: {val_loss_ce / val_batches:.4f}')
        print(f'val_loss_giou: {val_loss_giou / val_batches:.4f}')
        print(f'val_IoU: {total_iou / max(iou_count, 1)}')
        # print(f"val_Accuracy: {correct_preds / max(total_preds, 1)}")
        print(f'val_mAP: {val_metrics["map"]:.4f}')
        print(f"Bounding Box Accuracy: {mean_bbox_accuracy:.4f}")

        # print("CE & GIoU Loss:", val_loss_ce / val_batches, val_loss_giou / val_batches)
        # print("List of History", val_loss_ce_history, val_loss_giou_history)
        # map_metric.update(all_predictions, all_ground_truths)
        # print(f"mAP: {map_metric.compute():.4f}")
    mean_iou = total_iou / max(iou_count, 1)
    print(f"The mean IoU: {mean_iou:.4f}")
    map_result = metric_fn.compute()
    print(f"The map result: {map_result}")

    # Training and Validation Graphs
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(20, 5))

    # CE Loss
    plt.subplot(1, 4, 1)
    plt.plot(train_loss_ce_history, label='Train CE Loss')
    plt.plot(val_loss_ce_history, label='Val CE Loss')
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # IoU Loss
    plt.subplot(1, 4, 2)
    plt.plot(train_iou_history, label='Train IoU')
    plt.plot(val_iou_history, label='Val IoU')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    # GIoU Loss
    plt.subplot(1, 4, 3)
    plt.plot(train_loss_giou_history, label='Train GIoU Loss')
    plt.plot(val_loss_giou_history, label='Val GIoU Loss')
    plt.title('GIoU Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # BBox Accuracy
    plt.subplot(1, 4, 4)
    plt.plot(train_bbox_acc_history, label='Train BBox Accuracy')
    plt.plot(val_bbox_acc_history, label='Val BBox Accuracy')
    plt.title('BBox Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("Graphs/loss_curves.png")
    plt.show()

    # Simple Classification Accuracy
    correct = 0
    total = 0
    for pred in train_all_predictions:
        if pred['score'] > .7:
            correct += 1
        total += 1
    train_simple_accuracy = correct / total if total > 0 else 0
    print("The Simple Train accuracy for DeTR resnet-50 with >.7 Confidence:", train_simple_accuracy)

    correct = 0
    total = 0
    for pred in val_all_predictions:
        if pred['score'] > .7:
            correct += 1
        total += 1
    val_simple_accuracy = correct / total if total > 0 else 0
    print("The Simple Validation accuracy for DeTR resnet-50 with >.7 Confidence:", val_simple_accuracy)

    mean_iou = total_iou / iou_count if iou_count > 0 else 0
    print(f"Mean IoU: {mean_iou:.4f}")

    # Visualize a sample
    image = unnormalize(final_pixel_values[0]).permute(1, 2, 0).cpu().numpy()
    # image = final_pixel_values[0].permute(1, 2, 0).cpu().numpy()
    result = final_result
    target = final_target

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    ax.axis('off')
    final_label = None

    for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
        score = score.cpu().item()
        label = label.cpu().item()
        cx, cy, w, h = box.cpu().numpy()
        xmin = cx - w / 2
        xmax = cx + w / 2
        ymin = cy - h / 2
        ymax = cy + h / 2
        # xmin, ymin, xmax, ymax = box.cpu().numpy()
        print("Coords", xmin, ymin, xmax, ymax)

        if score < .7:
            continue
    # for label, box in zip(result['labels'], result['boxes']):
    #     label = label.cpu().item()
    #     xmin, ymin, xmax, ymax = box.cpu().numpy()
    #     print("Coords", xmin, ymin, xmax, ymax)
    #
    #     if score < .7:
    #         continue

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none', zorder=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{label_classes[label]}:{score:.2f}', fontsize=12, color='white',
                bbox=dict(facecolor='blue', alpha=0.5), zorder=3)
        final_label = label
    # save_path = os.path.join('images', f'{label_classes[final_label]}.png')
    # plt.savefig(save_path, bbox_inches='tight')
    plt.show()


    return model, val_dataloader


if __name__ == '__main__':
    detr()