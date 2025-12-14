from data import DETRData
from model import DETR 
import torch 
from torch import load 
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt 
from utils.boxes import rescale_bboxes 
from utils.setup import get_classes 
from utils.logger import get_logger 
from utils.rich_handlers import TestHandler, DetectionHandler
from torchvision.ops import nms, box_convert

logger = get_logger('test')
test_handler = TestHandler()
detection_handler = DetectionHandler()

logger.print_banner() 

# ============== DETECTION PARAMETERS ==============
CONFIDENCE_THRESHOLD = 0.5  # Only keep predictions with confidence > 0.5
NMS_THRESHOLD = 0.5         # IoU threshold for NMS (lower = more aggressive filtering)
# ==================================================

num_classes = 26 
test_dataset = DETRData('data/test', train=False)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, drop_last=True)
model = DETR(num_classes=num_classes)
model.eval() 
model.load_pretrained('checkpoints/100_model.pt')

x, y = next(iter(test_dataloader))

logger.test('running inference on test batch')

import time 
start_time = time.time()
result = model(x)

inference_time = (time.time() - start_time) * 1000  # convert to ms

# Get predictions
probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
max_probs, max_classes = probabilities.max(-1)

# Apply confidence threshold
keep_mask = max_probs > CONFIDENCE_THRESHOLD
batch_indices, query_indices = torch.where(keep_mask)

bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (224, 224))
classes = max_classes[batch_indices, query_indices]
probas = max_probs[batch_indices, query_indices]

# ============== APPLY NMS PER IMAGE ==============
final_batch_indices = []
final_classes = []
final_probas = []
final_bboxes = []

for batch_idx in range(x.shape[0]):
    # Get all detections for this image
    mask = batch_indices == batch_idx
    if not mask.any():
        continue
    
    img_boxes = bboxes[mask]
    img_scores = probas[mask]
    img_classes = classes[mask]
    
    # Convert boxes from [xmin, ymin, xmax, ymax] to [x, y, w, h] format if needed
    # Ensure boxes are in xyxy format for NMS
    if img_boxes.shape[0] == 0:
        continue
    
    # Apply NMS per class to avoid suppressing different objects
    keep_indices = []
    for class_id in img_classes.unique():
        class_mask = img_classes == class_id
        class_boxes = img_boxes[class_mask]
        class_scores = img_scores[class_mask]
        class_indices = torch.where(class_mask)[0]
        
        # Apply NMS
        keep = nms(class_boxes, class_scores, NMS_THRESHOLD)
        keep_indices.extend(class_indices[keep].tolist())
    
    # Store filtered results
    for idx in keep_indices:
        final_batch_indices.append(batch_idx)
        final_classes.append(img_classes[idx])
        final_probas.append(img_scores[idx])
        final_bboxes.append(img_boxes[idx])

# Convert back to tensors
if len(final_batch_indices) > 0:
    final_batch_indices = torch.tensor(final_batch_indices)
    final_classes = torch.stack(final_classes)
    final_probas = torch.stack(final_probas)
    final_bboxes = torch.stack(final_bboxes)
else:
    final_batch_indices = torch.tensor([])
    final_classes = torch.tensor([])
    final_probas = torch.tensor([])
    final_bboxes = torch.tensor([])

# =====================================================

detection_handler.log_inference_time(inference_time=inference_time)

detections = []
for i in range(len(final_classes)):
    detections.append({
        'class': get_classes()[final_classes[i].item()],
        'confidence': final_probas[i].item(),
        'bbox': final_bboxes[i].detach().numpy().tolist()
    })

detection_handler.log_detections(detections)

CLASSES = get_classes()

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
axs = ax.flatten()

for idx, (img, ax) in enumerate(zip(x, axs)):
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
    
    # Plot only filtered detections
    for batch_idx, box_class, box_prob, bbox in zip(final_batch_indices, final_classes, final_probas, final_bboxes):
        if batch_idx == idx:
            xmin, ymin, xmax, ymax = bbox.detach().numpy()
            print(f"Image {idx}: {CLASSES[box_class]} - Confidence: {box_prob:.2f} - Box: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")
            
            ax.add_patch(plt.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin, 
                fill=False,
                color=(0.000, 0.447, 0.741), 
                linewidth=3
            ))
            
            text = f'{CLASSES[box_class]}: {box_prob:.2f}'
            ax.text(
                xmin, ymin - 5, text, 
                fontsize=12, 
                bbox=dict(facecolor='yellow', alpha=0.7),
                verticalalignment='bottom'
            )

fig.tight_layout()
plt.show()