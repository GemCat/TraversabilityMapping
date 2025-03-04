import torch
import numpy as np
import cv2
import os
import random
from PIL import Image, ImageDraw
from torchvision import transforms

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer, _create_text_labels, GenericMask
import detectron2.data.transforms as T
from detectron2.structures import Instances, Boxes

def create_bitmap(image, instances):
    masks = instances.pred_masks.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    sorted_idxs = np.argsort(scores)

    bitmap = np.zeros((image.shape[0], image.shape[1], 8))

    for idx in sorted_idxs:
        mask = masks[idx].astype(bool)
        cls = classes[idx]
        score = scores[idx]

        if cls == 8:
            continue

        bitmap[mask, cls] = score

    return bitmap


def draw_instances_with_priority(image, instances, metadata):
    """
    Draws the instances on the image with priority for higher scores and class-specific colors.
    
    Args:
        - image (np.array): the image array on which to draw the instances.
        - instances (detectron2.structures.Instances): the instances with prediction results.
        - metadata (detectron2.data.MetadataCatalog): metadata of the dataset, used for class color mapping.
    
    Returns:
        - image (np.array): the image array with the instances drawn.
    """
    
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    
    masks = instances.pred_masks.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    class_colors = metadata.thing_colors
    
    sorted_idxs = np.argsort(-scores)
    
    # Canvas to track occupancy of pixels in the image
    canvas = np.zeros_like(masks[0], dtype=bool)
    
    # Draw each instance
    for idx in sorted_idxs:
        mask = masks[idx].astype(bool)
        cls = classes[idx]
        color = class_colors[cls]
        
        # Find overlap with existing canvas
        overlap = np.logical_and(mask, canvas)
        
        # Subtract overlap from current mask
        mask = np.logical_and(mask, np.logical_not(overlap))
        
        # Update the canvas
        canvas = np.logical_or(canvas, mask)

        mask_pil = Image.fromarray(mask.astype(np.uint8)*255)
        draw.bitmap((0, 0), mask_pil, fill=tuple(color))
    
    return np.array(img_pil)


# filter predictions based on confidence score
def filter_instances_with_score(instances, threshold):
    filt_inst = Instances(instances.image_size)
    idxs = np.argwhere(instances.scores > threshold)[0]
    filt_inst.pred_masks = instances.pred_masks[idxs]
    filt_inst.pred_boxes = instances.pred_boxes[idxs]
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst

# load config file
cfg = LazyConfig.load("projects/ViTDet/configs/COCO/tree_mask_rcnn_vitdet_b_10ep.py")
# replace with the path where you have your model\
#cfg.train.init_checkpoint = 'output_dino_best/model_best.pth'
#cfg.train.init_checkpoint = 'output_sam_best/model_best.pth'
cfg.train.init_checkpoint = 'output_vit_best/model_best.pth'

metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids
classes = metadata.thing_classes

model = instantiate(cfg.model)
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

# save model (to convert .pkl to .pth
# torch.save(model.backbone.net.state_dict(), 'vitdet_b_COCO_net.pth')
# print(model.backbone.net)
print(model)

model.eval()
save_images = True
save_bitmap = True
# path to images to be labeled
#target_pre_labeld_dataset_path = '/home/rsl/yamaha2/val/'    # COCO
target_pre_labeld_dataset_path = '/home/rsl/harveri_imgs/trail/'
# path to saved prediction images
#save_dir = '/home/rsl/yamaha2/pred_dinov2/'
#save_dir = '/home/rsl/yamaha2/pred_sam/'
#save_dir = '/home/rsl/yamaha2/pred_vitdet/'
save_dir = '/home/rsl/harveri_inference/trail/trail_sem_vitdet/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get all jpg file in directory
file_names = []
dir_path = target_pre_labeld_dataset_path
for file in [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]:
    file_names.append(target_pre_labeld_dataset_path + file)


# for harveri, sample 30 random images from the dataset
# selected_files = random.sample(file_names, 30)

with torch.inference_mode():
    #for image_name in file_names:
    for image_name in file_names[0:100]:
        im = utils.read_image(image_name, "BGR")
        image = np.array(im, dtype=np.uint8)
        image = torch.from_numpy(image)

        # im, transforms = T.apply_transform_gens(
        #     [T.ResizeShortestEdge(short_edge_length=476, max_size=896, sample_style='choice')], #896, 896 for sam
        #     im)
      
        # Resize the image to 476x896 fpr dinov2
        # im = cv2.resize(im, (896, 476), interpolation=cv2.INTER_LINEAR)

        # to tensor
        image = torch.as_tensor(im.astype("float32"))
        # hwc -> chw
        image_tensor = torch.as_tensor(im.transpose(2, 0, 1).astype("float32"))

        output = model([{'image': image_tensor}])


        # HERE TO SAVE SEMANTIC IMAGES
        visualizer = Visualizer(image, metadata=metadata, scale=1)

        instances = filter_instances_with_score(output[0]["instances"].to("cpu"), 0.4)

        out_synth = visualizer.draw_instance_predictions(instances)

        image_with_instances = draw_instances_with_priority(im, output[0]["instances"], metadata)

        if save_images:
            save_name = image_name.replace(target_pre_labeld_dataset_path, save_dir)
            cv2.imwrite(save_name, image_with_instances)
            #cv2.imwrite(save_name, out_synth.get_image()[:,:,::])

        # #cv2.imshow('predictions', out_synth.get_image()[:, :, ::])
        # cv2.imshow('predictions', image_with_instances)
        # k = cv2.waitKey(0)

        # if k == ord('x'):
        #     break

        # cv2.destroyAllWindows()

        # HERE TO SAVE SEMANTIC BITMAPS
        # bitmap = create_bitmap(im, output[0]["instances"])
        # bitmap_dir = '/home/rsl/harveri_inference/trail/trail_bitmaps_vitdet/'

        # if not os.path.exists(bitmap_dir):
        #     os.makedirs(bitmap_dir)

        # # print(image_name)
        # # print(bitmap[200:205, 400:405, :])

        # if save_bitmap:
        #     n1 = image_name.replace(target_pre_labeld_dataset_path, bitmap_dir)
        #     save_name = n1.replace('.png', '_bitmap.npy')
        #     np.save(save_name, bitmap)