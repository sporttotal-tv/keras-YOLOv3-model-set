import os
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_file(filepath, is_pred=False):
    with open(filepath, 'r') as f:
        data = f.readlines()
        annotations = []
        for line in data:
            elems = line.split(' ')
            class_name = elems[0]
            bbox = list(map(float, elems[1:])) # for ground truth file
            if is_pred:  # if it's a prediction file
                score = float(elems[1])
                bbox = list(map(float, elems[2:]))
                annotations.append({'class_name': class_name, 'score': score, 'bbox': bbox})
            else:  # for ground truth file
                annotations.append({'class_name': class_name, 'bbox': bbox})
    return annotations

def calculate_metrics(pred_dir, gt_dir):
    coco_gt = COCO()
    coco_dt = coco_gt.loadRes()

    # Get the .txt files
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    
    assert len(gt_files) == len(pred_files), "The number of ground truth and prediction files should be same."
    assert set(gt_files) == set(pred_files), "The names of ground truth and prediction files should match."

    for file_name in gt_files:
        gt_annotations = parse_file(os.path.join(gt_dir, file_name), is_pred=False)
        pred_annotations = parse_file(os.path.join(pred_dir, file_name), is_pred=True)
        
        for gt in gt_annotations:
            coco_gt.loadAnns(gt)
        
        for dt in pred_annotations:
            coco_dt.loadAnns(dt)
        
    cocoEval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print('mAP: ', cocoEval.stats[0])
    print('Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]: ', cocoEval.stats[1])
    print('Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]: ', cocoEval.stats[2])
    print('Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]: ', cocoEval.stats[3])
    print('Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]: ', cocoEval.stats[7])
    print('Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]: ', cocoEval.stats[8])
    print('Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]: ', cocoEval.stats[9])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate COCO Metrics')
    parser.add_argument('pred_dir', type=str, help='Directory path for predictions')
    parser.add_argument('gt_dir', type=str, help='Directory path for ground truth annotations')

    args = parser.parse_args()
   
