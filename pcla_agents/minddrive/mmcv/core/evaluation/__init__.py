# PCLA port: these are offline dataset-evaluation helpers (KITTI / Lyft / indoor /
# segmentation mAP). They are never touched during closed-loop inference, but they
# pull heavy optional third-party SDKs (lyft_dataset_sdk, nuscenes-devkit, open3d,
# ...) that we deliberately do not install -- adding them risks perturbing the
# shared PCLA env that ~20 other agents depend on. Import them opportunistically so
# a missing eval SDK cannot make the whole `mmcv` package unimportable; anything
# unavailable is bound to None and fails loudly only if actually called.
def _optional_import(module, names):
    try:
        mod = __import__(module, globals(), locals(), names, 1)
        return {n: getattr(mod, n) for n in names}
    except (ImportError, ModuleNotFoundError, OSError):
        return {n: None for n in names}


globals().update(_optional_import('indoor_eval', ['indoor_eval']))
globals().update(_optional_import('kitti_utils', ['kitti_eval', 'kitti_eval_coco_style']))
globals().update(_optional_import('lyft_eval', ['lyft_eval']))
globals().update(_optional_import('seg_eval', ['seg_eval']))
from .class_names import (cityscapes_classes, coco_classes, dataset_aliases,
                          get_classes, get_palette, imagenet_det_classes,
                          imagenet_vid_classes, voc_classes)
from .eval_hooks import DistEvalHook, EvalHook, CustomDistEvalHook
from .mean_ap import average_precision, eval_map, print_map_summary
from .recall import (eval_recalls, plot_iou_recall, plot_num_recall,
                     print_recall_summary)
from .metrics import eval_metrics, mean_dice, mean_fscore, mean_iou
from .metric_motion import get_ade,get_best_preds,get_fde