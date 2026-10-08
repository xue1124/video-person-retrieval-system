"""可复用的模型加载与特征提取服务。"""

from services.inference.runtime import (
    extract_osnet_feat,
    extract_osnet_feat_batch,
    extract_siglip_feat_img,
    extract_siglip_feat_img_batch,
    extract_siglip_feat_text,
    get_runtime_backend_info,
    live_yolo_predict,
    warm_live_detector,
)

__all__ = [
    "extract_osnet_feat",
    "extract_osnet_feat_batch",
    "extract_siglip_feat_img",
    "extract_siglip_feat_img_batch",
    "extract_siglip_feat_text",
    "get_runtime_backend_info",
    "live_yolo_predict",
    "warm_live_detector",
]
