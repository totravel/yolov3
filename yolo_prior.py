import torch
from torch import Tensor
from utils import box_xyxy, box_origin_wh, iou_matrix
from utils import show_boxes


class YOLOv3Prior:
  def __init__(self, device=torch.device('cpu')):
    self.device = device

    self.img_h, self.img_w = 416, 416
    self._num_fmaps = 3
    self.num_anchors = 3

    self._fmap_sizes = [[52, 52], [26, 26], [13, 13]]

    # prior bbox sizes for 416x416 images
    self._prior_bbox_sizes = torch.tensor([
      [10, 13], [16, 30], [33, 23],
      [30, 61], [62, 45], [59, 119],
      [116, 90], [156, 198], [373, 326]
    ]).to(self.device)

    # prior bboxes in xyxy format for 416x416 images
    self._size_only_prior_bboxes = torch.zeros(len(self._prior_bbox_sizes), 4, device=self.device)
    self._size_only_prior_bboxes[:, 2:] = self._prior_bbox_sizes

    # prior bboxes in xyxy format and its sizes for each fmaps
    self.prior_bboxes, self.prior_bbox_sizes = [], []
    prior_bbox_sizes = self._prior_bbox_sizes.reshape(self._num_fmaps, self.num_anchors, -1)

    for size_idx, (fmap_h, fmap_w) in enumerate(self._fmap_sizes):
      stride_h = self.img_h / fmap_h
      stride_w = self.img_w / fmap_w

      prior_bbox_size_scale = torch.tensor([stride_w, stride_h], device=self.device)
      fmap_prior_bbox_sizes = prior_bbox_sizes[size_idx] / prior_bbox_size_scale
      fmap_prior_bboxes = self._generate_prior_bbox(fmap_h, fmap_w, fmap_prior_bbox_sizes)

      self.prior_bboxes.append(box_xyxy(fmap_prior_bboxes.reshape(-1, 4)))
      self.prior_bbox_sizes.append(fmap_prior_bbox_sizes)

  def _generate_prior_bbox(self, fmap_h, fmap_w, fmap_prior_bbox_sizes: Tensor):
    """
    Args:
      fmap_h (int): fmap height
      fmap_w (int): fmap width
      fmap_prior_bbox_sizes (Tensor): fmap prior bbox sizes
    Returns:
      fmap_prior_bboxes (Tensor): tensor of shape (FH, FW, num_anchors, 4)
    """
    grid_y = torch.arange(fmap_h, device=self.device)
    grid_x = torch.arange(fmap_w, device=self.device)
    shift_y, shift_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    prior_bboxes_xy = torch.stack([shift_x, shift_y], dim=-1)  # (FH, FW, 2)
    # (FH, FW, num_anchors, 2)
    prior_bboxes_xy = prior_bboxes_xy.reshape(fmap_h, fmap_w, 1, 2).repeat(1, 1, self.num_anchors, 1)

    # (FH, FW, num_anchors, 2)
    prior_bboxes_wh = fmap_prior_bbox_sizes.reshape(1, 1, self.num_anchors, 2).repeat(fmap_h, fmap_w, 1, 1)
    
    # (FH, FW, num_anchors, 4)
    fmap_prior_bboxes = torch.cat([prior_bboxes_xy, prior_bboxes_wh], dim=-1)
    # show_boxes(torch.zeros(fmap_h, fmap_w, 3), box_xyxy(fmap_prior_bboxes[fmap_h//2, fmap_w//2]))
    return fmap_prior_bboxes

  def assign_prior_bbox(self, gt_bboxes: Tensor):
    # print(f'found {len(gt_bboxes)} gt_bboxes')
    # show_boxes(torch.zeros(self.img_h, self.img_w, 3), gt_bboxes)
    size_only_gt_bboxes = box_origin_wh(gt_bboxes)

    ious = iou_matrix(self._size_only_prior_bboxes, size_only_gt_bboxes)
    matched_indices = ious.argmax(dim=0)  # TODO: in case two gt bboxes fall into the same cell
    
    size_indices = matched_indices // self.num_anchors
    ratio_indices = matched_indices % self.num_anchors
    return size_indices, ratio_indices
  
  @staticmethod
  def decode_pred_bbox(pred_offsets: Tensor, prior_bboxes: Tensor):
    """
    Args:
      pred_offsets (Tensor): (..., 4)
      prior_bboxes (Tensor): (..., 4)
    Returns:
      pred_bboxes (Tensor): (..., 4)
    """
    pred_xy_offsets = torch.sigmoid(pred_offsets[..., :2])
    pred_wh_offsets = torch.exp(pred_offsets[..., 2:])

    pred_bboxes = torch.empty_like(prior_bboxes)
    pred_bboxes[..., :2] = prior_bboxes[..., :2] + pred_xy_offsets
    pred_bboxes[..., 2:] = prior_bboxes[..., 2:] * pred_wh_offsets
    return pred_bboxes
