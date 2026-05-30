import os
import sys
import json
from functools import partial
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F


class NemotronDetector:
    """
    Unified detector-only wrapper for Nemotron OCR2 word and line detection.

    Supported variants:
        variant="word"
        variant="line"

    It loads detector.safetensors or detector.pth, reads model_config.json, runs FOTSDetector,
    decodes RBOX outputs approximately, applies rectangular NMS, and exports:
        1. labeled image
        2. binary mask
        3. JSON diagnostics
        4. optional paired debug image

    This is detector-only. It does not run the recognizer or relational model.
    """

    VARIANT_DEFAULTS = {
        "word": {
            "prob_threshold": 0.30,
            "nms_iou_threshold": 0.30,
            "max_regions": 3000,
            "suffix": "word",
            "description": "word detector",
        },
        "line": {
            "prob_threshold": 0.50,
            "nms_iou_threshold": 0.20,
            "max_regions": 1000,
            "suffix": "line",
            "description": "line detector",
        },
    }

    def __init__(
        self,
        model_dir,
        variant="word",
        project_root=None,
        detector_path=None,
        config_path=None,
        device=None,
        infer_length=1024,
        downsample=4,
        backbone=None,
        coordinate_mode=None,
        order_mode="heuristic",
        scope=None,
        prob_threshold=None,
        nms_iou_threshold=None,
        max_regions=None,
    ):
        if variant not in self.VARIANT_DEFAULTS:
            raise ValueError(
                f"Unsupported variant: {variant}. "
                f"Expected one of {list(self.VARIANT_DEFAULTS.keys())}."
            )

        if project_root is None:
            project_root = os.path.dirname(os.path.abspath(__file__))

        self.project_root = project_root
        self.src_dir = project_root
        self.model_dir = model_dir
        self.variant = variant

        defaults = self.VARIANT_DEFAULTS[variant]

        if detector_path is None:
            safetensors_path = os.path.join(model_dir, "detector.safetensors")
            pth_path = os.path.join(model_dir, "detector.pth")

            if os.path.exists(safetensors_path):
                detector_path = safetensors_path
            elif os.path.exists(pth_path):
                detector_path = pth_path
            else:
                raise FileNotFoundError(
                    "No detector checkpoint found. Expected one of:\n"
                    f"  {safetensors_path}\n"
                    f"  {pth_path}"
                )

        if config_path is None:
            config_path = os.path.join(model_dir, "model_config.json")

        self.detector_path = detector_path
        self.config_path = config_path

        self.order_mode = order_mode

        self.model_config = self._load_model_config(config_path)

        if backbone is None:
            backbone = self.model_config.get("backbone", "regnet_x_8gf")

        if coordinate_mode is None:
            coordinate_mode = self.model_config.get("coordinate_mode", "RBOX")

        if scope is None:
            scope = self.model_config.get("scope", 2048)

        if prob_threshold is None:
            prob_threshold = defaults["prob_threshold"]

        if nms_iou_threshold is None:
            nms_iou_threshold = defaults["nms_iou_threshold"]

        if max_regions is None:
            max_regions = defaults["max_regions"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.infer_length = int(infer_length)
        self.downsample = int(downsample)

        self.backbone = str(backbone)
        self.coordinate_mode = str(coordinate_mode)
        self.scope = int(scope)

        self.prob_threshold = float(prob_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_regions = int(max_regions)

        self.output_suffix = defaults["suffix"]
        self.detector_description = defaults["description"]

        self.detector = None

    @classmethod
    def from_variant(
        cls,
        variant,
        root_dir=None,
        models_dir_name="models",
        **kwargs,
    ):
        """
        Convenience constructor.

        Example:
            detector = NemotronDetector.from_variant("word")
            detector = NemotronDetector.from_variant("line")

        Expected folders:
            root_dir/models/word/
            root_dir/models/line/
        """
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))

        model_dir = os.path.join(root_dir, models_dir_name, variant)

        return cls(
            model_dir=model_dir,
            variant=variant,
            project_root=root_dir,
            **kwargs,
        )

    # ============================================================
    # Setup and model loading
    # ============================================================

    @staticmethod
    def _load_model_config(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"model_config.json not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _add_src_to_path(self):
        # Kept for backward compatibility. Package-relative imports are used.
        return None

    def _patch_regnet_download_only(self):
        """
        Keep pretrained=True RegNet architecture shape, but prevent downloading
        ImageNet weights. The detector checkpoint is loaded afterward.
        """
        from .inference import regnet

        def _regnet_no_download(arch, block_params, pretrained, progress, **kwargs):
            model = regnet.RegNet(
                block_params,
                norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1),
                **kwargs,
            )
            return model

        regnet._regnet = _regnet_no_download

    def load(self):
        self._add_src_to_path()
        self._patch_regnet_download_only()

        from .inference.fots_detector import FOTSDetector

        detector = FOTSDetector(
            backbone=self.backbone,
            coordinate_mode=self.coordinate_mode,
            scope=self.scope,
            verbose=False,
        )

        if self.detector_path.lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(self.detector_path, device="cpu")
        else:
            state_dict = torch.load(self.detector_path, map_location="cpu")

        detector.load_state_dict(state_dict, strict=True)

        detector.eval()
        detector.to(self.device)

        self.detector = detector
        return self

    # ============================================================
    # Image preprocessing
    # ============================================================

    def _load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size

        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        padded_side = max(original_h, original_w)
        pad_right = padded_side - original_w
        pad_bottom = padded_side - original_h

        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), value=1.0)
        tensor = tensor.unsqueeze(0).to(self.device)

        tensor = F.interpolate(
            tensor,
            size=(self.infer_length, self.infer_length),
            mode="bilinear",
            align_corners=True,
        )

        image_info = {
            "image_path": image_path,
            "original_width": int(original_w),
            "original_height": int(original_h),
            "padded_side": int(padded_side),
            "infer_length": int(self.infer_length),
            "scale_back_to_original_padded_space": float(padded_side) / float(self.infer_length),
        }

        return tensor, image_info

    # ============================================================
    # RBOX decoding
    # ============================================================

    def _rboxes_to_quads_torch(self, rboxes):
        """
        Approximate replacement for official C++ rrect_to_quads.

        Assumed RBOX layout:
            [top, right, bottom, left, rotation]

        Input:
            rboxes: [B, H, W, 5]

        Output:
            quads: [B, H, W, 4, 2]
        """
        b, h, w, _ = rboxes.shape
        device = rboxes.device
        dtype = torch.float32

        rboxes = rboxes.to(dtype)

        top = rboxes[..., 0]
        right = rboxes[..., 1]
        bottom = rboxes[..., 2]
        left = rboxes[..., 3]
        angle = rboxes[..., 4]

        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * self.downsample
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * self.downsample
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        cx = xx.unsqueeze(0).expand(b, -1, -1)
        cy = yy.unsqueeze(0).expand(b, -1, -1)

        local_x = torch.stack([-left, right, right, -left], dim=-1)
        local_y = torch.stack([-top, -top, bottom, bottom], dim=-1)

        cos_a = torch.cos(angle).unsqueeze(-1)
        sin_a = torch.sin(angle).unsqueeze(-1)

        rx = local_x * cos_a - local_y * sin_a + cx.unsqueeze(-1)
        ry = local_x * sin_a + local_y * cos_a + cy.unsqueeze(-1)

        quads = torch.stack([rx, ry], dim=-1)
        return quads


    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        try:
            from torchvision.ops import nms
        except Exception as exc:
            raise RuntimeError(
                "torchvision.ops.nms is required for post-processing. "
                "Install a torchvision version that matches your PyTorch build."
            ) from exc

        return nms(boxes, scores, iou_threshold)

    @staticmethod
    def _boxes_from_quads(quads):
        x1 = quads[:, :, 0].min(dim=1).values
        y1 = quads[:, :, 1].min(dim=1).values
        x2 = quads[:, :, 0].max(dim=1).values
        y2 = quads[:, :, 1].max(dim=1).values
        return torch.stack([x1, y1, x2, y2], dim=1)

    # ============================================================
    # Prediction
    # ============================================================

    def predict(
        self,
        image_path,
        include_raw_candidates=True,
        max_raw_candidates_for_json=500,
    ):
        """
        Runs detection and returns a dictionary suitable for JSON export.
        """
        if self.detector is None:
            self.load()

        image_tensor, image_info = self._load_and_preprocess_image(image_path)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    conf_logits, offsets, rboxes, features = self.detector(image_tensor)
            else:
                conf_logits, offsets, rboxes, features = self.detector(image_tensor)

        if rboxes is None:
            raise RuntimeError("Detector returned rboxes=None. This wrapper expects coordinate_mode='RBOX'.")

        conf_logits = conf_logits.float()[0]
        conf = torch.sigmoid(conf_logits)

        rboxes_0 = rboxes.float()[0]
        quads_grid_infer = self._rboxes_to_quads_torch(rboxes.float())[0]

        candidate_mask = conf > self.prob_threshold
        candidate_indices = candidate_mask.nonzero(as_tuple=False)

        if candidate_indices.numel() == 0:
            return self._empty_result(image_info, conf_logits, rboxes, features, offsets)

        candidate_quads_infer = quads_grid_infer[candidate_mask]
        candidate_scores = conf[candidate_mask]
        candidate_logits = conf_logits[candidate_mask]
        candidate_rboxes = rboxes_0[candidate_mask]

        candidate_boxes_infer = self._boxes_from_quads(candidate_quads_infer)

        keep = self._nms(candidate_boxes_infer, candidate_scores, self.nms_iou_threshold)
        keep = keep[:self.max_regions]

        kept_quads_infer = candidate_quads_infer[keep]
        kept_scores = candidate_scores[keep]
        kept_logits = candidate_logits[keep]
        kept_rboxes = candidate_rboxes[keep]
        kept_indices = candidate_indices[keep]

        scale = float(image_info["padded_side"]) / float(self.infer_length)

        kept_quads_original = kept_quads_infer * scale
        kept_quads_original[:, :, 0] = kept_quads_original[:, :, 0].clamp(
            0,
            image_info["original_width"],
        )
        kept_quads_original[:, :, 1] = kept_quads_original[:, :, 1].clamp(
            0,
            image_info["original_height"],
        )

        detections = []

        for det_id in range(kept_quads_original.shape[0]):
            quad_original = kept_quads_original[det_id]
            quad_infer = kept_quads_infer[det_id]

            x_min = float(quad_original[:, 0].min().item())
            y_min = float(quad_original[:, 1].min().item())
            x_max = float(quad_original[:, 0].max().item())
            y_max = float(quad_original[:, 1].max().item())

            if x_max <= x_min or y_max <= y_min:
                continue

            grid_y = int(kept_indices[det_id, 0].item())
            grid_x = int(kept_indices[det_id, 1].item())

            rbox_values = kept_rboxes[det_id].detach().cpu().numpy().astype(float).tolist()

            detections.append({
                "id": int(len(detections) + 1),
                "variant": self.variant,
                "confidence": float(kept_scores[det_id].item()),
                "confidence_logit": float(kept_logits[det_id].item()),
                "grid": {
                    "x": grid_x,
                    "y": grid_y,
                    "center_x_infer": float((grid_x + 0.5) * self.downsample),
                    "center_y_infer": float((grid_y + 0.5) * self.downsample),
                    "downsample": int(self.downsample),
                },
                "rbox": {
                    "top": float(rbox_values[0]),
                    "right": float(rbox_values[1]),
                    "bottom": float(rbox_values[2]),
                    "left": float(rbox_values[3]),
                    "rotation": float(rbox_values[4]),
                    "raw": rbox_values,
                    "layout_assumption": "[top, right, bottom, left, rotation]",
                },
                "quad_infer_space": quad_infer.detach().cpu().numpy().astype(float).tolist(),
                "quad_original_space": quad_original.detach().cpu().numpy().astype(float).tolist(),
                "box_xyxy_original_space": [x_min, y_min, x_max, y_max],
                "box_normalized_xyxy": [
                    x_min / float(image_info["original_width"]),
                    y_min / float(image_info["original_height"]),
                    x_max / float(image_info["original_width"]),
                    y_max / float(image_info["original_height"]),
                ],
            })

        detections = self._apply_reading_order(detections, mode=self.order_mode)
        raw_candidates = []
        if include_raw_candidates:
            raw_candidates = self._make_raw_candidates_for_json(
                candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                candidate_logits=candidate_logits,
                candidate_rboxes=candidate_rboxes,
                candidate_quads_infer=candidate_quads_infer,
                image_info=image_info,
                max_items=max_raw_candidates_for_json,
            )

        result = {
            "metadata": self._make_metadata(
                image_info=image_info,
                conf_logits=conf_logits,
                rboxes=rboxes,
                features=features,
                offsets=offsets,
                thresholded_candidates=int(candidate_indices.shape[0]),
                final_detections=int(len(detections)),
                raw_candidates_in_json=int(len(raw_candidates)),
                note=(
                    f"{self.detector_description} only. "
                    "Windows-compatible approximation. "
                    "RBOX decoding and NMS are not official NVIDIA C++ implementations."
                ),
            ),
            "image": image_info,
            "detections": detections,
            "raw_candidates": raw_candidates,
        }

        return result

    def _empty_result(self, image_info, conf_logits, rboxes, features, offsets):
        return {
            "metadata": self._make_metadata(
                image_info=image_info,
                conf_logits=conf_logits,
                rboxes=rboxes,
                features=features,
                offsets=offsets,
                thresholded_candidates=0,
                final_detections=0,
                raw_candidates_in_json=0,
                note="No detections above threshold.",
            ),
            "image": image_info,
            "detections": [],
            "raw_candidates": [],
        }

    def _make_metadata(
        self,
        image_info,
        conf_logits,
        rboxes,
        features,
        offsets,
        thresholded_candidates,
        final_detections,
        raw_candidates_in_json,
        note,
    ):
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "device": self.device,
            "model": {
                "variant": self.variant,
                "description": self.detector_description,
                "detector_path": self.detector_path,
                "config_path": self.config_path,
                "backbone": self.backbone,
                "coordinate_mode": self.coordinate_mode,
                "scope": self.scope,
            },
            "inference": {
                "infer_length": self.infer_length,
                "downsample": self.downsample,
                "prob_threshold": self.prob_threshold,
                "nms_iou_threshold": self.nms_iou_threshold,
                "max_regions": self.max_regions,
                "order_mode": self.order_mode,
            },
            "tensor_shapes": {
                "conf_logits": list(conf_logits.shape),
                "rboxes": list(rboxes.shape),
                "features": list(features.shape) if features is not None else None,
                "offsets": list(offsets.shape) if offsets is not None else None,
            },
            "counts": {
                "thresholded_candidates": int(thresholded_candidates),
                "final_detections": int(final_detections),
                "raw_candidates_in_json": int(raw_candidates_in_json),
            },
            "image": {
                "original_width": image_info["original_width"],
                "original_height": image_info["original_height"],
                "padded_side": image_info["padded_side"],
            },
        }

    def _make_raw_candidates_for_json(
        self,
        candidate_indices,
        candidate_scores,
        candidate_logits,
        candidate_rboxes,
        candidate_quads_infer,
        image_info,
        max_items,
    ):
        """
        Stores top scoring raw thresholded grid candidates.
        """
        if candidate_indices.shape[0] == 0:
            return []

        max_items = int(max_items)
        if max_items <= 0:
            return []

        count = min(max_items, int(candidate_indices.shape[0]))
        top_scores, order = torch.topk(candidate_scores, k=count)
        scale = float(image_info["padded_side"]) / float(self.infer_length)

        raw_items = []

        for rank in range(count):
            idx = order[rank]
            grid_y = int(candidate_indices[idx, 0].item())
            grid_x = int(candidate_indices[idx, 1].item())

            rbox_values = candidate_rboxes[idx].detach().cpu().numpy().astype(float).tolist()
            quad_infer = candidate_quads_infer[idx]
            quad_original = quad_infer * scale

            quad_original[:, 0] = quad_original[:, 0].clamp(0, image_info["original_width"])
            quad_original[:, 1] = quad_original[:, 1].clamp(0, image_info["original_height"])

            raw_items.append({
                "rank": int(rank + 1),
                "variant": self.variant,
                "confidence": float(top_scores[rank].item()),
                "confidence_logit": float(candidate_logits[idx].item()),
                "grid": {
                    "x": grid_x,
                    "y": grid_y,
                    "center_x_infer": float((grid_x + 0.5) * self.downsample),
                    "center_y_infer": float((grid_y + 0.5) * self.downsample),
                },
                "rbox": {
                    "top": float(rbox_values[0]),
                    "right": float(rbox_values[1]),
                    "bottom": float(rbox_values[2]),
                    "left": float(rbox_values[3]),
                    "rotation": float(rbox_values[4]),
                    "raw": rbox_values,
                },
                "quad_infer_space": quad_infer.detach().cpu().numpy().astype(float).tolist(),
                "quad_original_space": quad_original.detach().cpu().numpy().astype(float).tolist(),
            })

        return raw_items

    def _apply_reading_order(self, detections, mode="heuristic"):
        """
        Adds reading_order to each detection.

        This is a geometry-only heuristic. It is not equivalent to the
        Nemotron relational model.
        """
        if mode == "none" or len(detections) == 0:
            for i, det in enumerate(detections):
                det["reading_order"] = int(i + 1)
            return detections

        if mode != "heuristic":
            raise ValueError(f"Unsupported order mode: {mode}")

        items = []

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["box_xyxy_original_space"]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            h = max(1.0, y2 - y1)

            items.append({
                "index": idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "h": h,
            })

        if self.variant == "line":
            # For line detection, each detection is already a line.
            ordered = sorted(items, key=lambda d: (d["cy"], d["x1"]))
        else:
            # For word detection, cluster into approximate rows.
            median_h = float(np.median([d["h"] for d in items]))
            row_eps = max(8.0, 0.60 * median_h)

            items_sorted = sorted(items, key=lambda d: d["cy"])
            rows = []

            for item in items_sorted:
                placed = False

                for row in rows:
                    if abs(item["cy"] - row["cy_mean"]) <= row_eps:
                        row["items"].append(item)
                        row["cy_mean"] = float(np.mean([x["cy"] for x in row["items"]]))
                        placed = True
                        break

                if not placed:
                    rows.append({
                        "cy_mean": item["cy"],
                        "items": [item],
                    })

            rows = sorted(rows, key=lambda r: r["cy_mean"])

            ordered = []
            for row in rows:
                row_items = sorted(row["items"], key=lambda d: d["x1"])
                ordered.extend(row_items)

        for order_idx, item in enumerate(ordered):
            detections[item["index"]]["reading_order"] = int(order_idx + 1)

        detections = sorted(detections, key=lambda d: d["reading_order"])
        return detections

    # ============================================================
    # Exports
    # ============================================================

    def save_json(self, result, output_json_path):
        self._ensure_parent_dir(output_json_path)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("Saved JSON:", output_json_path)

    def save_mask(
        self,
        result,
        output_mask_path,
        mask_value=255,
        use_quads=True,
    ):
        self._ensure_parent_dir(output_mask_path)

        w = int(result["image"]["original_width"])
        h = int(result["image"]["original_height"])

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for det in result["detections"]:
            if use_quads:
                quad = det["quad_original_space"]
                polygon = [(float(x), float(y)) for x, y in quad]
                draw.polygon(polygon, fill=int(mask_value))
            else:
                x1, y1, x2, y2 = det["box_xyxy_original_space"]
                draw.rectangle([x1, y1, x2, y2], fill=int(mask_value))

        mask.save(output_mask_path)
        print("Saved mask:", output_mask_path)

    def save_labeled_image(
        self,
        result,
        output_image_path,
        draw_quads=True,
        draw_axis_box=False,
        line_width=2,
        font_size=18,
        show_index=True,
        show_confidence=True,
    ):
        self._ensure_parent_dir(output_image_path)

        image_path = result["image"]["image_path"]
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = self._load_font(font_size)

        self._draw_detections(
            image=image,
            draw=draw,
            font=font,
            detections=result["detections"],
            draw_quads=draw_quads,
            draw_axis_box=draw_axis_box,
            line_width=line_width,
            show_index=show_index,
            show_confidence=show_confidence,
        )

        image.save(output_image_path)
        print("Saved labeled image:", output_image_path)

    def save_paired_image(
        self,
        result,
        output_paired_path,
        gap=20,
        background=(255, 255, 255),
        draw_quads=True,
        draw_axis_box=False,
        line_width=2,
        font_size=18,
        show_index=True,
        show_confidence=True,
    ):
        self._ensure_parent_dir(output_paired_path)

        image_path = result["image"]["image_path"]

        original = Image.open(image_path).convert("RGB")
        labeled = Image.open(image_path).convert("RGB")

        draw = ImageDraw.Draw(labeled)
        font = self._load_font(font_size)

        self._draw_detections(
            image=labeled,
            draw=draw,
            font=font,
            detections=result["detections"],
            draw_quads=draw_quads,
            draw_axis_box=draw_axis_box,
            line_width=line_width,
            show_index=show_index,
            show_confidence=show_confidence,
        )

        paired_w = original.width + gap + labeled.width
        paired_h = max(original.height, labeled.height)

        paired = Image.new("RGB", (paired_w, paired_h), background)
        paired.paste(original, (0, 0))
        paired.paste(labeled, (original.width + gap, 0))

        paired.save(output_paired_path)
        print("Saved paired debug image:", output_paired_path)

    def process(
        self,
        image_path,
        output_dir=None,
        output_labeled_path=None,
        output_mask_path=None,
        output_json_path=None,
        output_paired_path=None,
        save_labeled=True,
        save_mask=True,
        save_json=True,
        save_paired=False,
        include_raw_candidates=True,
        max_raw_candidates_for_json=500,
    ):
        result = self.predict(
            image_path=image_path,
            include_raw_candidates=include_raw_candidates,
            max_raw_candidates_for_json=max_raw_candidates_for_json,
        )

        (
            default_labeled_path,
            default_mask_path,
            default_json_path,
            default_paired_path,
        ) = self._make_default_output_paths(
            image_path=image_path,
            output_dir=output_dir,
        )

        if output_labeled_path is None:
            output_labeled_path = default_labeled_path

        if output_mask_path is None:
            output_mask_path = default_mask_path

        if output_json_path is None:
            output_json_path = default_json_path

        if output_paired_path is None:
            output_paired_path = default_paired_path

        result["exports"] = {
            "labeled_image": output_labeled_path if save_labeled else None,
            "mask": output_mask_path if save_mask else None,
            "json": output_json_path if save_json else None,
            "paired_image": output_paired_path if save_paired else None,
        }

        if save_labeled:
            self.save_labeled_image(result, output_labeled_path)

        if save_mask:
            self.save_mask(result, output_mask_path)

        if save_json:
            self.save_json(result, output_json_path)

        if save_paired:
            self.save_paired_image(result, output_paired_path)

        return result

    # ============================================================
    # Drawing and path helpers
    # ============================================================

    def _make_default_output_paths(self, image_path, output_dir=None):
        input_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if output_dir is None:
            output_dir = input_dir

        suffix = self.output_suffix

        labeled_path = os.path.join(output_dir, base_name + f"_{suffix}_detected.png")
        mask_path = os.path.join(output_dir, base_name + f"_{suffix}_mask.png")
        json_path = os.path.join(output_dir, base_name + f"_{suffix}_detections.json")
        paired_path = os.path.join(output_dir, base_name + f"_{suffix}_paired.png")

        return labeled_path, mask_path, json_path, paired_path

    def _draw_detections(
        self,
        image,
        draw,
        font,
        detections,
        draw_quads,
        draw_axis_box,
        line_width,
        show_index,
        show_confidence,
    ):
        for i, det in enumerate(detections):
            color = self._get_color(i)

            if draw_quads:
                quad = det["quad_original_space"]
                quad_tuples = [(float(x), float(y)) for x, y in quad]
                draw.line(quad_tuples + [quad_tuples[0]], fill=color, width=line_width)

            if draw_axis_box:
                x1, y1, x2, y2 = det["box_xyxy_original_space"]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            label_parts = []
            if show_index:
                label_parts.append(str(det.get("reading_order", det["id"])))
            if show_confidence:
                label_parts.append(f'{det["confidence"]:.2f}')

            label_text = " | ".join(label_parts)
            if label_text:
                self._draw_label(
                    draw=draw,
                    image=image,
                    label_text=label_text,
                    det=det,
                    color=color,
                    font=font,
                )

    @staticmethod
    def _get_color(index):
        palette = [
            (255, 0, 0),
            (0, 180, 0),
            (0, 120, 255),
            (255, 140, 0),
            (180, 0, 255),
            (0, 180, 180),
            (255, 0, 180),
            (120, 120, 0),
            (0, 0, 0),
            (120, 60, 0),
        ]
        return palette[index % len(palette)]

    @staticmethod
    def _load_font(font_size=18):
        font_candidates = [
            "arial.ttf",
            "Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]

        for font_path in font_candidates:
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                pass

        return ImageFont.load_default()

    @staticmethod
    def _draw_label(draw, image, label_text, det, color, font):
        quad = det["quad_original_space"]

        label_x = min(float(pt[0]) for pt in quad)
        label_y = min(float(pt[1]) for pt in quad)

        try:
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w, text_h = font.getsize(label_text)

        label_x = max(0, min(label_x, image.width - text_w - 6))
        label_y = max(0, label_y - text_h - 6)

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 6, label_y + text_h + 4],
            fill=color,
        )

        draw.text(
            (label_x + 3, label_y + 2),
            label_text,
            fill=(255, 255, 255),
            font=font,
        )

    @staticmethod
    def _ensure_parent_dir(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
