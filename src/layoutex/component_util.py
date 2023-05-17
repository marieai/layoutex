"""
Contains document component utility functions.
"""

import logging
logger = logging.getLogger(__name__)

def estimate_component_sizing(box, target_size, margin_size):
    """
    estimate the component sizing based on the bounding box size and the target size of the document including margins
    """
    logger.info(f"estimate_component_sizing(target_size={target_size}, margin_size={margin_size})")
    x1, y1, x2, y2 = box
    target_size = target_size - 2 * margin_size
    logger.info(f"computed target_size = {target_size}")
    w = int(x2 - x1)
    h = int(y2 - y1)
    logger.info(f"computed dimensions  = (w={w}, h={h})")
    ratio_w = w / target_size
    ratio_h = h / target_size
    logger.info(f"computed ratios      = (ratio_w={ratio_w}, ratio_h={ratio_h})")

    component_w = "FULL_WIDTH"
    component_h = "FULL_HEIGHT"

    if ratio_w > 0.75:
        component_w = "FULL_WIDTH"
    elif ratio_w > 0.5:
        component_w = "TWO_THIRDS_WIDTH"
    elif ratio_w > 0.25:
        component_w = "HALF_WIDTH"
    elif ratio_w > 0.01:
        component_w = "QUARTER_WIDTH"

    logger.info(f"computed component_w = {component_w}")

    if ratio_h > 0.75:
        component_h = "FULL_HEIGHT"
    elif ratio_h > 0.25:
        component_h = "HALF_HEIGHT"
    elif ratio_h > 0.05:
        component_h = "QUARTER_HEIGHT"
    elif ratio_h > 0.01:
        component_h = "LINE_HEIGHT"

    logger.info(f"computed component_h = {component_h}")

    # logger.debug(f"estimated component sizing - ratio_w: {ratio_w}, ratio_h: {ratio_h}  -> {component_w}, {component_h}")
    return component_w, component_h