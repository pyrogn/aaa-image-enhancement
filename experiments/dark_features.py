def is_dark_color(
    image: ImageConversions, threshold: int = 50
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale().flatten()
    mode_value = stats.mode(grayscale_image, axis=None).mode
    return {DefectNames.DARK_LIGHT: mode_value < threshold}


def is_dark_histogram(
    image: ImageConversions, dark_threshold: int = 50, dark_proportion: float = 0.5
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale()
    hist, _ = np.histogram(grayscale_image, bins=256, range=(0, 256))
    dark_pixels = np.sum(hist[:dark_threshold])
    total_pixels = grayscale_image.size
    return {DefectNames.DARK_HISTOGRAM: (dark_pixels / total_pixels) > dark_proportion}


def is_dark_threshold(
    image: ImageConversions,
    brightness_threshold: int = 50,
    dark_area_proportion: float = 0.5,
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale()
    dark_pixels = np.sum(grayscale_image < brightness_threshold)
    total_pixels = grayscale_image.size
    return {
        DefectNames.DARK_THRESHOLD: (dark_pixels / total_pixels) > dark_area_proportion
    }


def is_dark_local_contrast(
    image: ImageConversions,
    window_size: int = 15,
    dark_threshold: int = 50,
    dark_proportion: float = 0.5,
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale()
    mean_image = cv2.blur(grayscale_image, (window_size, window_size))
    dark_areas = grayscale_image < (mean_image - dark_threshold)
    dark_pixels = np.sum(dark_areas)
    total_pixels = grayscale_image.size
    return {
        DefectNames.DARK_LOCAL_CONTRAST: (dark_pixels / total_pixels) > dark_proportion
    }


def is_dark_adaptive_threshold(
    image: ImageConversions,
    max_value: int = 255,
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold_type: int = cv2.THRESH_BINARY,
    block_size: int = 11,
    C: int = 2,
    dark_proportion: float = 0.5,
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale()
    adaptive_thresh = cv2.adaptiveThreshold(
        grayscale_image, max_value, adaptive_method, threshold_type, block_size, C
    )
    dark_pixels = np.sum(adaptive_thresh == 0)
    total_pixels = adaptive_thresh.size
    return {
        DefectNames.DARK_ADAPTIVE_THRESHOLD: (dark_pixels / total_pixels)
        > dark_proportion
    }


def is_dark_v_channel(
    image: ImageConversions, threshold: int = 50, dark_proportion: float = 0.5
) -> dict[DefectNames, bool]:
    hsv_image = cv2.cvtColor(image.to_cv2(), cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]
    dark_pixels = np.sum(v_channel < threshold)
    total_pixels = v_channel.size
    return {DefectNames.DARK_V_CHANNEL: (dark_pixels / total_pixels) > dark_proportion}


def is_dark_edges(
    image: ImageConversions,
    edge_threshold1: int = 100,
    edge_threshold2: int = 200,
    brightness_threshold: int = 50,
    dark_proportion: float = 0.5,
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale()
    edges = cv2.Canny(grayscale_image, edge_threshold1, edge_threshold2)
    dark_areas = grayscale_image[edges > 0] < brightness_threshold
    dark_pixels = np.sum(dark_areas)
    total_pixels = edges.size
    return {DefectNames.DARK_EDGES: (dark_pixels / total_pixels) > dark_proportion}


def is_dark_blocks(
    image: ImageConversions,
    block_size: int = 32,
    dark_threshold: int = 50,
    dark_proportion: float = 0.5,
) -> dict[DefectNames, bool]:
    grayscale_image = image.to_grayscale()
    h, w = grayscale_image.shape
    block_dark_count = 0
    total_blocks = (h // block_size) * (w // block_size)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = grayscale_image[y : y + block_size, x : x + block_size]
            if np.mean(block) < dark_threshold:
                block_dark_count += 1

    return {
        DefectNames.DARK_BLOCKS: (block_dark_count / total_blocks) > dark_proportion
    }
