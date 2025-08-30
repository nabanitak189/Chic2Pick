# import numpy as np
# import cv2
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# from skimage.color import rgb2gray
# from skimage.util import img_as_ubyte
# from sklearn.cluster import KMeans

# # --- Basic helpers ---
# def to_cv(image_pil):
#     # PIL RGB -> OpenCV BGR
#     return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# def to_rgb(image_cv):
#     return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# def resize_for_analysis(img_rgb, max_side=1024):
#     h, w = img_rgb.shape[:2]
#     if max(h, w) <= max_side:
#         return img_rgb
#     scale = max_side / max(h, w)
#     return cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# # --- Colorfulness (Hasler & Süsstrunk) ---
# def colorfulness_score(img_rgb):
#     img = img_rgb.astype("float")
#     (R, G, B) = (img[:,:,0], img[:,:,1], img[:,:,2])
#     rg = np.abs(R - G)
#     yb = np.abs(0.5 * (R + G) - B)
#     std_rg, mean_rg = np.std(rg), np.mean(rg)
#     std_yb, mean_yb = np.std(yb), np.mean(yb)
#     cf = np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2)
#     # Normalize roughly to 0..100
#     return float(np.clip(cf / 3.0, 0, 100))

# # --- Edge density / complexity ---
# def edge_complexity(img_rgb):
#     gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#     gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(gray_blur, 50, 150)
#     density = edges.mean() * 100.0 / 255.0  # percent of edge pixels
#     return float(np.clip(density*2.0, 0, 100)), edges  # boost a little for visual scale

# # --- Texture entropy / pattern density ---
# def pattern_entropy(img_rgb):
#     gray = rgb2gray(img_rgb)
#     gray_u8 = img_as_ubyte(gray)
#     ent = entropy(gray_u8, disk(5))
#     score = float(np.clip(ent.mean()*10, 0, 100))  # scale to 0..100
#     return score, ent

# # --- Dominant colors ---
# def dominant_colors(img_rgb, k=5):
#     small = cv2.resize(img_rgb, (160, 160), interpolation=cv2.INTER_AREA)
#     data = small.reshape(-1, 3)
#     kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
#     labels = kmeans.fit_predict(data)
#     centers = kmeans.cluster_centers_.astype(np.uint8)
#     counts = np.bincount(labels, minlength=k).astype(float)
#     counts /= counts.sum()
#     return centers, counts

# # --- Fit tightness (heuristic using person mask) ---
# def _mask_with_mediapipe(img_rgb):
#     try:
#         import mediapipe as mp
#         mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
#         with mp_selfie as model:
#             res = model.process(img_rgb)
#             mask = (res.segmentation_mask > 0.5).astype(np.uint8)*255
#             return mask
#     except Exception:
#         return None

# def fit_tightness(img_rgb):
#     # Try to segment the person; fall back to central band analysis if not available
#     mask = _mask_with_mediapipe(img_rgb)
#     if mask is None:
#         h, w = img_rgb.shape[:2]
#         band = img_rgb[int(h*0.35):int(h*0.8), int(w*0.2):int(w*0.8)]
#         edges = cv2.Canny(cv2.cvtColor(band, cv2.COLOR_RGB2GRAY), 60, 180)
#         density = edges.mean()/255.0
#         score = float(np.clip(100 - (density*120), 0, 100))  # more edges -> looser
#         return score, None

#     # Compute solidity of the biggest segmented region (person)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return 50.0, None
#     c = max(contours, key=cv2.contourArea)
#     area = cv2.contourArea(c)
#     hull = cv2.convexHull(c)
#     hull_area = cv2.contourArea(hull) + 1e-6
#     solidity = float(area / hull_area)
#     # Tight fit -> smoother silhouette -> higher solidity
#     score = float(np.clip(solidity * 100, 0, 100))
#     return score, mask

# def aggregate_scores(img_rgb):
#     img_rgb = resize_for_analysis(img_rgb)
#     color = colorfulness_score(img_rgb)
#     complexity, edges = edge_complexity(img_rgb)
#     pattern, ent = pattern_entropy(img_rgb)
#     fit, mask = fit_tightness(img_rgb)

#     # Derive "vibes"
#     minimalist_vs_maximalist = np.clip((complexity*0.5 + pattern*0.35 + color*0.15), 0, 100)
#     boldness = np.clip((color*0.6 + complexity*0.2 + pattern*0.2), 0, 100)

#     return {
#         "colorfulness": color,
#         "complexity": complexity,
#         "pattern_density": pattern,
#         "fit_tightness": fit,
#         "vibe_maximalism": float(minimalist_vs_maximalist),
#         "boldness": float(boldness),
#         "debug": {
#             "edges": edges,
#             "entropy": ent,
#             "mask": mask
#         },
#         "processed_rgb": img_rgb
#     }
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import cv2  # make sure cv2 is imported in the file already
import numpy as np


def resize_for_analysis(img, max_size=512):
    """
    Resize the image so its largest dimension = max_size (default 512px).
    Keeps aspect ratio intact.
    """
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:  # shrink only if larger than max_size
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def pattern_entropy(img_rgb, mask=None):
    """
    Compute a more robust pattern-density score.
    - Applies a small blur to reduce micro-texture noise (denim threads).
    - Uses a larger entropy disk so only larger patterns register strongly.
    - If a person/mask is provided, computes entropy on the bounding box of the mask
      (focuses on clothing area) and returns a full-size entropy map for debugging.
    Returns:
        score (0..100), entropy_map (uint8 or float)
    """
    try:
        # grayscale (float 0..1) then convert to u8 for skimage.rank.entropy
        gray = rgb2gray(img_rgb)
        # reduce high-frequency noise (denim grain, camera noise)
        gray_u8 = img_as_ubyte(cv2.GaussianBlur((gray * 255).astype(np.uint8), (5, 5), 0))
    except Exception:
        # fallback simple conversion
        gray_u8 = img_as_ubyte(rgb2gray(img_rgb))

    # choose a larger neighborhood so only larger repeated motifs count as patterns
    ent_disk = disk(9)

    if mask is not None:
        # ensure mask is same size and boolean
        try:
            if mask.shape != gray_u8.shape:
                mask_resized = cv2.resize(mask, (gray_u8.shape[1], gray_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            mask_bool = (mask_resized > 0)
        except Exception:
            mask_bool = None

        if mask_bool is None or mask_bool.sum() < 100:
            # mask useless — compute on whole image
            ent = entropy(gray_u8, ent_disk)
            ent_map = ent
        else:
            # compute entropy on the bounding box of the mask to focus on clothing
            ys, xs = np.where(mask_bool)
            y1, y2 = max(0, ys.min()), min(gray_u8.shape[0] - 1, ys.max())
            x1, x2 = max(0, xs.min()), min(gray_u8.shape[1] - 1, xs.max())

            patch = gray_u8[y1:y2+1, x1:x2+1]
            # if patch is too small, fallback
            if patch.size < 100:
                ent = entropy(gray_u8, ent_disk)
                ent_map = ent
            else:
                ent_patch = entropy(patch, ent_disk)
                # put patch back into full-size map (for debugging/display)
                ent_map = np.zeros_like(gray_u8, dtype=ent_patch.dtype)
                ent_map[y1:y2+1, x1:x2+1] = ent_patch
    else:
        ent_map = entropy(gray_u8, ent_disk)

    # Score: scale mean entropy to 0..100, using a conservative multiplier so denim noise doesn't blow up
    score = float(np.clip(ent_map.mean() * 6.5, 0, 100))

    return score, ent_map

def colorfulness_score(img_rgb):
    """
    Measures how colorful the image is, using a standard metric (Hasler & Süsstrunk).
    Returns 0–100.
    """
    (B, G, R) = cv2.split(img_rgb.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))

    score = std_root + (0.3 * mean_root)
    return float(np.clip(score, 0, 100))


def edge_complexity(img_rgb):
    """
    Measures visual complexity based on edges (Canny detector).
    Returns 0–100 + edge map for debugging.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    score = 100.0 * (edges > 0).sum() / edges.size
    return float(np.clip(score, 0, 100)), edges


def fit_tightness(img_rgb):
    """
    Estimate fit tightness by segmenting person using GrabCut.
    - Returns tightness score (0..100) and mask (binary).
    """
    h, w = img_rgb.shape[:2]

    # initial mask
    mask = np.zeros((h, w), np.uint8)

    # bounding box covering most of image
    rect = (10, 10, w-20, h-20)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    except Exception:
        mask2 = np.ones((h, w), dtype=np.uint8)  # fallback

    # compute bounding box of segmented person
    ys, xs = np.where(mask2 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0, mask2

    bbox_h = ys.max() - ys.min()
    bbox_w = xs.max() - xs.min()

    # heuristic: tighter = tall narrow box
    aspect_ratio = bbox_h / float(bbox_w + 1e-6)
    tightness = np.clip(aspect_ratio * 50, 0, 100)

    return float(tightness), mask2 * 255  # return 0/255 mask for visualization


def aggregate_scores(img_rgb):
    """
    Reordered so we compute fit/mask first and then pattern_entropy(masked).
    """
    img_rgb = resize_for_analysis(img_rgb)

    # color + complexity first
    color = colorfulness_score(img_rgb)
    complexity, edges = edge_complexity(img_rgb)

    # compute fit and mask earlier so pattern density uses the mask if available
    fit, mask = fit_tightness(img_rgb)

    # pattern now can use mask (if available) to focus on clothing region
    pattern, ent = pattern_entropy(img_rgb, mask=mask)

    # remaining derived scores
    # Derive "vibes"
    minimalist_vs_maximalist = np.clip((complexity * 0.5 + pattern * 0.35 + color * 0.15), 0, 100)
    boldness = np.clip((color * 0.6 + complexity * 0.2 + pattern * 0.2), 0, 100)

    return {
        "colorfulness": float(color),
        "complexity": float(complexity),
        "pattern_density": float(pattern),
        "fit_tightness": float(fit),
        "vibe_maximalism": float(minimalist_vs_maximalist),
        "boldness": float(boldness),
        "debug": {
            "edges": edges,
            "entropy": ent,
            "mask": mask
        },
        "processed_rgb": img_rgb
    }