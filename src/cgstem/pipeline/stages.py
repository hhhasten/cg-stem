"""Стадії обробки зображення."""

from __future__ import annotations

from cgstem.methods.filtering import gaussian_blur, median_filter
from cgstem.methods.histogram import clahe
from cgstem.methods.morphology import closing, opening
from cgstem.methods.segmentation import otsu_mask
from cgstem.methods.texture import glcm_features
from cgstem.metrics.mask import (
    count_components,
    foreground_fraction,
    largest_component_area,
)
from cgstem.metrics.mse import compute_mse
from cgstem.metrics.psnr import compute_psnr
from cgstem.metrics.ssim import compute_ssim
from cgstem.pipeline.result import MethodResult
from cgstem.types import GrayImage, Mask


def process_image(image: GrayImage, config: dict) -> list[MethodResult]:
    """Виконує всі вибрані стадії для одного знімка."""
    results = filtering_results(image, config)
    if "otsu" in config or "morphology" in config:
        mask = otsu_mask(image)
        if "otsu" in config:
            results.append(segmentation_result(mask))
        if "morphology" in config:
            results.extend(morphology_results(mask, config))
    results.extend(texture_results(image, config))
    return results


def filtering_results(image: GrayImage, config: dict) -> list[MethodResult]:
    """Застосовує фільтри та рахує метрики відносно оригіналу."""
    results = []
    if "median" in config:
        for size in config["median"]["sizes"]:
            results.append(filter_result("median", f"size{size}",
                                         median_filter(image, size), image))
    if "gaussian" in config:
        size = config["gaussian"]["size"]
        sigma = config["gaussian"]["sigma"]
        results.append(filter_result("gaussian", f"size{size}_sigma{sigma}",
                                     gaussian_blur(image, size, sigma), image))
    if "clahe" in config:
        clip_limit = config["clahe"]["clip_limit"]
        grid = config["clahe"]["grid"]
        results.append(filter_result("clahe", f"clip{clip_limit}_grid{grid[0]}x{grid[1]}",
                                     clahe(image, clip_limit, tuple(grid)), image))
    return results


def filter_result(method: str, variant: str, output: GrayImage,
                  original: GrayImage) -> MethodResult:
    """Формує результат фільтрації з PSNR, MSE та SSIM."""
    metrics = {
        "mse": compute_mse(original, output),
        "psnr": compute_psnr(original, output),
        "ssim": compute_ssim(original, output),
    }
    return MethodResult(method, variant, "filtering", output, metrics)


def segmentation_result(mask: Mask) -> MethodResult:
    """Формує результат сегментації з показниками маски."""
    return MethodResult("otsu", "default", "segmentation", mask, mask_metrics(mask))


def morphology_results(mask: Mask, config: dict) -> list[MethodResult]:
    """Застосовує відкриття і закриття до маски."""
    size = config["morphology"]["size"]
    results = []
    for method, operation in (("opening", opening), ("closing", closing)):
        output = operation(mask, size)
        results.append(MethodResult(method, f"size{size}", "morphology",
                                    output, mask_metrics(output)))
    return results


def texture_results(image: GrayImage, config: dict) -> list[MethodResult]:
    """Рахує усереднені властивості GLCM."""
    if "glcm" not in config:
        return []
    parameters = config["glcm"]
    distances = tuple(parameters["distances"])
    angles = tuple(parameters["angles"])
    features = glcm_features(image, distances, angles)
    variant = f"d{join_values(distances)}_a{join_values(angles)}"
    return [MethodResult("glcm", variant, "texture", None, features)]


def mask_metrics(mask: Mask) -> dict[str, float]:
    """Рахує частку переднього плану, компоненти та найбільшу площу."""
    metrics = {
        "foreground": foreground_fraction(mask),
        "components": float(count_components(mask)),
        "largest": float(largest_component_area(mask)),
    }
    return metrics


def join_values(values: tuple[int, ...]) -> str:
    """З'єднує числа через дефіс для назви варіанта."""
    return "-".join(str(value) for value in values)
