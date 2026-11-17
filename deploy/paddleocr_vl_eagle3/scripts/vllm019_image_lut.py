# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in exact float32 PIL preprocessing for the native PaddleOCR-VL processor.

The native resize, rescale and normalize operations define this path. Only
parameter-dependent 256-value normalization tables are cached, never images or
image results. Unsupported inputs go through the supplied processor unchanged.
Nothing is installed or patched when this module is imported.
"""
from __future__ import annotations

from functools import lru_cache, wraps
import importlib
from pathlib import Path
import threading
import weakref

import numpy as np
from PIL import Image
from transformers.image_utils import ChannelDimension


_INSTALLED = {}
_INSTALL_LOCK = threading.RLock()


def make_exact_pil_preprocess(processor, *, original=None, counters=None):
    """Return an independent callable; do not mutate the supplied processor.

    Pass the image_processor owned by the existing native AutoProcessor. The
    returned callable has the same keyword contract as its preprocess method.
    """
    if processor.__class__.__name__ != "PaddleOCRVLImageProcessor":
        raise TypeError("Expected the native PaddleOCRVLImageProcessor")
    module = importlib.import_module(processor.__class__.__module__)
    original = processor.preprocess if original is None else original
    counts = (
        counters
        if counters is not None
        else {"fast_calls": 0, "fallback_calls": 0, "lookup_builds": 0}
    )

    @lru_cache(maxsize=32)
    def lookup(do_rescale, scale, do_normalize, mean, std, parameter_bits):
        values = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)
        if do_rescale:
            values = processor.rescale(
                values, scale=scale, input_data_format=ChannelDimension.LAST
            )
        if do_normalize:
            values = processor.normalize(
                values, mean=mean, std=std, input_data_format=ChannelDimension.LAST
            )
        counts["lookup_builds"] += 1
        return np.ascontiguousarray(values[:, 0, :])

    @wraps(original)
    def preprocess(
        images,
        videos=None,
        do_resize=None,
        size=None,
        resample=None,
        do_rescale=None,
        rescale_factor=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        do_convert_rgb=None,
        return_tensors=None,
        data_format=ChannelDimension.FIRST,
        input_data_format=None,
        **extra_kwargs,
    ):
        arguments = dict(
            images=images,
            videos=videos,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            return_tensors=return_tensors,
            data_format=data_format,
            input_data_format=input_data_format,
            **extra_kwargs,
        )

        def fallback():
            counts["fallback_calls"] += 1
            return original(**arguments)

        # Preserve native rejection or decorator filtering/warnings verbatim.
        if extra_kwargs:
            return fallback()
        image = images
        # Native make_batched_images accepts at most two list levels.
        for _ in range(2):
            if isinstance(image, (list, tuple)) and len(image) == 1:
                image = image[0]
        if (
            videos is not None
            or not isinstance(image, Image.Image)
            or image.mode != "RGB"
            or processor.temporal_patch_size != 1
            or input_data_format not in (None, ChannelDimension.LAST, "channels_last")
            or data_format
            not in (
                ChannelDimension.FIRST,
                ChannelDimension.LAST,
                "channels_first",
                "channels_last",
            )
            or return_tensors
            not in (
                None,
                "pt",
                "np",
                module.TensorType.PYTORCH,
                module.TensorType.NUMPY,
            )
        ):
            return fallback()

        def configured(value, name):
            return getattr(processor, name) if value is None else value

        do_resize = configured(do_resize, "do_resize")
        size = configured(size, "size")
        resample = configured(resample, "resample")
        do_rescale = configured(do_rescale, "do_rescale")
        rescale_factor = configured(rescale_factor, "rescale_factor")
        do_normalize = configured(do_normalize, "do_normalize")
        image_mean = configured(image_mean, "image_mean")
        image_std = configured(image_std, "image_std")
        module.validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # Handle unusual parameter types through the native method. Numeric
        # scalars and standard three-channel collections cover production.
        def triple(value):
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != 3:
                    return None
                return tuple(value)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return (value,) * 3
            return None

        mean, std = triple(image_mean), triple(image_std)
        if mean is None or std is None:
            return fallback()
        try:
            numeric_parameters = tuple(
                np.asarray(value) for value in (rescale_factor, mean, std)
            )
            if any(value.dtype.kind not in "biuf" for value in numeric_parameters):
                return fallback()
            # Python numeric equality merges +0.0 and -0.0. Include their bit
            # patterns so switching normalization parameters cannot reuse a
            # numerically equal table with different signed-zero output bits.
            parameter_bits = tuple(
                (value.dtype.str, value.shape, value.tobytes())
                for value in numeric_parameters
            )
            key = (
                bool(do_rescale),
                rescale_factor,
                bool(do_normalize),
                mean,
                std,
                parameter_bits,
            )
            hash(key)
        except TypeError:
            return fallback()
        width, height = image.size
        if do_resize:
            height, width = module.smart_resize(
                height,
                width,
                factor=processor.patch_size * processor.merge_size,
                min_pixels=processor.min_pixels,
                max_pixels=processor.max_pixels,
            )
            # Native transformers.resize reconstructs this same RGB PIL image
            # from uint8, then calls PIL.Image.resize with reducing_gap=None.
            image = image.resize((width, height), resample=resample, reducing_gap=None)
        patch = processor.patch_size
        if height % patch or width % patch:
            return fallback()
        table = lookup(*key)
        pixels = np.asarray(image)
        gh, gw = height // patch, width // patch
        # Rearrange uint8 first, avoiding a second full-sized float32 copy.
        packed = np.ascontiguousarray(
            pixels.reshape(gh, patch, gw, patch, 3).transpose(0, 2, 4, 1, 3)
        ).reshape(gh * gw, 3, patch, patch)
        if table[:, 0].tobytes() == table[:, 1].tobytes() == table[:, 2].tobytes():
            patches = table[:, 0][packed]
        else:
            patches = table[packed, np.arange(3)[None, :, None, None]]
        counts["fast_calls"] += 1
        return module.BatchFeature(
            data={
                "pixel_values": patches,
                "image_grid_thw": np.array([[1, gh, gw]], dtype=np.int64),
            },
            tensor_type=return_tensors,
        )

    preprocess.profile_counts = counts
    preprocess.lookup_cache_info = lookup.cache_info
    return preprocess


def install(model_path=None, *, image_processor=None):
    """Explicitly patch the native processor class in this process only.

    Call only for the accelerated service, before creating its AsyncLLM/renderer.
    No import-time activation, environment variable, package edit, or worker
    inference patch is used. ``image_processor`` may be an existing instance;
    otherwise ``model_path`` loads the same local native AutoProcessor class.
    Existing and future instances of that class receive the guarded fast path.
    """
    if (model_path is None) == (image_processor is None):
        raise ValueError("Supply exactly one of model_path or image_processor")
    if image_processor is None:
        from transformers import AutoProcessor

        image_processor = AutoProcessor.from_pretrained(
            Path(model_path), trust_remote_code=True, local_files_only=True
        ).image_processor
    cls = type(image_processor)
    if cls.__name__ != "PaddleOCRVLImageProcessor":
        raise TypeError("Expected the native PaddleOCRVLImageProcessor")
    with _INSTALL_LOCK:
        if cls in _INSTALLED:
            return metadata()
        native = cls.preprocess
        instances = weakref.WeakKeyDictionary()
        counters = {"fast_calls": 0, "fallback_calls": 0, "lookup_builds": 0}

        @wraps(native)
        def preprocess(self, *args, **kwargs):
            with _INSTALL_LOCK:
                fast = instances.get(self)
                if fast is None:
                    reference = weakref.ref(self)

                    @wraps(native)
                    def native_call(*native_args, **native_kwargs):
                        instance = reference()
                        if instance is None:
                            raise ReferenceError("Image processor was released")
                        return native(instance, *native_args, **native_kwargs)

                    # Neither the callable nor its lookup cache keeps a strong
                    # reference to the instance. Avoid adding nonserializable
                    # callable fields to the processor's configuration state.
                    fast = make_exact_pil_preprocess(
                        weakref.proxy(self), original=native_call, counters=counters
                    )
                    instances[self] = fast
            return fast(*args, **kwargs)

        cls.preprocess = preprocess
        _INSTALLED[cls] = dict(instances=instances, counters=counters)
    return metadata()


def metadata():
    """Parent-process runtime counters, including processor instances with overrides."""
    with _INSTALL_LOCK:
        classes = []
        totals = {"fast_calls": 0, "fallback_calls": 0, "lookup_builds": 0}
        for cls, state in _INSTALLED.items():
            classes.append(
                dict(
                    class_name=cls.__name__,
                    module=cls.__module__,
                    live_instances=len(state["instances"]),
                )
            )
            for name in totals:
                totals[name] += state["counters"][name]
    return dict(
        enabled=bool(classes),
        classes=classes,
        **totals,
        output_dtype="native processor dtype; normally float32",
        resize="same PIL.Image.resize and smart_resize as native",
        cache="parameter-only 256-value lookup tables; no image results",
    )
