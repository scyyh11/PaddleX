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

import json

import cv2
import numpy as np

from paddlex import create_pipeline
from paddlex.inference.pipelines import load_pipeline_config


def main():
    height, width = 112, 512
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        image, "PADDLE OCR", (26, 74), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4
    )

    config = load_pipeline_config("OCR")
    config["use_doc_preprocessor"] = False
    config["use_textline_orientation"] = False
    config["SubModules"]["TextDetection"]["model_name"] = "PP-OCRv5_mobile_det"
    config["SubModules"]["TextRecognition"]["model_name"] = "en_PP-OCRv5_mobile_rec"
    pipeline = create_pipeline(
        config=config,
        device="cpu",
        engine="paddle_static",
        engine_config={"run_mode": "paddle", "cpu_threads": 2},
    )

    (result,) = pipeline.predict(image)
    texts = result["rec_texts"]
    polygons = np.asarray(result["rec_polys"])
    scores = np.asarray(result["rec_scores"])

    assert "".join(texts).replace(" ", "").upper() == "PADDLEOCR", texts
    assert polygons.shape == (len(texts), 4, 2), polygons
    assert np.all((0 <= polygons) & (polygons <= (width, height))), polygons
    assert np.all(np.ptp(polygons, axis=1) > 0), polygons
    assert scores.shape == (len(texts),), scores
    assert np.all((0 <= scores) & (scores <= 1)), scores

    print(
        json.dumps(
            {
                "numpy": np.__version__,
                "texts": texts,
                "polygons": polygons.tolist(),
                "scores": scores.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
