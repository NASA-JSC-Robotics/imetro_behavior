# Copyright (c) 2026, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# All rights reserved.
#
# This software is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Build API docs with pdoc, mocking unavailable dependencies."""

import pdoc
import sys

from pathlib import Path
from unittest.mock import MagicMock

# Mock any modules that aren't available, many of these are pulled
# in my color_tools, but mocking them still lets that module have documentation.
mocks = ["cv2", "cv_bridge", "color_blob_centroid", "color_blob_centroid.bindings"]
for mod in mocks:
    sys.modules[mod] = MagicMock()

# Module paths
module_paths = [
    "src/imetro_behavior/imetro_behavior",
]

pdoc.render.configure(docformat="google")
pdoc.pdoc(
    *module_paths,
    output_directory=Path("docs/_site"),
)
