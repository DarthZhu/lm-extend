# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

model_name="Qwen2-merged-weighted-all"

python3 -m evaluation_main \
  --input_data=/nas02/thzhu/lm_extend_analysis/src/evaluations/instruction_following_eval/data/input_data.jsonl \
  --input_response_data=/nas02/thzhu/lm_extend_analysis/output/$model_name/ifeval.txt \
  --output_dir=/nas02/thzhu/lm_extend_analysis/output/$model_name/

exit 0