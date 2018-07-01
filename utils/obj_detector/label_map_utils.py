# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Label map utility functions."""

import re

"""
item {
  name: "/m/03120"
  id: 3
  display_name: "Flag"
}
"""


def string_to_label_map(str_pbtxt):
    label_map_dics = []
    str_items = str_pbtxt.split("item")
    cnt = 0
    for str_item in str_items:
        if str_item.find("display_name") == -1:
            continue
        try:
            cnt += 1
            str_item = str_item.replace('\n', '')
            str_item = re.sub('[{}]', '', str_item)

            _str = str_item
            sps = _str.split("display_name:")
            display_name = re.sub('["]', '', sps[-1])

            _str = sps[0]
            sps = _str.split("id:")
            id = int(sps[-1].replace(' ', ''))

            _str = sps[0]
            sps = _str.split("name:")
            name = re.sub('[" ]', '', sps[-1])

            if id == cnt:
                label_map_dics.append({
                    "name": name,
                    "id": id,
                    "display_name": display_name
                })
            else:
                print(str_item)
        except Exception as e:
            print(e)
            continue
    return label_map_dics
