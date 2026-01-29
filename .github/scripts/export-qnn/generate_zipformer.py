#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import json

from device_info import soc_info_dict
from dataclasses import asdict, dataclass
import itertools


@dataclass
class Config:
    soc: str  # SM8850
    soc_id: int  # 87
    arch: str  # v81
    input_in_seconds: str
    model_name: str


def main():

    input_in_seconds = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    ]
    model_name_list = ["20250703", "20251222"]

    configs = []

    for name, soc in soc_info_dict.items():
        for num_seconds, model_name in itertools.product(
            input_in_seconds, model_name_list
        ):
            configs.append(
                Config(
                    soc=name,
                    soc_id=soc.model.value,
                    arch=soc.info.arch.name,
                    input_in_seconds=num_seconds,
                    model_name=model_name,
                )
            )

    ans = [asdict(c) for c in configs]

    print(json.dumps({"include": ans}))


if __name__ == "__main__":
    main()
