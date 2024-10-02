# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash


#weights to be included
declare -a arr_preproc=(
    "last_model_epoch_54.pt"
)


for f in "${arr_preproc[@]}"; do
    echo "Download: ${f}"
    curl --create-dirs -o "checkpoints/${f}" "https://hub.dkfz.de/s/59WPXY3ZiaktRbS/${f// /%20}"
done
