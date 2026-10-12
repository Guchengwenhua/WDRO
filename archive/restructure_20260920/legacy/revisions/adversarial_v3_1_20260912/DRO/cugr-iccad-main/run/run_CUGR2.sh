#!/bin/bash

# 定义基础路径和数据文件的前缀
BASE_PATH="/remote-home/Chenlin/chip/routing/my_Diff2/cu-gr-2-main/benchmark"
DATA_PREFIX="ispd18_test"

# 循环遍历数据集test1到test10
for i in {1..10}; do
    TEST_NUMBER="${DATA_PREFIX}${i}"

    # 构建文件路径
    LEF_PATH="${BASE_PATH}/${TEST_NUMBER}/${TEST_NUMBER}.input.lef"
    DEF_PATH="${BASE_PATH}/${TEST_NUMBER}/${TEST_NUMBER}.input.def"
    OUTPUT_PATH="${BASE_PATH}/${TEST_NUMBER}/${TEST_NUMBER}.output2"

    # 检查输入文件是否存在
    if [[ ! -f "$LEF_PATH" || ! -f "$DEF_PATH" ]]; then
        echo "Files for ${TEST_NUMBER} do not exist. Skipping..."
        continue
    fi

    # 运行程序
    echo "Running test for ${TEST_NUMBER}..."
    "/remote-home/Chenlin/chip/routing/my_Diff2/cu-gr-2-main/run/route" \
        -lef "$LEF_PATH" \
        -def "$DEF_PATH" \
        -output "$OUTPUT_PATH" \
        -threads "1" \
        -sort "1"

    if [[ $? -ne 0 ]]; then
        echo "Error occurred while running test for ${TEST_NUMBER}."
    else
        echo "Test for ${TEST_NUMBER} completed successfully."
    fi
done
