#!/bin/bash

# 设置测试集名称，这里可以修改为 test1、test2 等
TEST_NAME="ispd18_test6"

declare -a via_counts=()
declare -a wire_lengths=()

for i in $(seq 0 15); do
    DGR="../../CUGR2_guide/CUgr_${TEST_NAME}_DGR_${i}.txt"

    output=$(./route \
        -via_cost 22 \
        -lef "../benchmark/${TEST_NAME}/${TEST_NAME}.input.lef" \
        -def "../benchmark/${TEST_NAME}/${TEST_NAME}.input.def" \
        -sort 1 \
        -output "../benchmark/${TEST_NAME}/${TEST_NAME}.DGR.guide" \
        -dgr "$DGR" 2>&1)

    # 去掉可能的回车符
    output=$(echo "$output" | tr -d '\r')

    echo "第 $i 轮匹配到的行:"
    printf '%s\n' "$output" | grep "total via count:"
    printf '%s\n' "$output" | grep "wire length (metric):"
    printf '%s\n' "$output" | grep "total wire overflow:"

    # 提取 total via count 的数值（整数）
    via_count=$(echo "$output" | grep "total via count:" | sed -E 's/.*total via count:[[:space:]]*([0-9]+).*/\1/')

    # 提取 wire length (metric) 的数值（浮点数）
    wire_length=$(echo "$output" | grep "wire length (metric):" | sed -E 's/.*wire length \(metric\):[[:space:]]*([0-9.]+).*/\1/')

    if [[ -n "$via_count" && "$via_count" =~ ^[0-9]+$ ]]; then
        echo "第 $i 轮提取到 total via count: $via_count"
        via_counts+=("$via_count")
    else
        echo "第 $i 轮未能提取到有效的 total via count 数值！"
    fi

    if [[ -n "$wire_length" && "$wire_length" =~ ^[0-9.]+$ ]]; then
        echo "第 $i 轮提取到 wire length (metric): $wire_length"
        wire_lengths+=("$wire_length")
    else
        echo "第 $i 轮未能提取到有效的 wire length (metric) 数值！"
    fi
done

# 计算 total via count 平均值（整数）
if [ ${#via_counts[@]} -gt 0 ]; then
    sum=0
    for val in "${via_counts[@]}"; do
        sum=$((sum + val))
    done
    avg_via=$((sum / ${#via_counts[@]}))
else
    avg_via="N/A"
fi

# 计算 wire length (metric) 平均值（浮点数）
if [ ${#wire_lengths[@]} -gt 0 ]; then
    sum_wire=0
    for val in "${wire_lengths[@]}"; do
        # 使用 bc 进行加法浮点计算
        sum_wire=$(echo "$sum_wire + $val" | bc)
    done
    count_wire=${#wire_lengths[@]}
    avg_wire=$(echo "scale=4; $sum_wire / $count_wire" | bc)
else
    avg_wire="N/A"
fi

echo "---------------------------------------------"
echo "total via count 数值列表：${via_counts[@]}"
echo "average total via count: $avg_via"
echo
echo "wire length (metric) 数值列表：${wire_lengths[@]}"
echo "average wire length (metric): $avg_wire"
