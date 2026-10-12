#!/bin/bash

# 设置测试集名称，这里可以修改为 test1、test2 等
TEST_NAME="ispd19_test10"

declare -a via_counts=()

for i in $(seq 0 15); do
    DGR="../../CUGR2_guideZ/CUgr_${TEST_NAME}_DGR_${i}.txt"

    # 使用 process substitution 将 ./route 的输出同时保存到变量和打印到终端
    # 方案1: 利用 tee 和临时文件
    tmpfile=$(mktemp)
    ./route \
        -via_cost 20 \
        -lef "../benchmark/${TEST_NAME}/${TEST_NAME}.input.lef" \
        -def "../benchmark/${TEST_NAME}/${TEST_NAME}.input.def" \
        -sort 1 \
        -output "../benchmark/${TEST_NAME}/${TEST_NAME}.DGR.guide" \
        -dgr "$DGR" 2>&1 | tee "$tmpfile"

    # 读取 tee 保存的输出
    output=$(cat "$tmpfile")
    rm -f "$tmpfile"

    # 去掉可能的回车符
    output=$(echo "$output" | tr -d '\r')

    echo "第 $i 轮匹配到的行:"
    printf '%s\n' "$output" | grep "total via count:"
    printf '%s\n' "$output" | grep "wire length (metric):"
    printf '%s\n' "$output" | grep "total wire overflow:"

    # 提取 total via count 的数值
    via_count=$(echo "$output" | grep "total via count:" | sed -E 's/.*total via count:[[:space:]]*([0-9]+).*/\1/')

    if [[ -n "$via_count" && "$via_count" =~ ^[0-9]+$ ]]; then
        echo "第 $i 轮提取到 total via count: $via_count"
        via_counts+=("$via_count")
    else
        echo "第 $i 轮未能提取到有效的 total via count 数值！"
    fi
done

if [ ${#via_counts[@]} -gt 0 ]; then
    sum=0
    for val in "${via_counts[@]}"; do
        sum=$((sum + val))
    done
    avg=$((sum / ${#via_counts[@]}))
    echo "total via count 数值列表：${via_counts[@]}"
    echo "平均 total via count：$avg"
else
    echo "未提取到任何 total via count 数值"
fi
