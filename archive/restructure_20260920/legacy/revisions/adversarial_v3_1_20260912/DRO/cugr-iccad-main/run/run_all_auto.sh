#!/bin/bash

# 设置测试集名称数组
TEST_NAMES=(
  "ispd19_test6"
  "ispd19_test7"
  "ispd19_test8"
  "ispd19_test9"
  "ispd19_test10"
)

# 指定日志文件夹
LOG_DIR="route_logs"

# 如果日志文件夹不存在，创建它
mkdir -p "$LOG_DIR"

for TEST_NAME in "${TEST_NAMES[@]}"; do
    echo "处理数据集：$TEST_NAME"
    declare -a via_counts=()

    # 定义存放输出的文件，放在日志文件夹下
    OUTPUT_FILE="${LOG_DIR}/${TEST_NAME}_route_output_best.log"
    # 先清空文件（如果存在）
    > "$OUTPUT_FILE"

    for i in $(seq 0 0); do
        DGR="../../CUGR2_guideZ/CUgr_${TEST_NAME}_DGR_${i}.txt"

        echo "第 $i 轮运行 ./route，输出写入 $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"

        ./route \
            -via_cost 21 \
            -lef "../benchmark/${TEST_NAME}/${TEST_NAME}.input.lef" \
            -def "../benchmark/${TEST_NAME}/${TEST_NAME}.input.def" \
            -sort 1 \
            -output "../benchmark/${TEST_NAME}/${TEST_NAME}.DGR.guide" \
            -dgr "$DGR" 2>&1 | tee -a "$OUTPUT_FILE"

        echo "第 $i 轮匹配到的行:" | tee -a "$OUTPUT_FILE"
        grep "total via count:" "$OUTPUT_FILE" | tail -n 5 | tee -a "$OUTPUT_FILE"
        grep "wire length (metric):" "$OUTPUT_FILE" | tail -n 5 | tee -a "$OUTPUT_FILE"
        grep "total wire overflow:" "$OUTPUT_FILE" | tail -n 5 | tee -a "$OUTPUT_FILE"

        via_count=$(grep "total via count:" "$OUTPUT_FILE" | tail -n 1 | sed -E 's/.*total via count:[[:space:]]*([0-9]+).*/\1/')

        if [[ -n "$via_count" && "$via_count" =~ ^[0-9]+$ ]]; then
            echo "第 $i 轮提取到 total via count: $via_count" | tee -a "$OUTPUT_FILE"
            via_counts+=("$via_count")
        else
            echo "第 $i 轮未能提取到有效的 total via count 数值！" | tee -a "$OUTPUT_FILE"
        fi
    done

    if [ ${#via_counts[@]} -gt 0 ]; then
        sum=0
        for val in "${via_counts[@]}"; do
            sum=$((sum + val))
        done
        avg=$((sum / ${#via_counts[@]}))
        echo "total via count 数值列表：${via_counts[@]}" | tee -a "$OUTPUT_FILE"
        echo "平均 total via count：$avg" | tee -a "$OUTPUT_FILE"
    else
        echo "未提取到任何 total via count 数值" | tee -a "$OUTPUT_FILE"
    fi

    echo "完成 $TEST_NAME 的处理，输出保存在 $OUTPUT_FILE"
    echo "==============================================" | tee -a "$OUTPUT_FILE"
done
