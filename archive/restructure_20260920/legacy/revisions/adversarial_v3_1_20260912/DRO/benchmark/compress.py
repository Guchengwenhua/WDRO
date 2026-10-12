import os
import fnmatch
import re

root = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(root, 'merged')
os.makedirs(out_dir, exist_ok=True)

degree_blocks = {}  # degree -> list of net blocks (strings)

for entry in os.listdir(root):
    path = os.path.join(root, entry)
    if not os.path.isdir(path):
        continue
    for fname in os.listdir(path):
        if fnmatch.fnmatch(fname, 'degree_*.txt'):
            m = re.match(r'degree_(\d+)\.txt$', fname)
            if not m:
                continue
            degree = int(m.group(1))
            fpath = os.path.join(path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception:
                # 尝试用 latin-1 回退以兼容非 utf-8 文件
                with open(fpath, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            i = 0
            blocks = degree_blocks.setdefault(degree, [])
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith('net '):
                    block_lines = [line]
                    i += 1
                    # 读取直到 END（包含 END）
                    while i < len(lines) and lines[i].strip() != 'END':
                        block_lines.append(lines[i])
                        i += 1
                    if i < len(lines) and lines[i].strip() == 'END':
                        block_lines.append(lines[i])
                        i += 1
                    # 确保以一个换行分隔不同 net 块
                    if not block_lines[-1].endswith('\n'):
                        block_lines[-1] = block_lines[-1] + '\n'
                    blocks.append(''.join(block_lines))
                else:
                    i += 1

# 写出每个 degree 的合并文件
for degree in sorted(degree_blocks.keys()):
    blocks = degree_blocks[degree]
    out_path = os.path.join(out_dir, f'degree_{degree}.txt')
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write(str(len(blocks)) + '\n')
        for b in blocks:
            out_f.write(b)
            # 保证每个 net 之间至少一个空行（可选）
            out_f.write('\n')
print(f'Merged {len(degree_blocks)} degree files into: {out_dir}')