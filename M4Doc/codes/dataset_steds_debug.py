import os
import re
import argparse
from collections import Counter

from zss import simple_distance, Node


base_dir="/home/pj24001974/ku60000761/project/test_project/M4Doc"

def pre_clean(text):
    text = re.sub(r'<bos>|<eos>|<pad>|<unk>', '', text)
    text = re.sub(r'\s##(\S)', r'\1', text)
    text = re.sub(r'\\\s', r'\\', text)
    text = re.sub(r'\s\*\s\*\s', r'**', text)
    text = re.sub(r'{\s', r'{', text)
    text = re.sub(r'\s}', r'}', text)
    text = re.sub(r'\\begin\s', r'\\begin', text)
    text = re.sub(r'\\end\s', r'\\end', text)
    text = re.sub(r'\\end{table}', r'\\end{table} \n\n', text)
    text = text.replace('\n', ' ')
    text = text.replace('*', ' ')
    text = text.replace('_', ' ')
    return text


def normalize_structure(text):
    """
    Normalize obvious heading variants ONLY for debugging.
    This does not modify your original files.
    """
    # Full-width # -> ASCII #
    text = text.replace('＃', '#')

    # Normalize spaced heading markers:
    # "# # # title" -> "### title"
    # "# # title"   -> "## title"
    text = re.sub(
        r'^[ \t]*#\s+#\s+#\s*',
        '### ',
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r'^[ \t]*#\s+#\s*',
        '## ',
        text,
        flags=re.MULTILINE
    )

    return text


def heading_stats(path):
    stats = Counter()
    #path=f"{base_dir}/{path}"
    path="/home/pj24001974/ku60000761/project/test_project/M4Doc/DoTA_dataset/zh_mmd/2001.00004_03.mmd"

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        raw = line.rstrip('\n')
        s = raw.strip()

        if not s:
            stats['blank_lines'] += 1
            continue

        stats['nonempty_lines'] += 1

        if s.startswith('###'):
            stats['ascii_h3_plus'] += 1
        elif s.startswith('##'):
            stats['ascii_h2'] += 1
        elif s.startswith('#'):
            stats['ascii_h1'] += 1

        if s.startswith('＃'):
            stats['fullwidth_heading'] += 1

        if re.match(r'^#\s+#', s):
            stats['spaced_ascii_heading'] += 1

        if re.match(r'^＃\s+＃', s):
            stats['spaced_fullwidth_heading'] += 1

    stats['total_lines'] = len(lines)

    return stats


def print_suspicious_headings(path):
    print("\n=== HEADING-LIKE LINES ===")
    path="/home/pj24001974/ku60000761/project/test_project/M4Doc/DoTA_dataset/zh_mmd/2001.00004_03.mmd"

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            s = line.rstrip('\n')

            if (
                '#' in s
                or '＃' in s
            ):
                print(f"{i:4d}: {repr(s)}")


def get_tree_from_lines(lines):
    tree = Node('ROOT')
    title_node = Node('TITLE')
    tree.addkid(title_node)

    last_title = ''

    for original_line in lines:
        line = pre_clean(original_line)

        if line.strip() == '':
            continue

        line = line.strip()

        if line.startswith('#'):
            child = tree.get('ROOT')

            # Same logic as author's code
            line = line.replace('#', '')

            child.addkid(Node(line))
            last_title = line

        else:
            if last_title == '':
                child = tree.get('TITLE')
                child.addkid(Node(line))
            else:
                child = tree.get(last_title)

                if child is None:
                    print(
                        "[WARNING] Could not find title node:",
                        repr(last_title)
                    )
                else:
                    child.addkid(Node(line))

    return tree

def simple_structure_signature(path):
    path="/home/pj24001974/ku60000761/project/test_project/M4Doc/DoTA_dataset/zh_mmd/2001.00004_03.mmd"
    with open(path, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f if x.strip()]

    signature = []

    for line in lines:
        if line.startswith('#'):
            signature.append('H')
        else:
            signature.append('P')

    return ''.join(signature)



def get_tree(path, normalize=False):
    path="/home/pj24001974/ku60000761/project/test_project/M4Doc/DoTA_dataset/zh_mmd/2001.00004_03.mmd"
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    if normalize:
        text = normalize_structure(text)

    lines = text.splitlines(keepends=True)

    return get_tree_from_lines(lines)


def STEDS(pred_tree, ref_tree):
    def my_distance(pred, ref):
        if len(pred.split()) == 0 or len(ref.split()) == 0:
            return 1
        else:
            return 0

    total_distance = simple_distance(
        pred_tree,
        ref_tree,
        label_dist=my_distance
    )

    pred_nodes = len(list(pred_tree.iter()))
    ref_nodes = len(list(ref_tree.iter()))

    num_of_nodes = max(pred_nodes, ref_nodes)

    score = 1 - total_distance / num_of_nodes

    return score, total_distance, pred_nodes, ref_nodes


def print_tree(node, depth=0, max_text=100):
    text = str(node.label)

    if len(text) > max_text:
        text = text[:max_text] + '...'

    print('  ' * depth + repr(text))

    for child in node.children:
        print_tree(child, depth + 1, max_text=max_text)


def inspect_file(path):
    print("=" * 80)
    print("FILE:", path)

    stats = heading_stats(path)

    print("\n=== RAW STRUCTURE STATS ===")
    for key, value in sorted(stats.items()):
        print(f"{key:30s}: {value}")

    print_suspicious_headings(path)

    print("\n=== AUTHOR TREE ===")
    raw_tree = get_tree(path, normalize=False)

    print(
        "Node count:",
        len(list(raw_tree.iter()))
    )

    print_tree(raw_tree)

    print("\n=== NORMALIZED TREE ===")
    normalized_tree = get_tree(path, normalize=True)

    print(
        "Node count:",
        len(list(normalized_tree.iter()))
    )

    print_tree(normalized_tree)

    raw_nodes = len(list(raw_tree.iter()))
    normalized_nodes = len(list(normalized_tree.iter()))

    print("\n=== NORMALIZATION EFFECT ===")
    print("Raw nodes       :", raw_nodes)
    print("Normalized nodes:", normalized_nodes)
    print("Difference      :", normalized_nodes - raw_nodes)


def compare_pred_ref(pred_path, ref_path):
    print("\n" + "=" * 80)
    print("PRED:", pred_path)
    print("REF :", ref_path)

    pred_raw = get_tree(pred_path, normalize=False)
    ref_raw = get_tree(ref_path, normalize=False)

    raw_score, raw_dist, pred_n, ref_n = STEDS(
        pred_raw,
        ref_raw
    )

    pred_norm = get_tree(pred_path, normalize=True)
    ref_norm = get_tree(ref_path, normalize=True)

    norm_score, norm_dist, pred_norm_n, ref_norm_n = STEDS(
        pred_norm,
        ref_norm
    )

    print("\n=== STEDS COMPARISON ===")
    print(
        f"Original STEDS   : {raw_score * 100:.4f}"
    )
    print(
        f"Original distance: {raw_dist}"
    )
    print(
        f"Original nodes   : pred={pred_n}, ref={ref_n}"
    )

    print()

    print(
        f"Normalized STEDS   : {norm_score * 100:.4f}"
    )
    print(
        f"Normalized distance: {norm_dist}"
    )
    print(
        f"Normalized nodes   : pred={pred_norm_n}, ref={ref_norm_n}"
    )

    print("\n=== PRED HEADING STATS ===")
    for k, v in sorted(heading_stats(pred_path).items()):
        print(f"{k:30s}: {v}")

    print("\n=== REF HEADING STATS ===")
    for k, v in sorted(heading_stats(ref_path).items()):
        print(f"{k:30s}: {v}")

def tree_signature(tree):
    out = []

    def walk(node, parent_label="NONE", depth=0):
        out.append((
            depth,
            parent_label,
            node.label[:40]
        ))

        for child in node.children:
            walk(child, node.label[:40], depth + 1)

    walk(tree)
    return out

def topology_signature(tree):
    sig = []

    def walk(node, depth=0):
        sig.append((depth, len(node.children)))

        for child in node.children:
            walk(child, depth + 1)

    walk(tree)
    return sig

def compare_dirs(pred_dir, ref_dir):
    rows = []

    for filename in sorted(os.listdir(ref_dir)):
        if not filename.endswith('.mmd'):
            continue

        pred_path = os.path.join(pred_dir, filename)
        ref_path = os.path.join(ref_dir, filename)

        if not os.path.exists(pred_path):
            print("[MISSING PRED]", filename)
            continue

        pred_raw = get_tree(pred_path, normalize=False)
        ref_raw = get_tree(ref_path, normalize=False)

        raw_score, _, pred_nodes, ref_nodes = STEDS(
            pred_raw,
            ref_raw
        )

        pred_norm = get_tree(pred_path, normalize=True)
        ref_norm = get_tree(ref_path, normalize=True)

        norm_score, _, pred_norm_nodes, ref_norm_nodes = STEDS(
            pred_norm,
            ref_norm
        )

        pstats = heading_stats(pred_path)
        rstats = heading_stats(ref_path)

        rows.append({
            'file': filename,
            'raw_steds': raw_score * 100,
            'norm_steds': norm_score * 100,
            'pred_nodes': pred_nodes,
            'ref_nodes': ref_nodes,
            'pred_norm_nodes': pred_norm_nodes,
            'ref_norm_nodes': ref_norm_nodes,
            'pred_ascii_headings':
                pstats['ascii_h1']
                + pstats['ascii_h2']
                + pstats['ascii_h3_plus'],
            'pred_fullwidth_headings':
                pstats['fullwidth_heading'],
            'ref_ascii_headings':
                rstats['ascii_h1']
                + rstats['ascii_h2']
                + rstats['ascii_h3_plus'],
            'ref_fullwidth_headings':
                rstats['fullwidth_heading'],
        })

    if not rows:
        print("No matching .mmd files.")
        return

    print("\n=== DATASET SUMMARY ===")

    raw_avg = sum(x['raw_steds'] for x in rows) / len(rows)
    norm_avg = sum(x['norm_steds'] for x in rows) / len(rows)

    print("Files:", len(rows))
    print(f"Raw STEDS avg       : {raw_avg:.4f}")
    print(f"Normalized STEDS avg: {norm_avg:.4f}")
    print(
        f"Difference          : {norm_avg - raw_avg:+.4f}"
    )

    print(
        "Pred ASCII headings:",
        sum(x['pred_ascii_headings'] for x in rows)
    )
    print(
        "Pred fullwidth headings:",
        sum(x['pred_fullwidth_headings'] for x in rows)
    )
    print(
        "Ref ASCII headings:",
        sum(x['ref_ascii_headings'] for x in rows)
    )
    print(
        "Ref fullwidth headings:",
        sum(x['ref_fullwidth_headings'] for x in rows)
    )

    print("\n=== FILES WITH BIGGEST NORMALIZATION EFFECT ===")

    rows.sort(
        key=lambda x: abs(x['norm_steds'] - x['raw_steds']),
        reverse=True
    )

    for row in rows[:20]:
        diff = row['norm_steds'] - row['raw_steds']

        print(
            f"{row['file']:40s} "
            f"raw={row['raw_steds']:7.2f} "
            f"norm={row['norm_steds']:7.2f} "
            f"diff={diff:+7.2f} "
            f"pred_nodes={row['pred_nodes']:4d} "
            f"ref_nodes={row['ref_nodes']:4d}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--file',
        type=str,
        help='Inspect a single .mmd file'
    )

    parser.add_argument(
        '--pred',
        type=str,
        help='Single prediction .mmd'
    )

    parser.add_argument(
        '--ref',
        type=str,
        help='Single reference .mmd'
    )

    parser.add_argument(
        '--pred_dir',
        type=str,
        help='Prediction directory'
    )

    parser.add_argument(
        '--ref_dir',
        type=str,
        help='Reference directory'
    )

    args = parser.parse_args()

    if args.file:
        inspect_file(args.file)

    elif args.pred and args.ref:
        compare_pred_ref(
            args.pred,
            args.ref
        )

    elif args.pred_dir and args.ref_dir:
        compare_dirs(
            args.pred_dir,
            args.ref_dir
        )

    else:
        parser.error(
            'Use --file FILE, '
            '--pred FILE --ref FILE, '
            'or --pred_dir DIR --ref_dir DIR'
        )

    paths = {
        "ckpt_0":"/home/pj24001974/ku60000761/project/test_project/M4Doc/DoTA_dataset/zh_mmd/2001.02870_04.mmd",
        "ckpt_1": "/home/pj24001974/ku60000761/project/test_project/M4Doc/results/predictions/dota_trans_jieba_lm_1536_v2/checkpoint-9000/2001.02870_04.mmd",
        "ckpt_2": "/home/pj24001974/ku60000761/project/test_project/M4Doc/results/predictions/dota_trans_jieba_lm_1536_v2/checkpoint-38000/2001.02870_04.mmd",
        "ckpt_3": "/home/pj24001974/ku60000761/project/test_project/M4Doc/results/predictions/dota_trans_jieba_lm_1536_v2/checkpoint-48000/2001.02870_04.mmd",
    }

    def topology_signature(tree):
        sig = []

        def walk(node, depth=0):
            sig.append((depth, len(node.children)))

            for child in node.children:
                walk(child, depth + 1)

        walk(tree)
        return tuple(sig)


    same_01 = 0
    same_02 = 0
    same_03 = 0
    all_same = 0
    total = 0

    ref_dir = (
        "/home/pj24001974/ku60000761/project/test_project/"
        "M4Doc/DoTA_dataset/zh_mmd"
    )

    ckpt1_dir = (
        "/home/pj24001974/ku60000761/project/test_project/"
        "M4Doc/results/predictions/dota_trans_jieba_lm_1536_v2/"
        "checkpoint-9000"
    )

    ckpt2_dir = (
        "/home/pj24001974/ku60000761/project/test_project/"
        "M4Doc/results/predictions/dota_trans_jieba_lm_1536_v2/"
        "checkpoint-38000"
    )

    ckpt3_dir = (
        "/home/pj24001974/ku60000761/project/test_project/"
        "M4Doc/results/predictions/dota_trans_jieba_lm_1536_v2/"
        "checkpoint-48000"
    )


    for filename in sorted(os.listdir(ref_dir)):
        if not filename.endswith(".mmd"):
            continue

        paths = [
            os.path.join(ref_dir, filename),
            os.path.join(ckpt1_dir, filename),
            os.path.join(ckpt2_dir, filename),
            os.path.join(ckpt3_dir, filename),
        ]

        if not all(os.path.exists(p) for p in paths):
            continue

        sigs = [
            topology_signature(get_tree(p))
            for p in paths
        ]

        total += 1

        if sigs[0] == sigs[1]:
            same_01 += 1

        if sigs[0] == sigs[2]:
            same_02 += 1

        if sigs[0] == sigs[3]:
            same_03 += 1

        if sigs[0] == sigs[1] == sigs[2] == sigs[3]:
            all_same += 1

    steds_1 = []
    steds_2 = []
    steds_3 = []

    for filename in sorted(os.listdir(ref_dir)):
        if not filename.endswith(".mmd"):
            continue

        paths = [
            os.path.join(ref_dir, filename),
            os.path.join(ckpt1_dir, filename),
            os.path.join(ckpt2_dir, filename),
            os.path.join(ckpt3_dir, filename),
        ]

        if not all(os.path.exists(p) for p in paths):
            continue

        trees = [get_tree(p) for p in paths]

        s1, _, _, _ = STEDS(trees[1], trees[0])
        s2, _, _, _ = STEDS(trees[2], trees[0])
        s3, _, _, _ = STEDS(trees[3], trees[0])

        steds_1.append(s1)
        steds_2.append(s2)
        steds_3.append(s3)

    print("ckpt9000 STEDS :", sum(steds_1) / len(steds_1) * 100)
    print("ckpt38000 STEDS:", sum(steds_2) / len(steds_2) * 100)
    print("ckpt48000 STEDS:", sum(steds_3) / len(steds_3) * 100)

    ref_tree = get_tree(
        os.path.join(ref_dir, "2001.02870_04.mmd")
    )

    score, dist, pn, rn = STEDS(ref_tree, ref_tree)

    print("REF vs REF")
    print("STEDS:", score * 100)
    print("distance:", dist)
    print("nodes:", pn, rn)


    print("Total:", total)

    print(
        "ref == ckpt9000:",
        same_01,
        same_01 / total
    )

    print(
        "ref == ckpt38000:",
        same_02,
        same_02 / total
    )

    print(
        "ref == ckpt48000:",
        same_03,
        same_03 / total
    )

    print(
        "all same:",
        all_same,
        all_same / total
    )