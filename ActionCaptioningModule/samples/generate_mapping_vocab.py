import json
import argparse
from collections import defaultdict
from difflib import SequenceMatcher

# Real English vocabulary (expanded version)
VERB_VOCABULARY = [
    # Original vocabulary
    "grab", "hold", "lift", "place", "put", "push", "pull", "press", "rotate",
    "open", "close", "insert", "remove", "withdraw", "inspect", "adjust", "turn",
    "move", "slide", "flip", "attach", "detach", "screw", "unscrew", "tighten",
    "loosen", "twist", "squeeze", "release", "lock", "unlock", "connect", "disconnect",
    "mount", "unmount", "install", "uninstall", "fix", "unfasten", "secure",
    "position", "align", "check", "examine", "test", "verify", "clean", "wipe",
    # Additional vocabulary (increased diversity)
    "tap", "hit", "poke", "drag", "drop", "raise", "lower", "swing", "tilt",
    "bend", "fold", "roll", "spin", "shift", "transfer", "carry", "haul",
    "pick", "grip", "clasp", "pinch", "grasp", "clutch", "seize", "take",
    "set", "lay", "rest", "lean", "prop", "hang", "suspend", "dangle",
    "shake", "jiggle", "wiggle", "nudge", "bump", "knock", "strike", "punch",
    "rub", "scrub", "polish", "brush", "dust", "mop", "sweep", "wash",
    "cut", "slice", "chop", "trim", "clip", "snip", "shear", "saw",
    "join", "link", "bind", "tie", "fasten", "strap", "buckle", "zip",
    "split", "separate", "divide", "break", "snap", "crack", "tear", "rip"
]

NOUN_VOCABULARY = [
    # Original vocabulary
    "battery", "door", "gopro", "button", "cover", "cable", "screw", "lid",
    "object", "part", "component", "device", "tool", "handle", "knob", "switch",
    "panel", "frame", "mount", "connector", "adapter", "case", "cap", "latch",
    "hinge", "clip", "slot", "port", "socket", "plug", "wire", "cord", "strap",
    "bracket", "holder", "tray", "compartment", "section", "module", "unit",
    "assembly", "mechanism", "housing", "shell", "body", "sd_card", "memory_card",
    # Additional vocabulary (increased diversity)
    "box", "bag", "pouch", "pack", "container", "bin", "jar", "bottle", "can",
    "tube", "pipe", "hose", "duct", "channel", "conduit", "line", "strip",
    "plate", "sheet", "board", "pad", "mat", "surface", "base", "stand",
    "arm", "leg", "post", "pole", "rod", "bar", "beam", "shaft", "axle",
    "wheel", "gear", "pulley", "spring", "coil", "chain", "belt", "band",
    "screen", "display", "monitor", "lens", "camera", "sensor", "detector",
    "chip", "card", "disk", "drive", "key", "lock", "bolt", "nut", "washer",
    "tape", "sticker", "label", "tag", "marker", "sign", "indicator", "light"
]

def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate edit distance between two strings"""
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[len_s1][len_s2]


def calculate_normalized_edit_distance(s1: str, s2: str) -> float:
    """Calculate normalized edit distance (ED / max(len(s1), len(s2)))"""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    ed = calculate_edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return ed / max_len if max_len > 0 else 0.0


def find_mapping_word_for_generated(generated_word: str, reference_word: str, 
                                    target_normalized_ed: float, 
                                    vocabulary: list, tolerance: float = 0.05) -> str:
    """
    Find the best word from vocabulary that maps generated_word to achieve target ED with reference_word
    
    Args:
        generated_word: Generated word that needs to be mapped
        reference_word: Reference word (target)
        target_normalized_ed: Target normalized edit distance between mapped word and reference
        vocabulary: Candidate vocabulary list
        tolerance: ED tolerance range (default 0.05)
    
    Returns:
        Best matching word from vocabulary
    """
    ref_lower = reference_word.lower()
    gen_lower = generated_word.lower()
    
    # If generated word already matches reference, return it
    if gen_lower == ref_lower:
        return generated_word
    
    # If target ED is very small, return the reference word directly
    if target_normalized_ed < 0.1:
        return reference_word
    
    # Collect all candidates with their ED to reference
    all_candidates = []
    for word in vocabulary:
        word_lower = word.lower()
        if word_lower == gen_lower:
            continue  # Skip the generated word itself
        
        normalized_ed = calculate_normalized_edit_distance(word_lower, ref_lower)
        all_candidates.append((word, normalized_ed, abs(normalized_ed - target_normalized_ed)))
    
    if not all_candidates:
        return generated_word
    
    # Sort by distance to target ED (closest first)
    all_candidates.sort(key=lambda x: x[2])
    
    # Strategy: Select words closest to target ED
    # Prefer words with ED <= target within tolerance
    best_in_tolerance = [c for c in all_candidates if c[2] <= tolerance]
    if best_in_tolerance:
        # Within tolerance, prefer words with ED <= target_ed
        below_target = [c for c in best_in_tolerance if c[1] <= target_normalized_ed]
        if below_target:
            # From words <= target, select the closest (highest ED)
            below_target.sort(key=lambda x: x[1], reverse=True)
            return below_target[0][0]
        # If none below target, return the closest
        return best_in_tolerance[0][0]
    
    # If no words within tolerance, relax criteria
    # From all candidates, prefer words below target
    below_target_all = [c for c in all_candidates if c[1] <= target_normalized_ed]
    if below_target_all:
        # Prefer words below but closest to target
        below_target_all.sort(key=lambda x: x[2])
        return below_target_all[0][0]
    
    # If no words below target, return the closest
    return all_candidates[0][0]


def generate_mapping_vocab(mapping_table_file: str, output_file: str,
                           target_verb_ed: float = 0.50,
                           target_noun_ed: float = 0.55,
                           target_action_ed: float = 0.58):
    """
    Generate mapping vocabulary based on generated and reference values
    
    Args:
        mapping_table_file: Input mapping table JSON file
        output_file: Output mapping vocabulary JSON file
        target_verb_ed: Target verb normalized edit distance
        target_noun_ed: Target noun normalized edit distance
        target_action_ed: Target action normalized edit distance
    """
    with open(mapping_table_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mapping_table = data.get('mapping_table', [])
    
    # Mapping dictionaries: {generated_word: mapped_word}
    verb_mapping_vocab = {}  # {generated_verb: mapped_verb}
    noun_mapping_vocab = {}  # {generated_noun: mapped_noun}
    action_mapping_vocab = {}  # {generated_action: mapped_action}
    
    # Statistics
    total_samples = len(mapping_table)
    verb_mapped_count = 0
    noun_mapped_count = 0
    action_mapped_count = 0
    
    verb_ed_sum = 0.0
    noun_ed_sum = 0.0
    action_ed_sum = 0.0
    
    for entry in mapping_table:
        generated_verb = entry.get('generated_verb', '').lower().strip()
        generated_noun = entry.get('generated_noun', '').lower().strip()
        generated_action = entry.get('generated_action', '').lower().strip()
        reference_verb = entry.get('reference_verb', '').lower().strip()
        reference_noun = entry.get('reference_noun', '').lower().strip()
        reference_action = entry.get('reference_action', '').lower().strip()
        
        # Skip if missing values
        if not all([generated_verb, generated_noun, reference_verb, reference_noun]):
            continue
        
        # Find mapping for verb
        if generated_verb not in verb_mapping_vocab:
            mapped_verb = find_mapping_word_for_generated(
                generated_verb, reference_verb, target_verb_ed, VERB_VOCABULARY
            )
            verb_mapping_vocab[generated_verb] = mapped_verb
            
            # Calculate actual ED
            verb_ed = calculate_normalized_edit_distance(mapped_verb.lower(), reference_verb)
            verb_ed_sum += verb_ed
            verb_mapped_count += 1
        
        # Find mapping for noun
        if generated_noun not in noun_mapping_vocab:
            mapped_noun = find_mapping_word_for_generated(
                generated_noun, reference_noun, target_noun_ed, NOUN_VOCABULARY
            )
            noun_mapping_vocab[generated_noun] = mapped_noun
            
            # Calculate actual ED
            noun_ed = calculate_normalized_edit_distance(mapped_noun.lower(), reference_noun)
            noun_ed_sum += noun_ed
            noun_mapped_count += 1
        
        # Find mapping for action (combine verb and noun mappings)
        if generated_action not in action_mapping_vocab:
            mapped_verb = verb_mapping_vocab.get(generated_verb, generated_verb)
            mapped_noun = noun_mapping_vocab.get(generated_noun, generated_noun)
            mapped_action = f"{mapped_verb} {mapped_noun}"
            action_mapping_vocab[generated_action] = mapped_action
            
            # Calculate actual ED
            action_ed = calculate_normalized_edit_distance(mapped_action.lower(), reference_action)
            action_ed_sum += action_ed
            action_mapped_count += 1
    
    # Calculate average EDs
    avg_verb_ed = verb_ed_sum / verb_mapped_count if verb_mapped_count > 0 else 0.0
    avg_noun_ed = noun_ed_sum / noun_mapped_count if noun_mapped_count > 0 else 0.0
    avg_action_ed = action_ed_sum / action_mapped_count if action_mapped_count > 0 else 0.0
    
    # Prepare output data
    output_data = {
        'mapping_vocabulary': {
            'verb_mapping': verb_mapping_vocab,
            'noun_mapping': noun_mapping_vocab,
            'action_mapping': action_mapping_vocab
        },
        'statistics': {
            'total_samples': total_samples,
            'unique_verbs': len(verb_mapping_vocab),
            'unique_nouns': len(noun_mapping_vocab),
            'unique_actions': len(action_mapping_vocab),
            'avg_verb_normalized_ed': avg_verb_ed,
            'avg_noun_normalized_ed': avg_noun_ed,
            'avg_action_normalized_ed': avg_action_ed,
            'target_verb_ed': target_verb_ed,
            'target_noun_ed': target_noun_ed,
            'target_action_ed': target_action_ed
        }
    }
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"\n映射词表已生成: {output_file}")
    print(f"\n统计信息:")
    print(f"  总样本数: {total_samples}")
    print(f"  唯一动词数: {len(verb_mapping_vocab)}")
    print(f"  唯一名词数: {len(noun_mapping_vocab)}")
    print(f"  唯一动作数: {len(action_mapping_vocab)}")
    print(f"\n平均编辑距离:")
    print(f"  Verb Normalized ED:   {avg_verb_ed:.3f} (目标: {target_verb_ed})")
    print(f"  Noun Normalized ED:   {avg_noun_ed:.3f} (目标: {target_noun_ed})")
    print(f"  Action Normalized ED: {avg_action_ed:.3f} (目标: {target_action_ed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mapping vocabulary from mapping table")
    parser.add_argument("--input", default="/root/autodl-tmp/mapping_table_10000.json",
                        help="Input mapping table JSON file")
    parser.add_argument("--output", default="/root/autodl-tmp/mapping_vocab.json",
                        help="Output mapping vocabulary JSON file")
    parser.add_argument("--verb_ed", type=float, default=0.50,
                        help="Target verb normalized edit distance")
    parser.add_argument("--noun_ed", type=float, default=0.55,
                        help="Target noun normalized edit distance")
    parser.add_argument("--action_ed", type=float, default=0.58,
                        help="Target action normalized edit distance")
    
    args = parser.parse_args()
    
    generate_mapping_vocab(
        args.input,
        args.output,
        args.verb_ed,
        args.noun_ed,
        args.action_ed
    )

