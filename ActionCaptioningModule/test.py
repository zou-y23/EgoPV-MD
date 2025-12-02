import json
import argparse
import random
from collections import defaultdict
from difflib import SequenceMatcher


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


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings"""
    matcher = SequenceMatcher(None, s1.lower(), s2.lower())
    return matcher.ratio()


def calculate_normalized_edit_distance(s1: str, s2: str) -> float:
    """Calculate normalized edit distance (ED / max(len(s1), len(s2)))"""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    ed = calculate_edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return ed / max_len if max_len > 0 else 0.0


def create_mapping_table(mapping_table_file: str, output_file: str):
    """
    Load mapping table and calculate ED between mapped and reference info
    """
    # Load mapping table
    with open(mapping_table_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    mapping_table = mapping_data.get('mapping_table', [])
    
    total_verb_improved = 0
    total_noun_improved = 0
    total_action_improved = 0
    
    verb_ed_sum = 0.0
    noun_ed_sum = 0.0
    action_ed_sum = 0.0
    
    verb_normalized_ed_sum = 0.0
    noun_normalized_ed_sum = 0.0
    action_normalized_ed_sum = 0.0
    
    for entry in mapping_table:
        # Get mapped and reference values directly from mapping table
        mapped_verb = entry.get('generated_verb', '')
        mapped_noun = entry.get('generated_noun', '')
        mapped_action = entry.get('generated_action', '')
        
        reference_verb = entry.get('reference_verb', '')
        reference_noun = entry.get('reference_noun', '')
        reference_action = entry.get('reference_action', '')
        
        # Directly calculate normalized edit distance between mapped and reference
        mapped_verb_normalized_ed = calculate_normalized_edit_distance(
            mapped_verb.lower(), reference_verb.lower()
        ) if mapped_verb and reference_verb else 1.0
        mapped_verb_ed = calculate_edit_distance(
            mapped_verb.lower(), reference_verb.lower()
        ) if mapped_verb and reference_verb else len(reference_verb) if reference_verb else 0
        
        mapped_noun_normalized_ed = calculate_normalized_edit_distance(
            mapped_noun.lower(), reference_noun.lower()
        ) if mapped_noun and reference_noun else 1.0
        mapped_noun_ed = calculate_edit_distance(
            mapped_noun.lower(), reference_noun.lower()
        ) if mapped_noun and reference_noun else len(reference_noun) if reference_noun else 0
        
        mapped_action_normalized_ed = calculate_normalized_edit_distance(
            mapped_action.lower(), reference_action.lower()
        ) if mapped_action and reference_action else 1.0
        mapped_action_ed = calculate_edit_distance(
            mapped_action.lower(), reference_action.lower()
        ) if mapped_action and reference_action else len(reference_action) if reference_action else 0
        
        # Get target EDs from mapping table
        target_verb_ed = entry.get('verb_mapping', {}).get('target_normalized_ed', 0.5)
        target_noun_ed = entry.get('noun_mapping', {}).get('target_normalized_ed', 0.55)
        target_action_ed = entry.get('action_mapping', {}).get('target_normalized_ed', 0.58)
        
        verb_improved = mapped_verb_normalized_ed <= target_verb_ed
        noun_improved = mapped_noun_normalized_ed <= target_noun_ed
        action_improved = mapped_action_normalized_ed <= target_action_ed
        
        if verb_improved:
            total_verb_improved += 1
        if noun_improved:
            total_noun_improved += 1
        if action_improved:
            total_action_improved += 1
        
        verb_ed_sum += mapped_verb_ed
        noun_ed_sum += mapped_noun_ed
        action_ed_sum += mapped_action_ed
        
        verb_normalized_ed_sum += mapped_verb_normalized_ed
        noun_normalized_ed_sum += mapped_noun_normalized_ed
        action_normalized_ed_sum += mapped_action_normalized_ed
    
    # Calculate statistics
    total_samples = len(mapping_table)
    verb_improvement_rate = total_verb_improved / total_samples * 100 if total_samples > 0 else 0
    noun_improvement_rate = total_noun_improved / total_samples * 100 if total_samples > 0 else 0
    action_improvement_rate = total_action_improved / total_samples * 100 if total_samples > 0 else 0
    
    avg_verb_ed = verb_ed_sum / total_samples if total_samples > 0 else 0
    avg_noun_ed = noun_ed_sum / total_samples if total_samples > 0 else 0
    avg_action_ed = action_ed_sum / total_samples if total_samples > 0 else 0
    
    avg_verb_normalized_ed = verb_normalized_ed_sum / total_samples if total_samples > 0 else 0
    avg_noun_normalized_ed = noun_normalized_ed_sum / total_samples if total_samples > 0 else 0
    avg_action_normalized_ed = action_normalized_ed_sum / total_samples if total_samples > 0 else 0
    
    # Print results
    print(f"  Verb Normalized ED:   {avg_verb_normalized_ed:.3f}")
    print(f"  Noun Normalized ED:   {avg_noun_normalized_ed:.3f}")
    print(f"  Action Normalized ED: {avg_action_normalized_ed:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load mapping table and calculate ED between mapped and reference")
    parser.add_argument("--mapping_table", default="/root/autodl-tmp/EILEV-main/generated_and_reference.json",
                        help="Input mapping table JSON file")
    parser.add_argument("--output", default="/root/autodl-tmp/EILEV-main/test_results.json",
                        help="Output file (optional)")
    
    args = parser.parse_args()
    
    create_mapping_table(
        args.mapping_table,
        args.output
    )

