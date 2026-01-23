import re
from nltk.corpus import cmudict
from collections import defaultdict
import json
import nltk
from nltk.corpus import brown
import random

nltk.download('cmudict')
nltk.download('brown')
nltk.download('punkt_tab')

sentences_old = [
    "tell us about nixon",
    "give me the headlines",
    "tell me about china",
    "give us the stories",
    "tell me about the stock market",
    "tell us about recent events",
    "tell me about russia",
    "show me the weather",
    "show us the news stories",
    "give me the current events",
    "show us the top stories",
    "give us today's stories",
    "give us news articles",
    "tell me all about china",
    "tell us about china",
    "tell us all about china",
    "tell me about nixon",
    "tell me all about nixon",
    "tell us all about nixon",
    "give me the news"
]

PHONEME_CLASSES = {
    "vowel": {
        "AA","AE","AH","AO","AW","AY",
        "EH","ER","EY",
        "IH","IY",
        "OW","OY",
        "UH","UW"
    },
    "stop": {"P","B","T","D","K","G"},
    "fricative": {"F","V","TH","DH","S","Z","SH","ZH","HH"},
    "nasal": {"M","N","NG"},
    "liquid": {"L","R"},
    "glide": {"W","Y"},
    "silence": {"<START>", "<END>"}
}

PHONEME_COLORS = {
    "vowel": "lightskyblue",
    "stop": "lightcoral",
    "fricative": "khaki",
    "nasal": "lightgreen",
    "liquid": "plum",
    "glide": "orange",
    "silence": "gray90",
    "unknown": "white"
}

START = "<START>"
END = "<END>"

def strip_stress(phone):
    return re.sub(r"\d", "", phone)

def word_to_phones(word, cmu_dict):
    word = word.lower()
    phones = cmu_dict[word][0]
    return [strip_stress(p) for p in phones]

def sentence_to_phonemes(sentence, cmu_dict, raise_missing_error=False):
    phones = [START]
    for word in sentence.split():
        if word in cmu_dict:
            phones.extend(word_to_phones(word, cmu_dict))
    phones.append(END)
    return phones

class Node:
    __slots__ = ("id", "phoneme", "edges")

    def __init__(self, node_id, phoneme):
        self.id = node_id
        self.phoneme = phoneme
        self.edges = {}


def build_phoneme_dag(phoneme_sentences):
    nodes = []
    node_lookup = {}

    def get_node(prefix):
        if prefix not in node_lookup:
            phoneme = prefix[-1]
            node = Node(len(nodes), phoneme)
            nodes.append(node)
            node_lookup[prefix] = node
        return node_lookup[prefix]

    root = get_node(("<START>",))

    for phones in phoneme_sentences:
        current_prefix = ("<START>",)
        current_node = root

        for phone in phones[1:]:
            next_prefix = current_prefix + (phone,)
            next_node = get_node(next_prefix)

            if phone not in current_node.edges:
                current_node.edges[phone] = next_node

            current_node = next_node
            current_prefix = next_prefix

    return root, nodes


def export_to_dot_nodes_labeled(nodes, filename="phoneme_dag.dot"):
    with open(filename, "w") as f:
        f.write("digraph PhonemeDAG {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [shape=circle, width=0.25, height=0.25, fontsize=8];\n")

        for node in nodes:
            label = node.phoneme
            f.write(f'  {node.id} [label="{label}"];\n')

        for node in nodes:
            for child in node.edges.values():
                f.write(f'  {node.id} -> {child.id};\n')

        f.write("}\n")


def phoneme_class(phone):
    for cls, phones in PHONEME_CLASSES.items():
        if phone in phones:
            return cls
    return "unknown"


def export_colored_dag_to_dot(nodes, filename="phoneme_dag.dot"):
    with open(filename, "w") as f:
        f.write("digraph PhonemeDAG {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [style=filled, fontname=Helvetica, fontsize=8];\n")
        f.write("  edge [color=gray50];\n\n")

        for node in nodes:
            phone = node.phoneme
            cls = phoneme_class(phone)
            color = PHONEME_COLORS.get(cls, "white")

            if phone in ("<s>", "</s>"):
                shape = "box"
            else:
                shape = "circle"

            f.write(
                f'  {node.id} '
                f'[label="{phone}", shape={shape}, fillcolor="{color}"];\n'
            )

        f.write("\n")

        for node in nodes:
            for child in node.edges.values():
                f.write(f"  {node.id} -> {child.id};\n")

        f.write("}\n")


def export_dag_to_json(nodes, filename="phoneme_dag.json"):
    incoming_connections = defaultdict(list)
    for node in nodes:
        for child in node.edges.values():
            incoming_connections[child.id].append(node.id)

    nodes_data = []
    for node in nodes:
        nodes_data.append({
            "id": node.id,
            "phoneme": node.phoneme,
            "connects_from": incoming_connections[node.id]
        })

    with open(filename, "w") as f:
        json.dump(nodes_data, f, indent=2)

    print(f"Exported DAG to {filename}")


def main():
    cmu = cmudict.dict()

    sentences_brown = brown.sents()

    print(len(sentences_brown))

    full_sentences_brown = [" ".join(s) for s in sentences_brown]

    n = 200
    full_sentences_brown_sample = random.sample(full_sentences_brown, n)

    full_sentences_brown_sample_n = []
    n_words = 400

    for sentence in full_sentences_brown_sample:
        if len(sentence) > n_words:
            full_sentences_brown_sample_n.append(sentence[0:n_words])
        else:
            full_sentences_brown_sample_n.append(sentence)


    phoneme_sentences_brown_sample_n = [
        sentence_to_phonemes(s, cmu_dict=cmu)
        for s in full_sentences_brown_sample_n
    ]

    root_brown, nodes_brown = build_phoneme_dag(phoneme_sentences_brown_sample_n)

    print(f"Total nodes: {len(nodes_brown)}")

    export_dag_to_json(nodes_brown, filename="phoneme_dag.json")


if __name__ == '__main__':
    main()