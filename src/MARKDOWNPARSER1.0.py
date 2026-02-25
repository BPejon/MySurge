import re
from collections import defaultdict

class MarkdownNode:
    def __init__(self, title, level, parent=None):
        self.title = title
        self.level = level
        self.parent = parent
        self.children = []
        self.content = []

    def add_child(self, child):
        self.children.append(child)

    def add_content(self, text):
        self.content.append(text)

    def to_dict(self):
        return {
            "title": self.title,
            "level": self.level,
            "parent": self.parent.title if self.parent else "root",
            "children": [child.to_dict() for child in self.children],
            "content": "\n".join(self.content),
        }

def parse_markdown(file_path):
    """Parse the markdown file and return its hierarchical structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    root = MarkdownNode(title="root", level=0)
    current_node = root
    node_stack = [root]

    header_pattern = re.compile(r"^(#+)\s+(.*)")

    for line in lines:
        line = line.rstrip()

        match = header_pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2)

            new_node = MarkdownNode(title=title, level=level)

            while node_stack and node_stack[-1].level >= level:
                node_stack.pop()

            parent_node = node_stack[-1]
            parent_node.add_child(new_node)
            new_node.parent = parent_node

            node_stack.append(new_node)
            current_node = new_node
        else:
            current_node.add_content(line)

    return root

def parse_refs(file_path):
    """Parse the references from the markdown file.
    
    There are two possible formats for the references section:
    (1) A list of references with the format:
        [num] Author(s). Title. (The title may be embraced by '*')
    (2) A list of references with the format:
        [num] Title
    
    The function returns a dict of titles with their num id as keys.
    """
    references:dict = {}
    pattern = re.compile(r'^\[(\d+)\]\s*(?:.*?[“"](.*?)[”"]|.*?([\w\s:,.-]+))')
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line.startswith("["):
                continue

            match = pattern.match(line)
            if match:
                # Verificar qual grupo capturou o título
                if match.group(2):  # Título entre aspas
                    title = match.group(2).strip()
                else:  # Título sem aspas
                    title = match.group(3).strip() if match.group(3) else ""
                ref_id = int(match.group(1))

                if title.endswith(('.', ',', ';')):
                    title = title[:-1].strip()
                references[ref_id] = title.strip()

                if title:
                    references[ref_id] = title

                print(f"Extracted reference - ID: {ref_id}, Title: {title.strip()}")
                print("Current references dict:", references)
    
    return references

