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
    
    Simple approach: split by "." and pick the longest segment.
    """
    references = {}
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line.startswith("["):
                continue
            
            # Extrai o número da referência
            id_match = re.match(r'^\[(\d+)\]', line)
            if not id_match:
                continue
            
            ref_id = int(id_match.group(1))
            content = line[id_match.end():].strip()
            
            # Divide a string por pontos
            segments = content.split('.')
            
            # Remove espaços em branco de cada segmento
            segments = [seg.strip() for seg in segments if seg.strip()]
            
            # Ignora segmentos que são claramente autores ou informações de publicação
            candidate_segments = []
            
            for seg in segments:
                # Ignora segmentos muito curtos (provavelmente iniciais)
                if len(seg) <= 3:
                    continue
                
                # Ignora segmentos que são só números ou datas
                if re.match(r'^\d+$', seg) or re.match(r'^\d{4}$', seg):
                    continue
                
                # Ignora segmentos com "et al"
                if 'et al' in seg.lower():
                    continue
                
                # Ignora segmentos com padrões de autoria (inicial + sobrenome)
                if re.match(r'^[A-Z]\.?\s+[A-Z][a-z]+$', seg):
                    continue
                
                # Ignora segmentos com informações de publicação
                pub_patterns = [
                    r'technical report',
                    r'pp\.',
                    r'volume',
                    r'no\.',
                    r'cornell',
                    r'ieee',
                    r'\d+\(\d+\):\d+--\d+',
                    r'neurips',
                    r'laboratory',
                ]
                
                is_publication = False
                for pattern in pub_patterns:
                    if re.search(pattern, seg.lower()):
                        is_publication = True
                        break
                
                if not is_publication:
                    candidate_segments.append(seg)
            
            # Se temos candidatos, escolhe o mais longo
    
            # Ordena por comprimento (do mais longo para o mais curto)
            candidate_segments.sort(key=len, reverse=True)
            title = candidate_segments[0]
                
            # Limpeza básica
            title = title.rstrip(',;:')
            title = re.sub(r'\s+', ' ', title).strip()
                
            references[ref_id] = title
            #print(f"Extracted reference - ID: {ref_id}, Title: {title}")
    
    return references

