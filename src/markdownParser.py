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
    """Parse the references from the markdown file, handling multi-line references."""
    references = {}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    current_ref_lines = []
    current_ref_id = None

    for line in lines:
        line = line.rstrip('\n')
        id_match = re.match(r'^\[(\d+)\]', line.strip())

        if id_match:
            # Processa referência anterior
            if current_ref_lines and current_ref_id is not None:
                full_ref = ' '.join(current_ref_lines)
                title = extract_title_from_ref(full_ref)
                references[current_ref_id] = title

            # Inicia nova referência
            current_ref_id = int(id_match.group(1))
            current_ref_lines = [line]
        else:
            if current_ref_id is not None:
                # Acumula linhas subsequentes (inclusive vazias, para não perder informação)
                current_ref_lines.append(line)

    # Processa última referência
    if current_ref_lines and current_ref_id is not None:
        full_ref = ' '.join(current_ref_lines)
        title = extract_title_from_ref(full_ref)
        references[current_ref_id] = title

    return references


def extract_title_from_ref(ref_text):
    """Extrai o título de uma string de referência completa usando múltiplas estratégias."""
    # Remove o ID inicial [n] e espaços
    content = re.sub(r'^\[\d+\]\s*', '', ref_text).strip()

    # Normaliza aspas curvas para retas (facilita a detecção)
    content = content.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # --- Estratégia 1: título entre aspas (duplas ou simples) ---
    quote_match = re.search(r'"([^"]+)"', content)  # aspas duplas
    if not quote_match:
        quote_match = re.search(r"'([^']+)'", content)  # aspas simples
    if quote_match:
        candidate = quote_match.group(1).strip()
        # Evita falsos positivos (ex.: aspas vazias ou muito curtas)
        if len(candidate) > 3 and not re.match(r'^\d+$', candidate):
            return clean_title(candidate)

    # --- Estratégia 2: título em itálico (marcado com * ou _ no Markdown) ---
    italic_match = re.search(r'\*([^*]+)\*', content)
    if not italic_match:
        italic_match = re.search(r'_([^_]+)_', content)
    if italic_match:
        candidate = italic_match.group(1).strip()
        if len(candidate) > 3 and not re.match(r'^\d+$', candidate):
            return clean_title(candidate)

    # --- Estratégia 3: divisão por pontos e identificação do título baseada em padrões ---
    segments = content.split('.')
    segments = [seg.strip() for seg in segments if seg.strip()]

    if len(segments) >= 2:
        # Estratégia: o segundo segmento costuma ser o título
        # Estrutura típica: [Autor (Ano)]. [TÍTULO]. [Publicação/DOI]
        
        # Se encontramos segmento [1] e não é muito curto, use-o
        if len(segments) >= 2:
            candidate = segments[1].strip()
            
            # Verificar se é realmente um título (tamanho > 15 caracteres, não é autor/pub)
            if (len(candidate) > 15 and 
                not is_author_segment(candidate) and 
                not is_publication_segment(candidate)):
                return clean_title(candidate)
        
        # Fallback: procura por padrão de publicação e pega só o antes
        pub_start = None
        for i in range(1, len(segments)):
            if is_publication_segment(segments[i]):
                pub_start = i
                break
        
        if pub_start is not None and pub_start >= 2:
            # Título = segmento 1 até antes da publicação
            title_segments = segments[1:pub_start]
            candidate = '. '.join(title_segments).strip()
            if len(candidate) > 15:
                return clean_title(candidate)

    # --- Estratégia 4: fallback - escolher o segmento mais longo que não seja autor/publicação ---
    candidate_segments = []
    for seg in segments:
        if len(seg) <= 3:
            continue
        if is_author_segment(seg) or is_publication_segment(seg):
            continue
        candidate_segments.append(seg)

    if candidate_segments:
        candidate_segments.sort(key=len, reverse=True)
        return clean_title(candidate_segments[0])

    # --- Último recurso: pegar o segmento mais longo (ou primeiro) e torcer ---
    if segments:
        segments.sort(key=len, reverse=True)
        return clean_title(segments[0])

    return ''


def is_author_segment(seg):
    """Retorna True se o segmento parece ser parte de uma lista de autores."""
    seg_lower = seg.lower()
    # Padrões comuns em autores
    if re.search(r'[A-Z]\.\s+[A-Z][a-z]+', seg):  # Inicial + Sobrenome (ex: J. Smith)
        return True
    if re.search(r'[A-Z][a-z]+,\s+[A-Z]\.', seg):  # Sobrenome, Inicial (ex: Smith, J.)
        return True
    if re.search(r'\band\b', seg_lower) and re.search(r'[A-Z]\.', seg):  # "and" com iniciais
        return True
    if 'et al' in seg_lower:
        return True
    # Se contém muitas palavras comuns, provavelmente não é autor
    common_words = ['the', 'of', 'in', 'on', 'at', 'for', 'with', 'from', 'to', 'and', 'a', 'an']
    words = seg_lower.split()
    if words and sum(1 for w in words if w in common_words) > len(words) // 2:
        return False
    return False


def is_publication_segment(seg):
    """Retorna True se o segmento parece ser informação de publicação."""
    seg_lower = seg.lower()
    # Padrões típicos de publicação
    pub_patterns = [
        r'\(\d{4}\)',                    # ano entre parênteses
        r'\b\d{4}\b',                    # ano solto (4 dígitos)
        r'vol\.?\s*\d+',                  # volume
        r'no\.?\s*\d+',                   # número
        r'pp\.?\s*\d+',                    # páginas
        r'pages?\s*\d+',
        r'\d+\(\d+\):\d+--\d+',           # formato comum: 12(3):45--67
        r'doi:\s*10\.\d{4,}',              # DOI
        r'https?://',
        r'\btechnical\s+report\b',        # technical report (word boundary)
        r'\bconference\b',                # conference (word boundary)
        r'\bproceedings\b',              # proceedings (word boundary)
        r'\bjournal\b',                  # journal (word boundary)
        r'\btransactions\b',             # transactions (word boundary)
        r'\bmagazine\b',                 # magazine (word boundary)
        r'\bpress\b(?!\s*ure)',          # press (word boundary, but not "pressure")
        r'\buniversity\b',               # university (word boundary)
        r'\blaboratory\b',               # laboratory (word boundary)
        r'in:\s',                        # "In:" geralmente precede o nome do evento/livro
    ]
    for pattern in pub_patterns:
        if re.search(pattern, seg_lower):
            return True
    return False


def clean_title(title):
    """Limpeza básica do título."""
    title = title.rstrip(',;:.')
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'^In:\s*', '', title, flags=re.IGNORECASE)
    # Remove aspas remanescentes no início/fim
    title = re.sub(r'^[\'"]+|[\'"]+$', '', title)
    return title