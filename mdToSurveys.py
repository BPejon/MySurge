'''
Características do Algoritmo:
    Identificação de Títulos: Reconhece títulos Markdown nos níveis # (section), ## (subsection) e ### (subsubsection).
    Hierarquia Automática: Constrói a árvore hierárquica baseada na ordem e nível dos títulos.
    UUIDs Únicos: Gera IDs únicos para cada nó usando a biblioteca uuid.
    Raiz Incluída: O nó raiz (level: "root") é criado automaticamente e todas as seções de nível section são adicionadas como suas subseções.
    Extração de Metadados: Inclui extração básica de título, autores, abstract e data de publicação.
    Formato JSON: Gera o arquivo no formato JSON especificado, com todos os campos necessários.

Pra executar python3 mdToSurveys.py artigo.md saida.json
 
'''

import json
import re
import uuid
from typing import List, Dict, Any, Optional

def generate_uuid() -> str:
    """Gera um UUID único para cada nó da estrutura."""
    return str(uuid.uuid4())


def extract_title_from_markdown(markdown_text: str) -> str:
    """
    Extrai o título principal do artigo (primeiro título # encontrado).
    """
    lines = markdown_text.split('\n')
    for line in lines:
        if line.startswith('# '):
            return line.replace('# ', '').strip()
    return "Untitled"


def extract_metadata(markdown_text: str) -> Dict[str, Any]:
    """
    Extrai metadados básicos do artigo.
    """
    # Tenta encontrar a data (formato: "Published online: 14 March 2022")
    date_match = re.search(r'Published online:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})', markdown_text)
    date = date_match.group(1) if date_match else ""
    
    # Converte para formato ISO (YYYY-MM-DD)
    if date:
        try:
            # Converte "14 March 2022" para "2022-03-14"
            from datetime import datetime
            date_obj = datetime.strptime(date, "%d %B %Y")
            date_iso = date_obj.strftime("%Y-%m-%dT00:00:00Z")
        except:
            date_iso = date
    else:
        date_iso = ""
    
    # Extrai ano
    year_match = re.search(r'Received:.*?(\d{4})', markdown_text)
    year = year_match.group(1) if year_match else ""
    
    # Extrai palavras-chave do abstract (categoria)
    abstract = extract_abstract(markdown_text)
    category = ""
    if abstract:
        # Tenta encontrar a linha de keywords
        keywords_match = re.search(r'Keywords?\s*(.*?)(?=\n|$)', markdown_text, re.IGNORECASE)
        if keywords_match:
            category = keywords_match.group(1).strip()
        else:
            # Pega as primeiras palavras do abstract como fallback
            words = abstract.split()[:5]
            category = " ".join(words)
    
    return {
        "date": date_iso,
        "year": year,
        "category": category
    }


def extract_abstract(markdown_text: str) -> str:
    """
    Extrai o abstract do artigo.
    """
    # Procura por "## Abstract" ou "Abstract" seguido de texto
    abstract_patterns = [
        r'##\s*Abstract\s*\n(.*?)(?=\n##|\Z)',
        r'Abstract\s*\n(.*?)(?=\n\s*\n|\Z)',
        r'^Abstract[:\s]*(.*?)(?=\n\s*\n|\Z)'
    ]
    
    for pattern in abstract_patterns:
        abstract_match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Remove múltiplas linhas e espaços extras
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract
    
    return ""


def extract_authors(markdown_text: str) -> List[str]:
    """
    Extrai a lista de autores do artigo.
    """
    authors = []
    
    # Procura por linhas que contêm nomes de autores com sobrescritos
    lines = markdown_text.split('\n')[:15]  # Primeiras 15 linhas
    
    for line in lines:
        # Padrão: Nome S \(1\) . Nome2 S \(2\) etc.
        if re.search(r'[A-Za-zÀ-ÿ\s\.]+\s+\(\d+\)', line):
            # Divide por " . " que separa autores no formato
            parts = line.split(' . ')
            for part in parts:
                # Remove números de sobrescrito
                clean_name = re.sub(r'\s+\(\d+\)', '', part).strip()
                if clean_name and len(clean_name) > 3:  # Nomes com mais de 3 caracteres
                    authors.append(clean_name)
    
    return authors if authors else ["Unknown"]


def parse_markdown_structure(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Analisa a estrutura do Markdown e retorna uma lista de nós (seções).
    """
    lines = markdown_text.split('\n')
    sections = []
    
    # Padrão para identificar títulos Markdown
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$')
    
    # Mapeamento de níveis
    level_map = {
        1: 'section',
        2: 'subsection',
        3: 'subsubsection'
    }
    
    for line_num, line in enumerate(lines):
        match = heading_pattern.match(line)
        if match:
            level_symbol = match.group(1)
            title = match.group(2).strip()
            
            # Determina o nível baseado no número de #
            level_count = len(level_symbol)
            level = level_map.get(level_count, 'section')
            
            # Ignora títulos que parecem ser partes do template/não relevantes
            if title.lower() in ['figure', 'table', 'list of figures', 'list of tables']:
                continue
                
            sections.append({
                'id': generate_uuid(),
                'title': title,
                'level': level,
                'line_num': line_num,
                'subsections': [],  # Inicializa subsections vazio
                'parent_id': None
            })
    
    return sections


def build_tree(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Constrói a árvore hierárquica a partir da lista de seções.
    """
    if not sections:
        return {
            'id': generate_uuid(),
            'title': 'root',
            'level': 'root',
            'subsections': [],
            'parent_id': None
        }
    
    # Cria o nó raiz
    root = {
        'id': generate_uuid(),
        'title': 'root',
        'level': 'root',
        'subsections': [],
        'parent_id': None
    }
    
    # Mapa para fácil acesso
    section_map = {s['id']: s for s in sections}
    
    # Primeiro, identifica todas as seções de nível 'section' como filhas diretas da raiz
    section_nodes = [s for s in sections if s['level'] == 'section']
    
    for section in section_nodes:
        section['parent_id'] = root['id']
        root['subsections'].append(section['id'])
    
    # Conecta subseções às suas seções pai
    subsection_nodes = [s for s in sections if s['level'] == 'subsection']
    
    for subsection in subsection_nodes:
        # Encontra a seção pai (última seção de nível superior antes desta)
        parent_section = None
        for i, s in enumerate(sections):
            if s['id'] == subsection['id']:
                # Procura para trás até encontrar uma 'section'
                for j in range(i-1, -1, -1):
                    if sections[j]['level'] == 'section':
                        parent_section = sections[j]
                        break
                break
        
        if parent_section:
            subsection['parent_id'] = parent_section['id']
            parent_section['subsections'].append(subsection['id'])
        else:
            # Se não encontrar pai, coloca diretamente na raiz
            subsection['parent_id'] = root['id']
            root['subsections'].append(subsection['id'])
    
    # Conecta subsubseções às suas subseções pai
    subsubsection_nodes = [s for s in sections if s['level'] == 'subsubsection']
    
    for subsubsection in subsubsection_nodes:
        # Encontra a subseção pai
        parent_subsection = None
        for i, s in enumerate(sections):
            if s['id'] == subsubsection['id']:
                # Procura para trás até encontrar uma 'subsection'
                for j in range(i-1, -1, -1):
                    if sections[j]['level'] == 'subsection':
                        parent_subsection = sections[j]
                        break
                break
        
        if parent_subsection:
            subsubsection['parent_id'] = parent_subsection['id']
            parent_subsection['subsections'].append(subsubsection['id'])
        else:
            # Se não encontrar pai, procura por uma 'section'
            parent_section = None
            for i, s in enumerate(sections):
                if s['id'] == subsubsection['id']:
                    for j in range(i-1, -1, -1):
                        if sections[j]['level'] == 'section':
                            parent_section = sections[j]
                            break
                    break
            
            if parent_section:
                subsubsection['parent_id'] = parent_section['id']
                parent_section['subsections'].append(subsubsection['id'])
            else:
                # Último recurso: coloca na raiz
                subsubsection['parent_id'] = root['id']
                root['subsections'].append(subsubsection['id'])
    
    return root


def find_node_by_id(nodes: List[Dict[str, Any]], node_id: str) -> Optional[Dict[str, Any]]:
    """
    Encontra um nó pelo ID.
    """
    for node in nodes:
        if node['id'] == node_id:
            return node
    return None


def build_prefix_titles(node: Dict[str, Any], all_nodes: List[Dict[str, Any]], root_title: str) -> List[List[str]]:
    """
    Constrói a lista prefix_titles para um nó.
    """
    prefixes = []
    current_id = node['id']
    
    # Constrói o caminho do nó até a raiz
    path = []
    current = node
    
    while current and current.get('parent_id'):
        path.insert(0, (current['level'], current['title']))
        
        # Encontra o pai
        parent = find_node_by_id(all_nodes, current['parent_id'])
        if parent:
            current = parent
        else:
            break
    
    # Adiciona os prefixos na ordem correta
    for level, title in path:
        prefixes.append([level, title])
    
    # Adiciona o título da raiz no início
    prefixes.insert(0, ["title", root_title])
    
    return prefixes


def prepare_json_output(root: Dict[str, Any], sections: List[Dict[str, Any]], 
                        title: str, authors: List[str], abstract: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepara a estrutura final para o JSON.
    """
    # Lista final com todos os nós
    all_nodes = [root] + sections
    
    # Garante que todos os nós tenham os campos necessários
    for node in all_nodes:
        # Remove campos temporários
        if 'line_num' in node:
            del node['line_num']
        
        # Garante que 'subsections' existe
        if 'subsections' not in node:
            node['subsections'] = []
        
        # Adiciona campos obrigatórios
        node['content'] = ""
        node['cites'] = []
        node['cite_extract_rate'] = 0
        node['origin_cites_number'] = 0
        
        # Adiciona prefix_titles
        node['prefix_titles'] = build_prefix_titles(node, all_nodes, title)
    
    # Cria a estrutura final
    output = [{
        "authors": authors,
        "survey_title": title,
        "year": metadata.get("year", ""),
        "date": metadata.get("date", ""),
        "category": metadata.get("category", ""),
        "abstract": abstract,
        "structure": all_nodes,
        "survey_id": 0,
        "all_cites": [],
        "Bertopic_CD": 0
    }]
    
    return output


def process_markdown_to_json(markdown_file_path: str, output_file_path: str) -> None:
    """
    Processa um arquivo Markdown e gera um arquivo JSON com sua estrutura.
    """
    try:
        print(f"Processando arquivo: {markdown_file_path}")
        
        # Lê o arquivo Markdown
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        print("Arquivo lido com sucesso.")
        
        # Extrai informações
        title = extract_title_from_markdown(markdown_text)
        print(f"Título encontrado: {title}")
        
        authors = extract_authors(markdown_text)
        print(f"Autores encontrados: {len(authors)}")
        
        abstract = extract_abstract(markdown_text)
        print(f"Abstract encontrado: {len(abstract)} caracteres")
        
        metadata = extract_metadata(markdown_text)
        print(f"Metadados extraídos: {metadata}")
        
        # Analisa a estrutura
        sections = parse_markdown_structure(markdown_text)
        print(f"Seções encontradas: {len(sections)}")
        
        # Debug: mostra as seções encontradas
        for i, section in enumerate(sections[:10]):  # Mostra apenas as 10 primeiras
            print(f"  Seção {i+1}: {section['level']} - {section['title']}")
        
        # Constrói a árvore hierárquica
        root = build_tree(sections)
        print("Árvore hierárquica construída.")
        
        # Prepara a saída JSON
        output = prepare_json_output(root, sections, title, authors, abstract, metadata)
        print("Estrutura JSON preparada.")
        
        # Salva o arquivo JSON
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Arquivo JSON gerado com sucesso: {output_file_path}")
        
        # Estatísticas
        section_count = len([s for s in sections if s['level'] == 'section'])
        subsection_count = len([s for s in sections if s['level'] == 'subsection'])
        subsubsection_count = len([s for s in sections if s['level'] == 'subsubsection'])
        
        print(f"\nEstatísticas:")
        print(f"  Sections: {section_count}")
        print(f"  Subsections: {subsection_count}")
        print(f"  Subsubsections: {subsubsection_count}")
        print(f"  Total nós na estrutura: {len(output[0]['structure'])}")
        
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Uso: python script.py <arquivo_markdown.md> <arquivo_saida.json>")
        print("Exemplo: python markdown_to_json.py artigo.md saida.json")
        sys.exit(1)
    
    markdown_file = sys.argv[1]
    json_file = sys.argv[2]
    
    process_markdown_to_json(markdown_file, json_file)