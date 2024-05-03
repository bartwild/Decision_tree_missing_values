from graphviz import Source

def convert_dot_to_png(dot_file_path, output_file_path):
    # Wczytanie zawartości pliku DOT
    with open(dot_file_path, 'r') as file:
        dot_content = file.read()

    # Utworzenie źródła Graphviz z zawartości DOT
    dot_graph = Source(dot_content)

    # Renderowanie do formatu PNG
    dot_graph.render(output_file_path, format='png', cleanup=True)
    print(f"Generated PNG file at {output_file_path}.png")

# Przykład użycia
dot_file_path = 'tree-small2.dot'  # Ścieżka do pliku DOT
output_file_path = 'output_image'  # Nazwa pliku wyjściowego bez rozszerzenia
convert_dot_to_png(dot_file_path, output_file_path)
