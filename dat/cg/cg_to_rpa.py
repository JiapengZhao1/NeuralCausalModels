import os
from collections import defaultdict

def parse_cg_file(file_path):
    """
    Parse a .cg file and extract nodes and edges.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    nodes = []
    edges = []
    bidirected_edges = []

    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line == "<NODES>":
            section = "nodes"
            continue
        elif line == "<EDGES>":
            section = "edges"
            continue

        if section == "nodes":
            nodes.append(line)
        elif section == "edges":
            if "<->" in line:
                bidirected_edges.append(tuple(line.split(" <-> ")))
            elif "->" in line:
                edges.append(tuple(line.split(" -> ")))

    return nodes, edges, bidirected_edges


def convert_to_rpa(nodes, edges, bidirected_edges):
    """
    Convert nodes, edges, and bidirected edges to RPA format.
    """
    rpa = defaultdict(list)

    # Add directed edges
    for parent, child in edges:
        rpa[child].append(parent)

    # Add bidirected edges
    for node1, node2 in bidirected_edges:
        rpa[node2].append((node1,))
        #rpa[node1].append((node2,))

    # Ensure all nodes are in the RPA dictionary
    for node in nodes:
        if node not in rpa:
            rpa[node] = []

    return dict(rpa)


def process_cg_files(cg_files):
    """
    Process multiple .cg files and output their RPA formats.
    """
    rpa_results = {}
    for cg_file in cg_files:
        graph_name = os.path.splitext(os.path.basename(cg_file))[0]
        nodes, edges, bidirected_edges = parse_cg_file(cg_file)
        rpa = convert_to_rpa(nodes, edges, bidirected_edges)
        rpa_results[graph_name] = rpa
    return rpa_results


def format_rpa_output(rpa_results):
    """
    Format the RPA results into the desired output style.
    """
    formatted_output = []
    for graph_name, rpa in rpa_results.items():
        formatted_rpa = ", ".join(
            f"{key}={value}" for key, value in rpa.items()
        )
        formatted_output.append(f"'{graph_name}': dict({formatted_rpa}),")
    return "\n".join(formatted_output)


# List of .cg files
cg_files = [
    "/home/NeuralCausalModels/dat/cg/exp1.cg",
    "/home/NeuralCausalModels/dat/cg/exp2.cg",
    "/home/NeuralCausalModels/dat/cg/exp3.cg",
    "/home/NeuralCausalModels/dat/cg/exp4.cg",
    "/home/NeuralCausalModels/dat/cg/exp5.cg",
    "/home/NeuralCausalModels/dat/cg/exp6.cg",
    "/home/NeuralCausalModels/dat/cg/exp7.cg",
    "/home/NeuralCausalModels/dat/cg/exp8.cg",
]

# Process the .cg files and print the RPA formats
rpa_results = process_cg_files(cg_files)
formatted_output = format_rpa_output(rpa_results)
print(formatted_output)