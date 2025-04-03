import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import openai
from PIL import Image
import os

def generate_hierarchy(topics, theme, terms=None, depth=2, temperature=0.7, model="gpt-4o", num=3):
    """
    Generate hierarchical data using the OpenAI API while preventing duplicate node names.
    """
    hierarchy = {}
    seen_nodes = set()

    def parse_json_response(response_text):
        """ Extract and parse JSON safely from model output """
        try:
            json_data = json.loads(response_text)
            if isinstance(json_data, list):
                return json_data[:num]
        except json.JSONDecodeError:
            pass

        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            try:
                json_data = json.loads(match.group(0))
                if isinstance(json_data, list):
                    return json_data[:num]
            except json.JSONDecodeError:
                pass

        extracted = re.findall(r'^\d+\.\s*(.+)$', response_text, re.MULTILINE) 
        if not extracted:
            extracted = re.findall(r'^\*\s*(.+)$', response_text, re.MULTILINE)

        if extracted:
            return extracted[:num]

        print(f"Warning: Failed to parse valid JSON from response: {response_text}")
        return []

    def expand_topic(topic, parent_dict, level):
        """ Recursively expand a topic into subtopics inside its correct parent node, ensuring uniqueness. """
        if level > depth:
            return

        prompt = f"""
        You are an AI that generates hierarchical topic structures.
        Given a topic and a theme, return a JSON list of exactly {num} unique subtopics that fit the theme.
        Each subtopic should be one that can have an **increasing** or **decreasing** nature.
        The words increasing and decreasing should not be contained within any subtopic.
        The subtopics should be specifically related to the parent topic and not general.
        The subtopic should contain all relevant information to its context.
        Ensure that all subtopics are distinct.

        **Example:**
        Theme:  Irish Economy
        Parent Topic: Potato Production
        Output: ["Potato yield per hectacre", "Use of modern technology in potato farming", "Issues with potato crops due to disease"]

        Theme: {theme}
        Parent Topic: {topic}

        Format your response **only** as a JSON list, e.g.:
        ["Subtopic 1", "Subtopic 2", "Subtopic 3"]
        """

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You generate hierarchical topic structures."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        content = response.choices[0].message.content.strip()

        subtopics = parse_json_response(content)

        parent_dict[topic] = {}
        for subtopic in subtopics:
            if subtopic not in seen_nodes:  # Ensure uniqueness
                seen_nodes.add(subtopic)
                parent_dict[topic][subtopic] = {}
                expand_topic(subtopic, parent_dict[topic], level + 1)
            else:
                print(f"Skipping duplicate subtopic: {subtopic}")

    for topic in topics:
        if topic not in seen_nodes:  # Prevent duplicate top-level topics
            seen_nodes.add(topic)
            expand_topic(topic, hierarchy, 1)

    return hierarchy

def build_hierarchy_graph(hierarchy, graph=None, parent=None):
    """
    Recursively builds a directed graph for hierarchical data visualization.
    
    :param hierarchy: Nested dictionary representing hierarchical data.
    :param graph: NetworkX DiGraph to store the relationships.
    :param parent: Parent node in the graph.
    :return: A directed graph with hierarchical relationships.
    """
    if graph is None:
        graph = nx.DiGraph()

    for topic, subtopics in hierarchy.items():
        graph.add_node(topic)  # Add node
        if parent:
            graph.add_edge(parent, topic)  # Connect to parent

        if isinstance(subtopics, dict) and subtopics:
            build_hierarchy_graph(subtopics, graph, topic)

    return graph

def plot_hierarchy(hierarchy):
    """
    Plots the hierarchy as a tree using Graphviz for a strict hierarchical layout,
    saves it as a PNG file, and opens it using the default image viewer.
    
    :param hierarchy: The hierarchical JSON structure (nested dictionary).
    """
    graph = build_hierarchy_graph(hierarchy)

    plt.figure(figsize=(30, 10))
    pos = graphviz_layout(graph, prog="dot")  # Use Graphviz for strict tree layout
    nx.draw(graph, pos, with_labels=False, node_size=7000, node_color="lightblue",
            edge_color="black", font_size=12, font_weight="bold", arrows=True)

    # Manually draw rotated labels at 45 degrees
    ax = plt.gca()
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=12, rotation=15, ha="center", va="center")

    plt.title("Hierarchical Topic Structure", fontsize=25)

    # Save the plot to a PNG file
    output_file = "hierarchical_plot.png"
    plt.savefig(output_file)

    # Close the plot to avoid it being shown in the background
    plt.close()

    # Open the saved PNG file using the default image viewer
    try:
        # This will work on most systems
        Image.open(output_file).show()
    except Exception as e:
        print(f"Could not open the image. Error: {e}")
        os.system(f"start {output_file}")

        
def generate_hierarchy_gui():
    topics = topics_entry.get().split(',')
    theme = theme_entry.get()
    terms = terms_entry.get()
    depth = int(depth_entry.get())
    temperature = float(temp_entry.get())
    model = model_entry.get()
    num = int(num_entry.get())

    if not topics or not theme:
        messagebox.showerror("Error", "Topics and Theme are required fields.")
        return

    hierarchy = generate_hierarchy(topics, theme, terms, depth, temperature, model, num)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, json.dumps(hierarchy, indent=4))
    plot_hierarchy(hierarchy)

# GUI Setup
root = tk.Tk()
root.title("Hierarchy Generator")

tk.Label(root, text="Topics (comma-separated):").pack()
topics_entry = tk.Entry(root, width=50)
topics_entry.pack()

tk.Label(root, text="Theme:").pack()
theme_entry = tk.Entry(root, width=50)
theme_entry.pack()

tk.Label(root, text="Optional Terms:").pack()
terms_entry = tk.Entry(root, width=50)
terms_entry.pack()

tk.Label(root, text="Depth:").pack()
depth_entry = tk.Entry(root, width=10)
depth_entry.insert(0, "2")
depth_entry.pack()

tk.Label(root, text="Temperature:").pack()
temp_entry = tk.Entry(root, width=10)
temp_entry.insert(0, "0.7")
temp_entry.pack()

tk.Label(root, text="Model:").pack()
model_entry = tk.Entry(root, width=20)
model_entry.insert(0, "gpt-4o")
model_entry.pack()

tk.Label(root, text="Number of Subtopics:").pack()
num_entry = tk.Entry(root, width=10)
num_entry.insert(0, "3")
num_entry.pack()

tk.Button(root, text="Generate Hierarchy", command=generate_hierarchy_gui).pack()

result_text = scrolledtext.ScrolledText(root, width=80, height=20)
result_text.pack()

root.mainloop()
