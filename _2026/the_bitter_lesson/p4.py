from manimlib import *
import json

class Node(VGroup):
    def __init__(self, node_data, radius=0.5, font_size=16, padding=0.1, **kwargs):
        super().__init__(**kwargs)

        self.node_id = node_data["id"]
        self.phoneme = node_data["phoneme"]
        self.connects_from = node_data["connects_from"]

        self.circle = Circle(radius=radius, color=WHITE, stroke_color=WHITE, stroke_width=2)
        self.circle.set_fill(color=BLACK, opacity=1)

        self.text = Text(self.phoneme, font_size=font_size, color=WHITE)

        # Calculate max width allowed (diameter minus padding on both sides)
        max_width = (radius * 2) - (padding * 2)

        # Scale down text if it's too wide
        if self.text.get_width() > max_width:
            scale_factor = max_width / self.text.get_width()
            self.text.scale(scale_factor)

        self.add(self.circle, self.text)


class Connection(VGroup):
    def __init__(self, node_from, node_to, arc_amount=0, **kwargs):
        super().__init__(**kwargs)

        self.node_from = node_from
        self.node_to = node_to

        self.arrow = Arrow(
            node_from.get_center(),
            node_to.get_center(),
            buff=0.55,  # Adjusted for larger nodes
            color=WHITE,
            path_arc=arc_amount,
        )

        self.add(self.arrow)


class TestNode(Scene):
    def construct(self):
        node_data = {
            "id": 0,
            "phoneme": "T",
            "connects_from": []
        }
        node = Node(node_data)
        self.add(node)
        self.wait()


class Network(VGroup):
    def __init__(self, nodes_data, layer_spacing=2.5, node_spacing=1.0, **kwargs):
        super().__init__(**kwargs)

        self.nodes_data = nodes_data
        self.layer_spacing = layer_spacing
        self.node_spacing = node_spacing

        # Create Node objects
        self.nodes = {data["id"]: Node(data) for data in nodes_data}

        # Organize nodes into layers based on connection depth
        self.layers = self._organize_into_layers()

        # Position nodes
        self._position_nodes()

        # Create connections with smart arc routing
        self.connections = []
        self._create_connections()

        # Add connections first (so they appear behind nodes)
        for connection in self.connections:
            self.add(connection)

        # Add nodes to the VGroup
        for node in self.nodes.values():
            self.add(node)

    def _find_all_paths(self):
        """Find all paths from START to END nodes"""
        paths = []

        # Find START nodes
        start_nodes = [node_id for node_id, node in self.nodes.items()
                      if not node.connects_from]

        # Find END nodes (nodes with no outgoing edges)
        outgoing = {node_id: [] for node_id in self.nodes.keys()}
        for node_id, node in self.nodes.items():
            for parent_id in node.connects_from:
                outgoing[parent_id].append(node_id)

        # DFS to find all paths
        def dfs(node_id, current_path):
            current_path.append(node_id)

            if not outgoing[node_id]:  # END node
                paths.append(current_path[:])
            else:
                for child_id in outgoing[node_id]:
                    dfs(child_id, current_path)

            current_path.pop()

        for start_id in start_nodes:
            dfs(start_id, [])

        return paths, outgoing

    def _organize_into_layers(self):
        """Organize nodes by depth for x-coordinate"""
        layers = []
        visited = set()

        # Find root node(s)
        current_layer = [node_id for node_id, node in self.nodes.items()
                        if not node.connects_from]

        while current_layer:
            layers.append(current_layer)
            visited.update(current_layer)

            next_layer = []
            for node_id, node in self.nodes.items():
                if node_id not in visited:
                    if all(parent_id in visited for parent_id in node.connects_from):
                        next_layer.append(node_id)

            current_layer = next_layer

        return layers

    def _position_nodes(self):
        """Position nodes using path-based layout like Graphviz"""
        # Find all paths through the graph
        paths, outgoing = self._find_all_paths()

        # Calculate x-position based on depth (topological order)
        node_depth = {}

        # Process nodes in topological order (using layers)
        for layer_idx, layer in enumerate(self.layers):
            for node_id in layer:
                node_depth[node_id] = layer_idx

        # Assign each path to a y-position (row)
        # Nodes appearing in multiple paths will get averaged y-position
        node_y_positions = {}  # node_id -> list of y positions

        for path_idx, path in enumerate(paths):
            y_pos = path_idx * self.node_spacing
            for node_id in path:
                if node_id not in node_y_positions:
                    node_y_positions[node_id] = []
                node_y_positions[node_id].append(y_pos)

        # Average y-positions for shared nodes
        node_final_y = {node_id: sum(positions) / len(positions)
                       for node_id, positions in node_y_positions.items()}

        # Position nodes
        max_y = max(node_final_y.values()) if node_final_y else 0
        min_y = min(node_final_y.values()) if node_final_y else 0
        y_center = (max_y + min_y) / 2

        for node_id, node in self.nodes.items():
            x_pos = node_depth.get(node_id, 0) * self.layer_spacing
            y_pos = node_final_y.get(node_id, 0) - y_center
            node.move_to([x_pos, y_pos, 0])

    def _create_connections(self):
        # ALL arrows are straight - no curves
        for node_id, node in self.nodes.items():
            for parent_id in node.connects_from:
                parent_node = self.nodes[parent_id]
                target_node = node

                # Completely straight arrows
                connection = Connection(parent_node, target_node, arc_amount=0)
                self.connections.append(connection)


class TestConnection(Scene):
    def construct(self):
        node1_data = {
            "id": 0,
            "phoneme": "T",
            "connects_from": []
        }
        node2_data = {
            "id": 1,
            "phoneme": "E",
            "connects_from": [0]
        }

        node1 = Node(node1_data).shift(LEFT * 2)
        node2 = Node(node2_data).shift(RIGHT * 2)
        connection = Connection(node1, node2)

        self.add(node1, node2, connection)
        self.wait()


class TestNetwork(Scene):
    def construct(self):
        # Load nodes from JSON
        with open("phoneme_dag.json", "r") as f:
            nodes_data = json.load(f)

        # Create network with consistent spacing
        network = Network(nodes_data, layer_spacing=2.5, node_spacing=3.0)

        self.add(network)
        self.wait()
